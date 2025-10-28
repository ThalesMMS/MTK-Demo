@testable import VolumeRendering_iOS
import XCTest
import simd
import Foundation
import Metal

enum DicomTestSupport {
    private static var cache: [URL: DicomImportResult] = [:]

    private static func projectRoot(callerFilePath: StaticString = #filePath) -> URL {
        URL(fileURLWithPath: "\(callerFilePath)")
            .deletingLastPathComponent() // Tests directory
            .deletingLastPathComponent() // VolumeRendering-iOS root
    }

    static func appResourcesRoot(callerFilePath: StaticString = #filePath) -> URL {
        projectRoot(callerFilePath: callerFilePath)
            .appendingPathComponent("VolumeRendering-iOS", isDirectory: true)
    }

    static func examplesRoot(callerFilePath: StaticString = #filePath) -> URL {
        projectRoot(callerFilePath: callerFilePath)
            .appendingPathComponent("DICOM_Example", isDirectory: true)
    }

    static func seriesURL(callerFilePath: StaticString = #filePath) -> URL {
        examplesRoot(callerFilePath: callerFilePath).appendingPathComponent("dicom_series_example.zip")
    }

    static func sliceURL(callerFilePath: StaticString = #filePath) -> URL {
        examplesRoot(callerFilePath: callerFilePath).appendingPathComponent("dicom_slice_example.dcm")
    }

    static func cachedResult(for url: URL) -> DicomImportResult? {
        cache[url.standardizedFileURL]
    }

    static func store(result: DicomImportResult, for url: URL) {
        cache[url.standardizedFileURL] = result
    }

    static func transferFunctionURL(named name: String,
                                    callerFilePath: StaticString = #filePath) -> URL {
        appResourcesRoot(callerFilePath: callerFilePath)
            .appendingPathComponent("Resource", isDirectory: true)
            .appendingPathComponent("TransferFunction", isDirectory: true)
            .appendingPathComponent("\(name).tf", isDirectory: false)
    }
}

extension XCTestCase {
    func loadDicomImportResult(at url: URL,
                               timeout: TimeInterval = 60,
                               file: StaticString = #filePath,
                               line: UInt = #line) throws -> DicomImportResult {
        if let cached = DicomTestSupport.cachedResult(for: url) {
            return cached
        }

        let expectation = expectation(description: "Load DICOM at \(url.lastPathComponent)")
        let loader = DicomVolumeLoader()

        var captured: Result<DicomImportResult, Error>?
        loader.loadVolume(from: url, progress: { _ in }, completion: { result in
            captured = result
            expectation.fulfill()
        })

        wait(for: [expectation], timeout: timeout)

        guard let captured else {
            XCTFail("Loader did not produce a result for \(url)", file: file, line: line)
            throw NSError(domain: "DicomTestSupport",
                          code: 0,
                          userInfo: [NSLocalizedDescriptionKey: "Missing loader result"])
        }

        switch captured {
        case .success(let success):
            DicomTestSupport.store(result: success, for: url)
            return success
        case .failure(let error):
            if case DicomVolumeLoaderError.bridgeError(let nsError) = error,
               nsError.domain == DICOMSeriesLoaderErrorDomain,
               nsError.code == DICOMSeriesLoaderErrorUnavailable {
                throw XCTSkip("GDCM bridge unavailable – skipping DICOM-dependent test.")
            }
            throw error
        }
    }
}

final class DicomVolumeLoaderTests: XCTestCase {

    private let seriesZipName = "dicom_series_example.zip"
    private let sliceName = "dicom_slice_example.dcm"

    func testLoadZippedSeriesProducesExpectedVolume() throws {
        let url = DicomTestSupport.seriesURL()
        let loader = DicomVolumeLoader()
        let expectation = expectation(description: "Zipped DICOM series loaded")

        var result: DicomImportResult?
        var startedEvent: (total: Int, preview: Int)?
        var previewEvent: (dataset: VolumeDataset, fraction: Double)?
        var failure: Error?

        loader.loadVolume(from: url, progress: { update in
            switch update {
            case .started(let total, let preview):
                startedEvent = (total, preview)
            case .partialPreview(let dataset, let fraction):
                previewEvent = (dataset, fraction)
            case .reading:
                break
            }
        }, completion: { outcome in
            switch outcome {
            case .success(let success):
                result = success
            case .failure(let error):
                failure = error
            }
            expectation.fulfill()
        })

        wait(for: [expectation], timeout: 45.0)

        if let failure {
            if case DicomVolumeLoaderError.bridgeError(let nsError) = failure,
               nsError.domain == DICOMSeriesLoaderErrorDomain,
               nsError.code == DICOMSeriesLoaderErrorUnavailable {
                throw XCTSkip("Native DICOM loader unavailable – skipping zipped series test.")
            }
            XCTFail("Unexpected failure loading zipped DICOM series: \(failure)")
            return
        }

        guard let result else {
            XCTFail("Expected successful DICOM import result.")
            return
        }

        guard let started = startedEvent else {
            XCTFail("Did not receive progress start update.")
            return
        }
        XCTAssertEqual(started.total, 211)
        XCTAssertEqual(started.preview, 26)

        if let preview = previewEvent {
            XCTAssertEqual(preview.dataset.dimensions.x, 512)
            XCTAssertEqual(preview.dataset.dimensions.y, 512)
            XCTAssertEqual(preview.dataset.dimensions.z, 26)
            XCTAssertEqual(preview.dataset.intensityRange.lowerBound, -1024)
            XCTAssertEqual(preview.dataset.intensityRange.upperBound, 3071)
            XCTAssertGreaterThan(preview.fraction, 0.0)
            XCTAssertLessThan(preview.fraction, 1.0)
        } else {
            XCTFail("Expected to receive a partial preview update.")
        }

        let dataset = result.dataset
        XCTAssertEqual(dataset.dimensions.x, 512)
        XCTAssertEqual(dataset.dimensions.y, 512)
        XCTAssertEqual(dataset.dimensions.z, 211)
        XCTAssertEqual(dataset.pixelFormat, .int16Signed)
        XCTAssertEqual(dataset.data.count, dataset.voxelCount * MemoryLayout<Int16>.size)
        XCTAssertEqual(dataset.intensityRange.lowerBound, -1024)
        XCTAssertEqual(dataset.intensityRange.upperBound, 3071)
        let expectedSpacing = float3(0.000488, 0.000488, 0.001)
        assert(spacing: dataset.spacing, isApproximately: expectedSpacing)
        assert(orientation: dataset.orientation,
               row: SIMD3<Float>(1, 0, 0),
               column: SIMD3<Float>(0, 1, 0),
               normal: SIMD3<Float>(0, 0, 1))
        assert(origin: dataset.origin, isApproximately: SIMD3<Float>(-0.124755, -0.1366308, -0.552))
        XCTAssertEqual(result.seriesDescription.trimmingCharacters(in: .whitespacesAndNewlines), "P. MOLES Head 2.0")
    }

    func testLoadSingleSliceDirectoryProducesDepthOneVolume() throws {
        let source = DicomTestSupport.sliceURL()
        let temporaryDirectory = try createTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: temporaryDirectory) }
        let destination = temporaryDirectory.appendingPathComponent(sliceName)
        try FileManager.default.copyItem(at: source, to: destination)

        let loader = DicomVolumeLoader()
        let expectation = expectation(description: "Single DICOM slice loaded")

        var result: DicomImportResult?
        var startedEvent: (total: Int, preview: Int)?
        var previewEvent: (dataset: VolumeDataset, fraction: Double)?
        var failure: Error?

        loader.loadVolume(from: temporaryDirectory, progress: { update in
            switch update {
            case .started(let total, let preview):
                startedEvent = (total, preview)
            case .partialPreview(let dataset, let fraction):
                previewEvent = (dataset, fraction)
            case .reading:
                break
            }
        }, completion: { outcome in
            switch outcome {
            case .success(let success):
                result = success
            case .failure(let error):
                failure = error
            }
            expectation.fulfill()
        })

        wait(for: [expectation], timeout: 20.0)

        if let failure {
            if case DicomVolumeLoaderError.bridgeError(let nsError) = failure,
               nsError.domain == DICOMSeriesLoaderErrorDomain,
               nsError.code == DICOMSeriesLoaderErrorUnavailable {
                throw XCTSkip("Native DICOM loader unavailable – skipping single slice test.")
            }
            XCTFail("Unexpected failure loading single DICOM slice: \(failure)")
            return
        }

        guard let result else {
            XCTFail("Expected successful DICOM import result.")
            return
        }

        guard let started = startedEvent else {
            XCTFail("Did not receive progress start update.")
            return
        }
        XCTAssertEqual(started.total, 1)
        XCTAssertEqual(started.preview, 1)

        if let preview = previewEvent {
            XCTAssertEqual(preview.dataset.dimensions.x, 512)
            XCTAssertEqual(preview.dataset.dimensions.y, 512)
            XCTAssertEqual(preview.dataset.dimensions.z, 1)
            XCTAssertEqual(preview.dataset.intensityRange.lowerBound, -1024)
            XCTAssertEqual(preview.dataset.intensityRange.upperBound, 1718)
            XCTAssertEqual(preview.fraction, 1.0, accuracy: 1e-6)
        } else {
            XCTFail("Expected to receive a partial preview update.")
        }

        let dataset = result.dataset
        XCTAssertEqual(dataset.dimensions.x, 512)
        XCTAssertEqual(dataset.dimensions.y, 512)
        XCTAssertEqual(dataset.dimensions.z, 1)
        XCTAssertEqual(dataset.pixelFormat, .int16Signed)
        XCTAssertEqual(dataset.data.count, dataset.voxelCount * MemoryLayout<Int16>.size)
        XCTAssertEqual(dataset.intensityRange.lowerBound, -1024)
        XCTAssertEqual(dataset.intensityRange.upperBound, 1718)
        let expectedSpacing = float3(0.000488, 0.000488, 0.001)
        assert(spacing: dataset.spacing, isApproximately: expectedSpacing)
        assert(orientation: dataset.orientation,
               row: SIMD3<Float>(1, 0, 0),
               column: SIMD3<Float>(0, 1, 0),
               normal: SIMD3<Float>(0, 0, 1))
        assert(origin: dataset.origin, isApproximately: SIMD3<Float>(-0.124755, -0.1366308, -0.449))
        XCTAssertEqual(result.seriesDescription.trimmingCharacters(in: .whitespacesAndNewlines), "P. MOLES Head 2.0")
    }

    func testLoadVolumePropagatesBridgeErrors() throws {
        let directory = try createTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let underlyingError = NSError(domain: DICOMSeriesLoaderErrorDomain,
                                      code: DICOMSeriesLoaderErrorUnsupportedFormat,
                                      userInfo: [NSLocalizedDescriptionKey: "Stub unsupported syntax"])
        let loader = DicomVolumeLoader(seriesLoader: ThrowingSeriesLoader(error: underlyingError))
        let expectation = expectation(description: "Bridge error propagated")

        var captured: Error?
        loader.loadVolume(from: directory, progress: { _ in }) { result in
            if case .failure(let error) = result {
                captured = error
            }
            expectation.fulfill()
        }

        wait(for: [expectation], timeout: 1.0)
        guard let captured else {
            XCTFail("Expected failure for bridge error propagation.")
            return
        }

        guard case DicomVolumeLoaderError.bridgeError(let nsError) = captured else {
            XCTFail("Expected bridgeError wrapping original NSError, got \(captured).")
            return
        }

        XCTAssertEqual(nsError.domain, underlyingError.domain)
        XCTAssertEqual(nsError.code, underlyingError.code)
        XCTAssertEqual(nsError.localizedDescription, underlyingError.localizedDescription)
    }

    func testLoadVolumeFailsForUnsupportedBitDepth() throws {
        let directory = try createTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let loader = DicomVolumeLoader(seriesLoader: UnsupportedBitDepthSeriesLoader())
        let expectation = expectation(description: "Unsupported bit depth reported")

        var captured: Error?
        loader.loadVolume(from: directory, progress: { _ in }) { result in
            if case .failure(let error) = result {
                captured = error
            }
            expectation.fulfill()
        }

        wait(for: [expectation], timeout: 1.0)
        guard let captured else {
            XCTFail("Expected failure for unsupported bit depth.")
            return
        }

        guard case DicomVolumeLoaderError.unsupportedBitDepth = captured else {
            XCTFail("Expected unsupportedBitDepth error, got \(captured).")
            return
        }
    }
}

private extension DicomVolumeLoaderTests {
    // MARK: - Helpers

    func makePerspectiveMatrix(fovY: Float,
                               aspect: Float,
                               nearZ: Float,
                               farZ: Float) -> simd_float4x4 {
        let yScale = 1.0 / tanf(fovY * 0.5)
        let xScale = yScale / max(aspect, 1.0e-6)
        let zRange = farZ - nearZ
        let zScale = -(farZ + nearZ) / zRange
        let wzScale = -(2 * farZ * nearZ) / zRange

        return simd_float4x4(columns: (
            SIMD4<Float>(xScale, 0, 0, 0),
            SIMD4<Float>(0, yScale, 0, 0),
            SIMD4<Float>(0, 0, zScale, -1),
            SIMD4<Float>(0, 0, wzScale, 0)
        ))
    }

    func makeLookAtMatrix(eye: SIMD3<Float>,
                          center: SIMD3<Float>,
                          up: SIMD3<Float>) -> simd_float4x4 {
        let forward = simd_normalize(eye - center)
        let right = simd_normalize(simd_cross(up, forward))
        let trueUp = simd_cross(forward, right)

        let translation = SIMD3<Float>(
            -simd_dot(right, eye),
            -simd_dot(trueUp, eye),
            -simd_dot(forward, eye)
        )

        return simd_float4x4(columns: (
            SIMD4<Float>(right.x, trueUp.x, forward.x, 0),
            SIMD4<Float>(right.y, trueUp.y, forward.y, 0),
            SIMD4<Float>(right.z, trueUp.z, forward.z, 0),
            SIMD4<Float>(translation.x, translation.y, translation.z, 1)
        ))
    }

    func createTemporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    func assert(spacing: float3, isApproximately expected: float3, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(Double(spacing.x), Double(expected.x), accuracy: 1e-6, file: file, line: line)
        XCTAssertEqual(Double(spacing.y), Double(expected.y), accuracy: 1e-6, file: file, line: line)
        XCTAssertEqual(Double(spacing.z), Double(expected.z), accuracy: 1e-6, file: file, line: line)
    }

    func assert(orientation: simd_float3x3,
                row: SIMD3<Float>,
                column: SIMD3<Float>,
                normal: SIMD3<Float>,
                file: StaticString = #filePath,
                line: UInt = #line) {
        assert(vector: orientation.columns.0, matches: row, tolerance: 1e-4, file: file, line: line)
        assert(vector: orientation.columns.1, matches: column, tolerance: 1e-4, file: file, line: line)
        assert(vector: orientation.columns.2, matches: normal, tolerance: 1e-4, file: file, line: line)
    }

    func assert(origin: SIMD3<Float>, isApproximately expected: SIMD3<Float>, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(Double(origin.x), Double(expected.x), accuracy: 1e-6, file: file, line: line)
        XCTAssertEqual(Double(origin.y), Double(expected.y), accuracy: 1e-6, file: file, line: line)
        XCTAssertEqual(Double(origin.z), Double(expected.z), accuracy: 1e-6, file: file, line: line)
    }

    func assert(vector: SIMD3<Float>, matches expected: SIMD3<Float>, tolerance: Float, file: StaticString, line: UInt) {
        XCTAssertLessThanOrEqual(abs(vector.x - expected.x), tolerance, file: file, line: line)
        XCTAssertLessThanOrEqual(abs(vector.y - expected.y), tolerance, file: file, line: line)
        XCTAssertLessThanOrEqual(abs(vector.z - expected.z), tolerance, file: file, line: line)
    }

    func makeSyntheticDataset(size: Int, lowerValue: Int16, upperValue: Int16) -> VolumeDataset {
        precondition(size > 0)
        let voxelCount = size * size * size
        var data = Data(count: voxelCount * MemoryLayout<Int16>.size)
        let center = SIMD3<Float>(Float(size) / 2)
        let innerRadius = Float(size) * 0.3
        let innerRadiusSq = innerRadius * innerRadius
        let outerRadius = Float(size) * 0.45
        let outerRadiusSq = outerRadius * outerRadius

        data.withUnsafeMutableBytes { buffer in
            guard let base = buffer.bindMemory(to: Int16.self).baseAddress else { return }
            for z in 0..<size {
                for y in 0..<size {
                    for x in 0..<size {
                        let idx = z * size * size + y * size + x
                        let pos = SIMD3<Float>(Float(x), Float(y), Float(z))
                        let distanceSq = simd_length_squared(pos - center)
                        if distanceSq <= innerRadiusSq {
                            base[idx] = upperValue
                        } else if distanceSq <= outerRadiusSq {
                            base[idx] = Int16((Int(lowerValue) + Int(upperValue)) / 2)
                        } else {
                            base[idx] = lowerValue
                        }
                    }
                }
            }
        }

        let minValue = Int32(min(lowerValue, upperValue))
        let maxValue = Int32(max(lowerValue, upperValue))
        return VolumeDataset(data: data,
                             dimensions: int3(Int32(size), Int32(size), Int32(size)),
                             spacing: float3(repeating: 1.0 / Float(size)),
                             pixelFormat: .int16Signed,
                             intensityRange: minValue...maxValue)
    }

    func makeLinearTransferTexture(device: MTLDevice, sampleCount: Int = 256) throws -> MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float,
                                                                  width: sampleCount,
                                                                  height: 1,
                                                                  mipmapped: false)
        descriptor.usage = [.shaderRead]
        descriptor.storageMode = .shared

        guard let texture = device.makeTexture(descriptor: descriptor) else {
            throw NSError(domain: "DicomVolumeLoaderTests",
                          code: -1,
                          userInfo: [NSLocalizedDescriptionKey: "Failed to allocate synthetic transfer texture."])
        }

        var lut = [SIMD4<Float>](repeating: .zero, count: sampleCount)
        for index in 0..<sampleCount {
            let t = Float(index) / Float(max(sampleCount - 1, 1))
            lut[index] = SIMD4<Float>(repeating: t)
        }

        lut.withUnsafeBytes { rawBuffer in
            texture.replace(region: MTLRegionMake2D(0, 0, sampleCount, 1),
                            mipmapLevel: 0,
                            withBytes: rawBuffer.baseAddress!,
                            bytesPerRow: MemoryLayout<SIMD4<Float>>.stride * sampleCount)
        }

        texture.label = "SyntheticLinearTF"
        return texture
    }

    func sampleCenterPixel(fromBGRA data: Data, width: Int, height: Int) -> SIMD4<Float> {
        precondition(data.count == width * height * 4)
        let x = width / 2
        let y = height / 2
        let index = (y * width + x) * 4
        return data.withUnsafeBytes { rawBuffer -> SIMD4<Float> in
            let base = rawBuffer.bindMemory(to: UInt8.self).baseAddress!
            let b = Float(base[index]) / 255.0
            let g = Float(base[index + 1]) / 255.0
            let r = Float(base[index + 2]) / 255.0
            let a = Float(base[index + 3]) / 255.0
            return SIMD4<Float>(r, g, b, a)
        }
    }

    final class ThrowingSeriesLoader: DicomSeriesLoading {
        private let error: NSError

        init(error: NSError) {
            self.error = error
        }

        func loadSeries(at url: URL,
                        progress: ((Double, UInt, Data?, DICOMSeriesVolume) -> Void)?) throws -> DICOMSeriesVolume {
            throw error
        }
    }

    final class UnsupportedBitDepthSeriesLoader: DicomSeriesLoading {
        func loadSeries(at url: URL,
                        progress: ((Double, UInt, Data?, DICOMSeriesVolume) -> Void)?) throws -> DICOMSeriesVolume {
            let voxelCount = 4
            let voxels = NSMutableData(length: voxelCount * MemoryLayout<UInt16>.size) ?? NSMutableData()
            let orientation = matrix_identity_float3x3
            let origin = vector_float3(0, 0, 0)
            let volume = DICOMSeriesVolume(mutableVoxels: voxels,
                                           width: 2,
                                           height: 2,
                                           depth: 1,
                                           spacingX: 1.0,
                                           spacingY: 1.0,
                                           spacingZ: 1.0,
                                           rescaleSlope: 1.0,
                                           rescaleIntercept: 0.0,
                                           bitsAllocated: 8,
                                           signedPixel: false,
                                           seriesDescription: "Unsupported Bit Depth",
                                           orientation: orientation,
                                           origin: origin)
            progress?(0.1, 1, nil, volume)
            return volume
        }
    }
}
