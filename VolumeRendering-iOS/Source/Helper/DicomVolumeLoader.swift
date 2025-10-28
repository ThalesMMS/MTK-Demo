import Foundation
import simd
import ZIPFoundation

protocol DicomSeriesLoading: AnyObject {
    func loadSeries(at url: URL,
                    progress: ((Double, UInt, Data?, DICOMSeriesVolume) -> Void)?) throws -> DICOMSeriesVolume
}

extension DICOMSeriesLoader: DicomSeriesLoading {}

enum DicomVolumeLoaderError: Error {
    case securityScopeUnavailable
    case unsupportedBitDepth
    case missingResult
    case bridgeError(NSError)
}

struct DicomImportResult {
    let dataset: VolumeDataset
    let sourceURL: URL
    let seriesDescription: String
}

final class DicomVolumeLoader {

    private let loader: DicomSeriesLoading

    init(seriesLoader: DicomSeriesLoading = DICOMSeriesLoader()) {
        self.loader = seriesLoader
    }

    enum ProgressUpdate {
        case started(totalSlices: Int, previewTarget: Int)
        case reading(Double)
        case partialPreview(dataset: VolumeDataset, fraction: Double)
    }

    enum UIProgressUpdate {
        case started(totalSlices: Int, previewTarget: Int)
        case reading(Double)
        case previewAvailable(Double)
    }

    private let previewSliceCap = 64

    private struct PreparedDirectory {
        let url: URL
        let cleanupRoot: URL?
    }

    func loadVolume(from url: URL,
                    progress: @escaping (ProgressUpdate) -> Void,
                    completion: @escaping (Result<DicomImportResult, Error>) -> Void) {
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let prepared = try self.prepareDirectory(from: url)
                let directoryURL = prepared.url

                var convertedData: Data?
                var dimensions = int3(0, 0, 0)
                var spacing = float3.zero
                var orientation = matrix_identity_float3x3
                var origin = float3.zero
                var slope: Double = 1.0
                var intercept: Double = 0.0
                var isSigned = false
                var minHU = Int32.max
                var maxHU = Int32.min
                var previewSent = false
                var previewTarget = 0
                var encounteredFatalError = false

                let volume = try self.loader.loadSeries(at: directoryURL, progress: { fraction, slicesLoaded, sliceData, partialVolume in
                    if encounteredFatalError { return }

                    if convertedData == nil {
                        if partialVolume.bitsAllocated != 16 {
                            encounteredFatalError = true
                            DispatchQueue.main.async {
                                completion(.failure(DicomVolumeLoaderError.unsupportedBitDepth))
                            }
                            return
                        }

                        dimensions = int3(Int32(partialVolume.width), Int32(partialVolume.height), Int32(partialVolume.depth))
                        let meterScale: Float = 0.001
                        spacing = float3(Float(partialVolume.spacingX) * meterScale,
                                         Float(partialVolume.spacingY) * meterScale,
                                         Float(partialVolume.spacingZ) * meterScale)
                        orientation = partialVolume.orientation
                        origin = SIMD3<Float>(partialVolume.origin) * meterScale
                        slope = partialVolume.rescaleSlope == 0 ? 1.0 : partialVolume.rescaleSlope
                        intercept = partialVolume.rescaleIntercept
                        isSigned = partialVolume.isSignedPixel
                        let voxelCount = Int(partialVolume.width) * Int(partialVolume.height) * Int(partialVolume.depth)
                        convertedData = Data(count: voxelCount * MemoryLayout<Int16>.size)
                        previewTarget = self.previewSliceThreshold(depth: Int(partialVolume.depth))
                        DispatchQueue.main.async {
                            progress(.started(totalSlices: Int(partialVolume.depth), previewTarget: previewTarget))
                        }
                    }

                    guard convertedData != nil else { return }

                    if let sliceData = sliceData {
                        convertedData!.withUnsafeMutableBytes { destBuffer in
                            guard let destPtr = destBuffer.bindMemory(to: Int16.self).baseAddress else { return }
                            let sliceVoxelCount = Int(dimensions.x) * Int(dimensions.y)
                            let sliceIndex = max(Int(slicesLoaded) - 1, 0)
                            let offset = sliceIndex * sliceVoxelCount
                            sliceData.withUnsafeBytes { rawBuffer in
                                if isSigned {
                                    let source = rawBuffer.bindMemory(to: Int16.self)
                                    for index in 0..<sliceVoxelCount {
                                        let rawValue = Int32(source[index])
                                        let huDouble = Double(rawValue) * slope + intercept
                                        let huRounded = Int32(lround(huDouble))
                                        minHU = min(minHU, huRounded)
                                        maxHU = max(maxHU, huRounded)
                                        destPtr[offset + index] = Self.clampedHU(huRounded)
                                    }
                                } else {
                                    let source = rawBuffer.bindMemory(to: UInt16.self)
                                    for index in 0..<sliceVoxelCount {
                                        let rawValue = Int32(source[index])
                                        let huDouble = Double(rawValue) * slope + intercept
                                        let huRounded = Int32(lround(huDouble))
                                        minHU = min(minHU, huRounded)
                                        maxHU = max(maxHU, huRounded)
                                        destPtr[offset + index] = Self.clampedHU(huRounded)
                                    }
                                }
                            }
                        }
                    }

                    if !previewSent, let data = convertedData, previewTarget > 0, slicesLoaded >= previewTarget {
                        previewSent = true
                        let sliceVoxelCount = Int(dimensions.x) * Int(dimensions.y)
                        let previewBytes = previewTarget * sliceVoxelCount * MemoryLayout<Int16>.size
                        let previewData = data.prefix(previewBytes)
                        let previewRange = Self.intensityRange(minHU: minHU, maxHU: maxHU)
                        let previewDataset = VolumeDataset(data: previewData,
                                                           dimensions: int3(dimensions.x, dimensions.y, Int32(previewTarget)),
                                                           spacing: spacing,
                                                           pixelFormat: .int16Signed,
                                                           intensityRange: previewRange,
                                                           orientation: orientation,
                                                           origin: origin)
                        DispatchQueue.main.async {
                            progress(.partialPreview(dataset: previewDataset, fraction: fraction))
                        }
                    }

                    DispatchQueue.main.async {
                        progress(.reading(fraction))
                    }
                })

                if encounteredFatalError {
                    if let cleanupRoot = prepared.cleanupRoot {
                        try? FileManager.default.removeItem(at: cleanupRoot)
                    }
                    return
                }

                guard let convertedData else {
                    throw DicomVolumeLoaderError.missingResult
                }

                let range = Self.intensityRange(minHU: minHU, maxHU: maxHU)
                let dataset = VolumeDataset(data: convertedData,
                                             dimensions: dimensions,
                                             spacing: spacing,
                                             pixelFormat: .int16Signed,
                                             intensityRange: range,
                                             orientation: orientation,
                                             origin: origin)

                if let cleanupRoot = prepared.cleanupRoot {
                    try? FileManager.default.removeItem(at: cleanupRoot)
                }

                let result = DicomImportResult(dataset: dataset,
                                               sourceURL: url,
                                               seriesDescription: volume.seriesDescription)

                DispatchQueue.main.async {
                    completion(.success(result))
                }
            } catch let error as NSError {
                DispatchQueue.main.async {
                    completion(.failure(DicomVolumeLoaderError.bridgeError(error)))
                }
            } catch {
                DispatchQueue.main.async {
                    completion(.failure(error))
                }
            }
        }
    }

    private func prepareDirectory(from url: URL) throws -> PreparedDirectory {
        if url.hasDirectoryPath {
            return PreparedDirectory(url: url, cleanupRoot: nil)
        }

        if url.pathExtension.lowercased() == "zip" {
            return try unzip(url: url)
        }

        // Assume individual file inside a directory; use parent directory.
        return PreparedDirectory(url: url.deletingLastPathComponent(), cleanupRoot: nil)
    }

    private func unzip(url: URL) throws -> PreparedDirectory {
        let temporaryDirectory = URL(fileURLWithPath: NSTemporaryDirectory(),
                                     isDirectory: true).appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: temporaryDirectory, withIntermediateDirectories: true)

        guard let archive = Archive(url: url, accessMode: .read) else {
            throw DicomVolumeLoaderError.missingResult
        }

        for entry in archive {
            let destinationURL = temporaryDirectory.appendingPathComponent(entry.path)
            let destinationDir = destinationURL.deletingLastPathComponent()
            try FileManager.default.createDirectory(at: destinationDir, withIntermediateDirectories: true)
            _ = try archive.extract(entry, to: destinationURL)
        }

        // If the archive expands to a single directory, dive into it for cleanliness.
        let contents = try FileManager.default.contentsOfDirectory(at: temporaryDirectory,
                                                                   includingPropertiesForKeys: [.isDirectoryKey],
                                                                   options: [.skipsHiddenFiles])
        if contents.count == 1, (try contents.first?.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true {
            return PreparedDirectory(url: contents[0], cleanupRoot: temporaryDirectory)
        }

        return PreparedDirectory(url: temporaryDirectory, cleanupRoot: temporaryDirectory)
    }
}

private extension DicomVolumeLoader {
    static func clampedHU(_ value: Int32) -> Int16 {
        let clampMin: Int32 = -1024
        let clampMax: Int32 = 3071
        let huClamped = max(clampMin, min(clampMax, value))
        let int16Min = Int32(Int16.min)
        let int16Max = Int32(Int16.max)
        let clamped = max(int16Min, min(int16Max, huClamped))
        return Int16(clamped)
    }

    static func intensityRange(minHU: Int32, maxHU: Int32) -> ClosedRange<Int32> {
        var minHU = minHU
        var maxHU = maxHU
        let clampMin: Int32 = -1024
        let clampMax: Int32 = 3071
        if minHU > maxHU {
            minHU = clampMin
            maxHU = clampMax
        } else {
            minHU = max(minHU, clampMin)
            maxHU = min(maxHU, clampMax)
        }
        return minHU...maxHU
    }

    func previewSliceThreshold(depth: Int) -> Int {
        guard depth > 0 else { return 0 }
        let target = max(1, depth / 8)
        return min(target, previewSliceCap)
    }

}

extension DicomVolumeLoader {
    /// Public shim to translate internal loader progress into UI-friendly events.
    static func uiUpdate(from update: ProgressUpdate) -> UIProgressUpdate {
        switch update {
        case .started(let total, let preview):
            return .started(totalSlices: total, previewTarget: preview)
        case .reading(let fraction):
            return .reading(fraction)
        case .partialPreview(_, let fraction):
            return .previewAvailable(fraction)
        }
    }
}
