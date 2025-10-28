@testable import VolumeRendering_iOS
import XCTest

final class VolumeCubeMaterialDicomTests: XCTestCase {

    func testSetDatasetUpdatesUniformsAndTextureForDicomVolume() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal indisponível neste dispositivo de teste.")
        }

        let featureFlags = FeatureFlags.evaluate(for: device)
        let importResult = try loadDicomImportResult(at: DicomTestSupport.seriesURL())
        let dataset = importResult.dataset

        let material = VolumeCubeMaterial(device: device, featureFlags: featureFlags)
        material.setDataset(device: device, dataset: dataset, completion: nil)

        XCTAssertTrue(waitUntil(timeout: 45.0) {
            material.currentVolumeTexture() != nil
        }, "Timeout aguardando upload do volume DICOM.")

        guard let texture = material.currentVolumeTexture() else {
            XCTFail("Expected volume texture after dataset upload.")
            return
        }

        XCTAssertEqual(texture.width, Int(dataset.dimensions.x))
        XCTAssertEqual(texture.height, Int(dataset.dimensions.y))
        XCTAssertEqual(texture.depth, Int(dataset.dimensions.z))

        let uniforms = material.snapshotUniforms()
        XCTAssertEqual(uniforms.dimX, dataset.dimensions.x)
        XCTAssertEqual(uniforms.dimY, dataset.dimensions.y)
        XCTAssertEqual(uniforms.dimZ, dataset.dimensions.z)
        XCTAssertEqual(uniforms.voxelMinValue, dataset.intensityRange.lowerBound)
        XCTAssertEqual(uniforms.voxelMaxValue, dataset.intensityRange.upperBound)
        XCTAssertEqual(uniforms.datasetMinValue, dataset.intensityRange.lowerBound)
        XCTAssertEqual(uniforms.datasetMaxValue, dataset.intensityRange.upperBound)

        let expectedScale = dataset.scale
        XCTAssertEqual(material.scale.x, expectedScale.x, accuracy: 1e-6)
        XCTAssertEqual(material.scale.y, expectedScale.y, accuracy: 1e-6)
        XCTAssertEqual(material.scale.z, expectedScale.z, accuracy: 1e-6)

        let orientation = dataset.orientation
        let transform = material.transform
        let basisX = SIMD3<Float>(transform.columns.0.x, transform.columns.0.y, transform.columns.0.z)
        let basisY = SIMD3<Float>(transform.columns.1.x, transform.columns.1.y, transform.columns.1.z)
        let basisZ = SIMD3<Float>(transform.columns.2.x, transform.columns.2.y, transform.columns.2.z)

        XCTAssertTrue(isApproximatelyEqual(basisX, orientation.columns.0 * expectedScale.x))
        XCTAssertTrue(isApproximatelyEqual(basisY, orientation.columns.1 * expectedScale.y))
        XCTAssertTrue(isApproximatelyEqual(basisZ, orientation.columns.2 * expectedScale.z))
    }
}

private extension VolumeCubeMaterialDicomTests {
    func isApproximatelyEqual(_ lhs: SIMD3<Float>, _ rhs: SIMD3<Float>, tolerance: Float = 1e-4) -> Bool {
        abs(lhs.x - rhs.x) <= tolerance &&
        abs(lhs.y - rhs.y) <= tolerance &&
        abs(lhs.z - rhs.z) <= tolerance
    }

    func waitUntil(timeout: TimeInterval, pollInterval: TimeInterval = 0.05, condition: @escaping () -> Bool) -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if condition() { return true }
            RunLoop.current.run(until: Date().addingTimeInterval(pollInterval))
        }
        return condition()
    }
}
