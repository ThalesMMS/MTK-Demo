@testable import VolumeRendering_iOS
import Foundation
import SceneKit
import XCTest

@MainActor
final class SceneViewControllerDicomIntegrationTests: XCTestCase {

    func testApplySyntheticDatasetPublishesVolumeAndMetadata() throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal indisponível neste ambiente.")
        }

        let (controller, view) = makeController()
        controller.onAppear(view)

        let dataset = makeSyntheticDataset(size: 32,
                                           lowerValue: -800,
                                           upperValue: 1200)
        controller.applyImportedDataset(dataset)

        XCTAssertTrue(waitUntil(timeout: 5.0) {
            controller.currentVolumeTexture() != nil
        }, "Textura de volume não ficou disponível após aplicar dataset.")
        XCTAssertTrue(waitUntil(timeout: 5.0) {
            controller.currentDatasetMeta() != nil
        }, "Metadados do dataset (dimension/resolution) não ficaram disponíveis.")

        guard let texture = controller.currentVolumeTexture() else {
            XCTFail("Textura de volume ausente.")
            return
        }
        XCTAssertEqual(texture.width, Int(dataset.dimensions.x))
        XCTAssertEqual(texture.height, Int(dataset.dimensions.y))
        XCTAssertEqual(texture.depth, Int(dataset.dimensions.z))

        guard let metadata = controller.currentDatasetMeta() else {
            XCTFail("Metadados de volume não foram publicados.")
            return
        }
        XCTAssertEqual(metadata.dimension, dataset.dimensions)
        XCTAssertEqual(metadata.resolution.x, dataset.spacing.x, accuracy: 1e-6)
        XCTAssertEqual(metadata.resolution.y, dataset.spacing.y, accuracy: 1e-6)
        XCTAssertEqual(metadata.resolution.z, dataset.spacing.z, accuracy: 1e-6)
    }
}

@MainActor
private extension SceneViewControllerDicomIntegrationTests {
    func makeController() -> (SceneViewController, SCNView) {
        let controller = SceneViewController()
        let view = SCNView(frame: CGRect(x: 0, y: 0, width: 256, height: 256), options: nil)
        view.scene = SCNScene()
        return (controller, view)
    }

    func waitUntil(timeout: TimeInterval, pollInterval: TimeInterval = 0.05, condition: @escaping () -> Bool) -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if condition() { return true }
            RunLoop.current.run(until: Date().addingTimeInterval(pollInterval))
        }
        return condition()
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
                        let index = z * size * size + y * size + x
                        let position = SIMD3<Float>(Float(x), Float(y), Float(z))
                        let distanceSq = simd_length_squared(position - center)
                        if distanceSq <= innerRadiusSq {
                            base[index] = upperValue
                        } else if distanceSq <= outerRadiusSq {
                            base[index] = Int16((Int(lowerValue) + Int(upperValue)) / 2)
                        } else {
                            base[index] = lowerValue
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
}
