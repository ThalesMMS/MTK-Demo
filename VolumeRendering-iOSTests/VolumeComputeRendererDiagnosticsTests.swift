@testable import VolumeRendering_iOS
import simd
import XCTest

final class VolumeComputeRendererDiagnosticsTests: XCTestCase {

    func testComputeRayDirectionNormalizesDelta() {
        let cameraLocal01 = SIMD3<Float>(0.1, 0.2, 0.3)
        let pixelLocal01 = SIMD3<Float>(0.6, 0.6, 0.9)

        let direction = VolumeComputeRenderer.computeRayDirection(cameraLocal01: cameraLocal01,
                                                                  pixelLocal01: pixelLocal01)

        XCTAssertGreaterThan(simd_length(direction), 0.0)
        XCTAssertEqual(simd_length(direction), 1.0, accuracy: 1e-5)

        let expected = simd_normalize(pixelLocal01 - cameraLocal01)
        XCTAssertEqual(direction.x, expected.x, accuracy: 1e-5)
        XCTAssertEqual(direction.y, expected.y, accuracy: 1e-5)
        XCTAssertEqual(direction.z, expected.z, accuracy: 1e-5)
    }

    func testComputeRayDirectionZeroLengthDeltaReturnsZeroVector() {
        let origin = SIMD3<Float>(repeating: 0.5)

        let direction = VolumeComputeRenderer.computeRayDirection(cameraLocal01: origin,
                                                                  pixelLocal01: origin)

        XCTAssertEqual(direction, SIMD3<Float>(repeating: 0.0))
    }

    func testRayBoxIntersectionReturnsOrderedNearAndFarValues() {
        let origin = SIMD3<Float>(-1.0, 0.5, 0.5)
        let direction = SIMD3<Float>(1.0, 0.0, 0.0)
        let boxMin = SIMD3<Float>(repeating: 0.0)
        let boxMax = SIMD3<Float>(repeating: 1.0)

        let intersection = VolumeComputeRenderer.rayBoxIntersection(rayOrigin: origin,
                                                                    rayDirection: direction,
                                                                    boxMin: boxMin,
                                                                    boxMax: boxMax)

        XCTAssertEqual(intersection.x, 1.0, accuracy: 1e-4)
        XCTAssertEqual(intersection.y, 2.0, accuracy: 1e-4)
        XCTAssertLessThan(intersection.x, intersection.y)
    }

    func testRayBoxIntersectionHandlesAxisAlignedComponentsNearZero() {
        let origin = SIMD3<Float>(0.5, 0.5, -1.0)
        let direction = SIMD3<Float>(0.0, 0.0, 1.0)
        let boxMin = SIMD3<Float>(repeating: 0.0)
        let boxMax = SIMD3<Float>(repeating: 1.0)

        let intersection = VolumeComputeRenderer.rayBoxIntersection(rayOrigin: origin,
                                                                    rayDirection: direction,
                                                                    boxMin: boxMin,
                                                                    boxMax: boxMax)

        XCTAssertGreaterThan(intersection.x, 0.0)
        XCTAssertEqual(intersection.x, 1.0, accuracy: 1e-4)
        XCTAssertEqual(intersection.y, 2.0, accuracy: 1e-4)
    }
}
