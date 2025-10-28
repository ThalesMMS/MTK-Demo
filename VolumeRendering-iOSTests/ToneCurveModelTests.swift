@testable import VolumeRendering_iOS
import XCTest

final class ToneCurveModelTests: XCTestCase {
    func testSampledValuesMatchExpectedCount() {
        let model = ToneCurveModel()
        let samples = model.sampledValues()
        XCTAssertEqual(samples.count, ToneCurveModel.sampleCount)
        XCTAssertTrue(samples.allSatisfy { $0 >= 0 && $0 <= 1 })
    }

    func testUpdatePointMaintainsOrdering() {
        var model = ToneCurveModel()
        let original = model.currentControlPoints()
        XCTAssertGreaterThan(original.count, 3)

        let targetIndex = 2
        let invalidPoint = ToneCurvePoint(x: original[targetIndex - 1].x - 50, y: 0.5)
        model.updatePoint(at: targetIndex, to: invalidPoint)

        let updated = model.currentControlPoints()
        XCTAssertTrue(updated.isStrictlyIncreasing)
        let delta = updated[targetIndex].x - updated[targetIndex - 1].x
        XCTAssertGreaterThanOrEqual(delta, ToneCurveModel.minimumDeltaX - 1e-4)
    }

    func testAutoWindowPresetProducesSanitizedCurve() {
        var model = ToneCurveModel()
        let histogram = (0..<512).map { UInt32($0 * 2) }
        model.setHistogram(histogram)

        model.applyAutoWindow(.abdomen)
        let points = model.currentControlPoints()

        XCTAssertTrue(points.isStrictlyIncreasing)
        if let first = points.first {
            XCTAssertEqual(first.x, ToneCurveModel.xRange.lowerBound, accuracy: 1e-4)
        } else {
            XCTFail("Preset gerou lista de pontos vazia.")
        }

        if let last = points.last {
            XCTAssertEqual(last.x, ToneCurveModel.xRange.upperBound, accuracy: 1e-4)
        } else {
            XCTFail("Preset gerou lista de pontos vazia.")
        }

    }

    func testInsertAndRemovePointKeepsSplineContinuous() {
        var model = ToneCurveModel()
        let initialCount = model.currentControlPoints().count
        let newPoint = ToneCurvePoint(x: 128, y: 0.5)
        model.insertPoint(newPoint)
        XCTAssertEqual(model.currentControlPoints().count, initialCount + 1)
        XCTAssertTrue(model.currentControlPoints().isStrictlyIncreasing)

        model.removePoint(at: 2)
        XCTAssertEqual(model.currentControlPoints().count, initialCount)
        XCTAssertTrue(model.currentControlPoints().isStrictlyIncreasing)
    }
}

private extension Array where Element == ToneCurvePoint {
    var isStrictlyIncreasing: Bool {
        guard count > 1 else { return true }
        for index in 1..<count where self[index].x <= self[index - 1].x {
            return false
        }
        return true
    }
}
