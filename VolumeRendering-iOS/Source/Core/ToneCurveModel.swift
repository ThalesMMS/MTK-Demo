//
//  ToneCurveModel.swift
//  VolumeRendering-iOS
//
//  Provides cubic-spline tone curve editing and auto-window presets until MTK exposes equivalent controls.
//  Thales Matheus Mendonça Santos — October 2025
//

import Foundation

struct ToneCurvePoint: Codable, Equatable {
    var x: Float // 0...255
    var y: Float // 0...1
}

struct ToneCurveAutoWindowPreset: Identifiable, Equatable {
    let id: String
    let title: String
    let lowerPercentile: Double?
    let upperPercentile: Double?
    let smoothingRadius: Int

    static let abdomen   = ToneCurveAutoWindowPreset(id: "auto.abdomen", title: "Abdômen CT", lowerPercentile: 0.10, upperPercentile: 0.90, smoothingRadius: 3)
    static let lung      = ToneCurveAutoWindowPreset(id: "auto.lung", title: "Pulmão", lowerPercentile: 0.005, upperPercentile: 0.60, smoothingRadius: 4)
    static let bone      = ToneCurveAutoWindowPreset(id: "auto.bone", title: "Ossos", lowerPercentile: 0.40, upperPercentile: 0.995, smoothingRadius: 2)
    static let otsu      = ToneCurveAutoWindowPreset(id: "auto.otsu", title: "Otsu", lowerPercentile: nil, upperPercentile: nil, smoothingRadius: 3)

    static let allPresets: [ToneCurveAutoWindowPreset] = [.abdomen, .lung, .bone, .otsu]
}

final class ToneCurveModel {
    static let sampleScale: Int = 10
    static let minimumDeltaX: Float = 0.5
    static let xRange: ClosedRange<Float> = 0...255
    static let yRange: ClosedRange<Float> = 0...1
    static var sampleCount: Int { Int(255 * sampleScale) + 1 }

    private(set) var controlPoints: [ToneCurvePoint] {
        didSet { rebuildSpline() }
    }

    private var spline: CubicSplineInterpolator?
    private(set) var histogram: [UInt32] = [] {
        didSet { cachedSmoothedHistogram = nil }
    }

    private var cachedSmoothedHistogram: [Double]?
    var interpolationMode: CubicSplineInterpolator.InterpolationMode = .cubicSpline {
        didSet { rebuildSpline() }
    }

    init(points: [ToneCurvePoint] = ToneCurveModel.defaultControlPoints()) {
        self.controlPoints = ToneCurveModel.sanitized(points)
        rebuildSpline()
    }

    func setHistogram(_ values: [UInt32]) {
        histogram = values
    }

    func currentControlPoints() -> [ToneCurvePoint] {
        controlPoints
    }

    func reset() {
        controlPoints = ToneCurveModel.defaultControlPoints()
    }

    func setControlPoints(_ points: [ToneCurvePoint]) {
        controlPoints = ToneCurveModel.sanitized(points)
    }

    func updatePoint(at index: Int, to newPoint: ToneCurvePoint) {
        guard controlPoints.indices.contains(index) else { return }
        var updated = controlPoints
        updated[index] = newPoint
        controlPoints = ToneCurveModel.sanitized(updated, preservingIndex: index)
    }

    func insertPoint(_ point: ToneCurvePoint) {
        var updated = controlPoints
        updated.append(point)
        controlPoints = ToneCurveModel.sanitized(updated)
    }

    func removePoint(at index: Int) {
        guard controlPoints.indices.contains(index) else { return }
        guard index != 0 && index != controlPoints.count - 1 else { return }
        controlPoints.remove(at: index)
        rebuildSpline()
    }

    func sampledValues(scale: Int = ToneCurveModel.sampleScale) -> [Float] {
        guard let spline else {
            return []
        }

        let clampedScale = max(1, scale)
        let sampleCount = Int(255 * clampedScale) + 1
        let step = 1.0 / Float(clampedScale)

        var values = [Float]()
        values.reserveCapacity(sampleCount)

        var x: Float = 0
        for _ in 0..<sampleCount {
            let value = max(Self.yRange.lowerBound,
                            min(Self.yRange.upperBound,
                                spline.interpolate(x)))
            values.append(value)
            x += step
        }
        return values
    }

    func applyAutoWindow(_ preset: ToneCurveAutoWindowPreset) {
        guard !histogram.isEmpty else { return }

        if let lower = preset.lowerPercentile,
           let upper = preset.upperPercentile {
            applyPercentileAutoWindow(lowerPercentile: lower,
                                      upperPercentile: upper,
                                      smoothingRadius: preset.smoothingRadius)
        } else {
            applyOtsuAutoWindow(smoothingRadius: preset.smoothingRadius)
        }
    }
}

// MARK: - Private helpers
private extension ToneCurveModel {
    static func defaultControlPoints() -> [ToneCurvePoint] {
        [
            .init(x: 0, y: 0),
            .init(x: 32, y: 0.05),
            .init(x: 96, y: 0.3),
            .init(x: 160, y: 0.7),
            .init(x: 224, y: 0.95),
            .init(x: 255, y: 1)
        ]
    }

    static func sanitized(_ points: [ToneCurvePoint],
                          preservingIndex index: Int? = nil) -> [ToneCurvePoint] {
        guard !points.isEmpty else {
            return defaultControlPoints()
        }

        var sorted = points.sorted { $0.x < $1.x }

        if sorted.first?.x != xRange.lowerBound {
            sorted[0].x = xRange.lowerBound
            sorted[0].y = yRange.lowerBound
        }
        if sorted.last?.x != xRange.upperBound {
            sorted[sorted.count - 1].x = xRange.upperBound
            sorted[sorted.count - 1].y = yRange.upperBound
        }

        for current in 1..<sorted.count {
            let previous = current - 1
            if sorted[current].x <= sorted[previous].x {
                sorted[current].x = sorted[previous].x + minimumDeltaX
            }
        }

        for idx in 0..<sorted.count {
            sorted[idx].x = max(xRange.lowerBound, min(xRange.upperBound, sorted[idx].x))
            sorted[idx].y = max(yRange.lowerBound, min(yRange.upperBound, sorted[idx].y))
        }

        if let index,
           sorted.indices.contains(index) {
            // Preserve fixed x for endpoints
            if index == 0 {
                sorted[index].x = xRange.lowerBound
            } else if index == sorted.count - 1 {
                sorted[index].x = xRange.upperBound
            }
        }

        for idx in 1..<sorted.count {
            let prev = sorted[idx - 1].x
            if sorted[idx].x - prev < minimumDeltaX {
                sorted[idx].x = min(prev + minimumDeltaX, xRange.upperBound)
            }
        }

        // When points cluster near the upper bound the forward pass above may
        // saturate multiple samples at 255. Propagate adjustments backwards to
        // guarantee strictly increasing X coordinates while staying within range.
        if sorted.count >= 2 {
            for idx in stride(from: sorted.count - 2, through: 0, by: -1) {
                let next = sorted[idx + 1].x
                let maxAllowed = next - minimumDeltaX
                if sorted[idx].x > maxAllowed {
                    sorted[idx].x = max(xRange.lowerBound + Float(idx) * minimumDeltaX,
                                        maxAllowed)
                }
            }
            sorted[0].x = xRange.lowerBound
            sorted[sorted.count - 1].x = xRange.upperBound
        }

        return sorted
    }

    func rebuildSpline() {
        let xs = controlPoints.map { $0.x }
        let ys = controlPoints.map { $0.y }

        if spline == nil {
            spline = CubicSplineInterpolator(xPoints: xs, yPoints: ys)
        } else {
            spline?.updateSpline(xPoints: xs, yPoints: ys)
        }

        spline?.mode = interpolationMode
    }

    func smoothedHistogram(radius: Int) -> [Double] {
        if radius <= 0 {
            return histogram.map { Double($0) }
        }

        if let cached = cachedSmoothedHistogram, cached.count == histogram.count {
            return cached
        }

        let input = histogram.map { Double($0) }
        var output = [Double](repeating: 0, count: histogram.count)
        for index in 0..<input.count {
            let lower = max(0, index - radius)
            let upper = min(input.count - 1, index + radius)
            var sum: Double = 0
            for sample in lower...upper {
                sum += input[sample]
            }
            output[index] = sum / Double(upper - lower + 1)
        }
        cachedSmoothedHistogram = output
        return output
    }

    func percentileIndex(in distribution: [Double], percentile: Double) -> Int? {
        guard !distribution.isEmpty else { return nil }
        let clamped = max(0.0, min(percentile, 1.0))
        let total = distribution.reduce(0, +)
        guard total > 0 else { return nil }

        let target = clamped * total
        var cumulative = 0.0
        for (idx, value) in distribution.enumerated() {
            cumulative += value
            if cumulative >= target {
                return idx
            }
        }
        return distribution.count - 1
    }

    func applyPercentileAutoWindow(lowerPercentile: Double,
                                   upperPercentile: Double,
                                   smoothingRadius: Int) {
        guard lowerPercentile < upperPercentile else { return }
        let smoothed = smoothedHistogram(radius: smoothingRadius)
        guard let lowerIndex = percentileIndex(in: smoothed, percentile: lowerPercentile),
              let upperIndex = percentileIndex(in: smoothed, percentile: upperPercentile),
              lowerIndex < upperIndex else {
            return
        }
        commitAutoWindow(lowerBin: lowerIndex,
                         upperBin: upperIndex,
                         totalBins: smoothed.count)
    }

    func applyOtsuAutoWindow(smoothingRadius: Int) {
        let smoothed = smoothedHistogram(radius: smoothingRadius)
        guard let threshold = otsuThreshold(for: smoothed) else { return }

        let windowWidth = max(1, Int(Double(smoothed.count) * 0.08))
        let lower = max(0, threshold - windowWidth)
        let upper = min(smoothed.count - 1, threshold + windowWidth)

        commitAutoWindow(lowerBin: lower, upperBin: upper, totalBins: smoothed.count)
    }

    func otsuThreshold(for histogram: [Double]) -> Int? {
        let total = histogram.reduce(0, +)
        guard total > 0 else { return nil }

        var sum: Double = 0
        for (idx, value) in histogram.enumerated() {
            sum += Double(idx) * value
        }

        var sumBackground: Double = 0
        var weightBackground: Double = 0
        var maxVariance: Double = -1
        var threshold: Int = 0

        for (idx, value) in histogram.enumerated() {
            weightBackground += value
            if weightBackground == 0 {
                continue
            }

            let weightForeground = total - weightBackground
            if weightForeground == 0 {
                break
            }

            sumBackground += Double(idx) * value
            let meanBackground = sumBackground / weightBackground
            let meanForeground = (sum - sumBackground) / weightForeground
            let betweenClassVariance = weightBackground * weightForeground * pow(meanBackground - meanForeground, 2)

            if betweenClassVariance > maxVariance {
                maxVariance = betweenClassVariance
                threshold = idx
            }
        }
        return threshold
    }

    func commitAutoWindow(lowerBin: Int, upperBin: Int, totalBins: Int) {
        guard totalBins > 1 else { return }

        let lowerPosition = Float(lowerBin) * 255.0 / Float(totalBins - 1)
        var upperPosition = Float(upperBin) * 255.0 / Float(totalBins - 1)

        if upperPosition - lowerPosition < 1 {
            upperPosition = min(255, lowerPosition + 1)
        }

        let span = max(upperPosition - lowerPosition, 1)
        let shoulder = min(20.0, span * 0.15)
        let startShoulder = max(0, lowerPosition - shoulder)
        let endShoulder = min(255, upperPosition + shoulder)
        let midLow = lowerPosition + span * 0.35
        let midHigh = lowerPosition + span * 0.75

        let points: [ToneCurvePoint] = [
            .init(x: 0, y: 0),
            .init(x: startShoulder, y: 0),
            .init(x: lowerPosition, y: 0.05),
            .init(x: midLow, y: 0.35),
            .init(x: midHigh, y: 0.85),
            .init(x: upperPosition, y: 1),
            .init(x: endShoulder, y: 1),
            .init(x: 255, y: 1)
        ]

        controlPoints = ToneCurveModel.sanitized(points)
    }
}
