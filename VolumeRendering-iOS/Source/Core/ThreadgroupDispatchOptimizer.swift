import Foundation
import Metal

struct ThreadgroupPipelineLimits {
    let threadExecutionWidth: Int
    let maxTotalThreadsPerThreadgroup: Int

    init(threadExecutionWidth: Int, maxTotalThreadsPerThreadgroup: Int) {
        self.threadExecutionWidth = threadExecutionWidth
        self.maxTotalThreadsPerThreadgroup = maxTotalThreadsPerThreadgroup
    }

    init(pipeline: MTLComputePipelineState) {
        self.init(threadExecutionWidth: pipeline.threadExecutionWidth,
                  maxTotalThreadsPerThreadgroup: pipeline.maxTotalThreadsPerThreadgroup)
    }
}

struct ThreadgroupDispatchConfiguration: Hashable, CustomStringConvertible {
    let width: Int
    let height: Int

    var description: String {
        "\(width)x\(height)"
    }

    var threadCount: Int {
        width * height
    }

    var threadsPerThreadgroup: MTLSize {
        MTLSize(width: width, height: height, depth: 1)
    }

    func threadgroupsPerGrid(forWidth width: Int, height: Int) -> MTLSize {
        let groupsX = max(1, (width + self.width - 1) / self.width)
        let groupsY = max(1, (height + self.height - 1) / self.height)
        return MTLSize(width: groupsX, height: groupsY, depth: 1)
    }

    func clamped(to pipeline: MTLComputePipelineState) -> ThreadgroupDispatchConfiguration? {
        clamped(to: ThreadgroupPipelineLimits(pipeline: pipeline))
    }

    func clamped(to limits: ThreadgroupPipelineLimits) -> ThreadgroupDispatchConfiguration? {
        let maxThreads = max(1, limits.maxTotalThreadsPerThreadgroup)
        let clampedWidth = max(1, min(width, min(maxThreads, 1024)))
        let maxHeight = maxThreads / clampedWidth
        guard maxHeight > 0 else {
            return nil
        }
        let clampedHeight = max(1, min(height, min(maxHeight, 1024)))
        return ThreadgroupDispatchConfiguration(width: clampedWidth, height: clampedHeight)
    }

    static func `default`(for pipeline: MTLComputePipelineState) -> ThreadgroupDispatchConfiguration {
        self.default(for: ThreadgroupPipelineLimits(pipeline: pipeline))
    }

    static func `default`(for limits: ThreadgroupPipelineLimits) -> ThreadgroupDispatchConfiguration {
        let width = max(1, min(limits.threadExecutionWidth, 1024))
        let maxThreads = max(1, limits.maxTotalThreadsPerThreadgroup)
        let height = max(1, min(maxThreads / width, 1024))
        return ThreadgroupDispatchConfiguration(width: width, height: height)
    }

    static func candidates(for pipeline: MTLComputePipelineState) -> [ThreadgroupDispatchConfiguration] {
        candidates(for: ThreadgroupPipelineLimits(pipeline: pipeline))
    }

    static func candidates(for limits: ThreadgroupPipelineLimits) -> [ThreadgroupDispatchConfiguration] {
        let presets: [ThreadgroupDispatchConfiguration] = [
            .default(for: limits),
            ThreadgroupDispatchConfiguration(width: 8, height: 8),
            ThreadgroupDispatchConfiguration(width: 16, height: 8),
            ThreadgroupDispatchConfiguration(width: 16, height: 16),
            ThreadgroupDispatchConfiguration(width: 32, height: 4)
        ]

        var unique = Set<ThreadgroupDispatchConfiguration>()
        var results: [ThreadgroupDispatchConfiguration] = []

        for preset in presets {
            guard let adjusted = preset.clamped(to: limits) else {
                continue
            }

            if unique.insert(adjusted).inserted {
                results.append(adjusted)
            }
        }

        return results
    }
}

final class ThreadgroupDispatchOptimizer {
    private struct CacheEntry {
        let configuration: ThreadgroupDispatchConfiguration
        let measurements: [ThreadgroupDispatchConfiguration: CommandBufferTimings]
    }

    private var cache: [ObjectIdentifier: CacheEntry] = [:]
    private let cacheLock = NSLock()

    func configuration(
        for pipeline: MTLComputePipelineState,
        width: Int,
        height: Int,
        benchmark: (ThreadgroupDispatchConfiguration) -> CommandBufferTimings?
    ) -> ThreadgroupDispatchConfiguration? {
        configuration(
            key: ObjectIdentifier(pipeline),
            limits: ThreadgroupPipelineLimits(pipeline: pipeline),
            label: pipeline.label,
            width: width,
            height: height,
            benchmark: benchmark
        )
    }

    func configuration(
        key: ObjectIdentifier,
        limits: ThreadgroupPipelineLimits,
        label: String?,
        width: Int,
        height: Int,
        benchmark: (ThreadgroupDispatchConfiguration) -> CommandBufferTimings?
    ) -> ThreadgroupDispatchConfiguration? {
        let identifier = key

        cacheLock.lock()
        if let entry = cache[identifier] {
            cacheLock.unlock()
            return entry.configuration
        }
        cacheLock.unlock()

        var bestConfiguration: ThreadgroupDispatchConfiguration?
        var bestTiming: Double = .infinity
        var measurements: [ThreadgroupDispatchConfiguration: CommandBufferTimings] = [:]

        for candidate in ThreadgroupDispatchConfiguration.candidates(for: limits) {
            guard let timing = benchmark(candidate) else {
                continue
            }
            measurements[candidate] = timing

            if timing.gpuTime < bestTiming {
                bestTiming = timing.gpuTime
                bestConfiguration = candidate
            }
        }

        guard let selectedConfiguration = bestConfiguration,
              let selectedTiming = measurements[selectedConfiguration] else {
            return nil
        }

        cacheLock.lock()
        cache[identifier] = CacheEntry(configuration: selectedConfiguration, measurements: measurements)
        cacheLock.unlock()

        logSelection(label: label,
                     width: width,
                     height: height,
                     measurements: measurements,
                     bestConfiguration: selectedConfiguration,
                     bestTiming: selectedTiming)

        return selectedConfiguration
    }

    func invalidateAll() {
        cacheLock.lock()
        cache.removeAll()
        cacheLock.unlock()
    }

    private func logSelection(
        label: String?,
        width: Int,
        height: Int,
        measurements: [ThreadgroupDispatchConfiguration: CommandBufferTimings],
        bestConfiguration: ThreadgroupDispatchConfiguration,
        bestTiming: CommandBufferTimings
    ) {
        guard AppConfig.IS_DEBUG_MODE else {
            return
        }

        let sortedResults = measurements.sorted { lhs, rhs in
            lhs.value.gpuTime < rhs.value.gpuTime
        }

        var lines: [String] = []
        lines.append("Dispatch benchmark for \(label ?? "unnamed pipeline") @ \(width)x\(height)")
        for (config, timing) in sortedResults {
            let marker = config == bestConfiguration ? "*" : " "
            let label = "\(marker) \(config.description)".padding(toLength: 8, withPad: " ", startingAt: 0)
            let timingSummary = String(
                format: "gpu=%6.3fms kernel=%6.3fms cpu=%6.3fms",
                timing.gpuTime,
                timing.kernelTime,
                timing.cpuTime
            )
            lines.append("\(label) \(timingSummary)")
        }

        lines.append("Selected threadgroup \(bestConfiguration.description) (gpu=\(String(format: "%.3f", bestTiming.gpuTime))ms)")

        Logger.log(lines.joined(separator: "\n"),
                   level: .debug,
                   category: "ThreadgroupDispatch")
    }
}
