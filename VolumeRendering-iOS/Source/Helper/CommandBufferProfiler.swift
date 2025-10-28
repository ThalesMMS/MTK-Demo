import Foundation
import Metal

struct CommandBufferMetrics {
    let label: String
    let cpuMs: Double
    let gpuMs: Double?
    let kernelMs: Double?
    let status: MTLCommandBufferStatus
    let error: Error?

    var formattedSummary: String {
        let cpu = CommandBufferMetrics.format(cpuMs)
        let gpu = gpuMs.map { CommandBufferMetrics.format($0) } ?? "--"
        let kernel = kernelMs.map { CommandBufferMetrics.format($0) } ?? "--"
        return "cpu=\(cpu) ms | gpu=\(gpu) ms | kernel=\(kernel) ms"
    }

    private static func format(_ value: Double) -> String {
        String(format: "%.3f", value)
    }
}

enum CommandBufferProfiler {

    static func captureTimes(for commandBuffer: MTLCommandBuffer,
                             label: String,
                             category: String = "Benchmark") {
        let cpuStart = DispatchTime.now().uptimeNanoseconds

        commandBuffer.addCompletedHandler { buffer in
            let cpuEnd = DispatchTime.now().uptimeNanoseconds
            let cpuMs = Double(cpuEnd &- cpuStart) / 1_000_000.0

            let gpuMs = interval(buffer.gpuStartTime, buffer.gpuEndTime)
            let kernelMs = interval(buffer.kernelStartTime, buffer.kernelEndTime)

            let metrics = CommandBufferMetrics(label: label,
                                               cpuMs: cpuMs,
                                               gpuMs: gpuMs,
                                               kernelMs: kernelMs,
                                               status: buffer.status,
                                               error: buffer.error)

            log(metrics: metrics, category: category)
        }
    }

    private static func interval(_ start: CFTimeInterval, _ end: CFTimeInterval) -> Double? {
        guard start > 0, end > 0 else { return nil }
        return max(0, (end - start) * 1000.0)
    }

    private static func log(metrics: CommandBufferMetrics, category: String) {
        if metrics.status == .completed {
            Logger.log("CommandBuffer [\(metrics.label)] completed: \(metrics.formattedSummary)",
                       level: .info,
                       category: category)
        } else if let error = metrics.error {
            Logger.log("CommandBuffer [\(metrics.label)] failed: \(metrics.formattedSummary) → \(error.localizedDescription)",
                       level: .error,
                       category: category)
        } else {
            Logger.log("CommandBuffer [\(metrics.label)] ended with status \(describe(metrics.status)): \(metrics.formattedSummary)",
                       level: .warn,
                       category: category)
        }
    }

    private static func describe(_ status: MTLCommandBufferStatus) -> String {
        switch status {
        case .notEnqueued: return "notEnqueued"
        case .enqueued: return "enqueued"
        case .committed: return "committed"
        case .scheduled: return "scheduled"
        case .completed: return "completed"
        case .error: return "error"
        @unknown default: return "unknown(\(status.rawValue))"
        }
    }
}

struct CommandBufferTimings {
    let kernelTime: Double
    let gpuTime: Double
    let cpuTime: Double
}

extension MTLCommandBuffer {
    func timings(cpuStart: CFTimeInterval, cpuEnd: CFTimeInterval) -> CommandBufferTimings {
        let kernelDuration = Self.durationMillis(start: kernelStartTime, end: kernelEndTime)
        let gpuDuration = Self.durationMillis(start: gpuStartTime, end: gpuEndTime)
        let cpuDuration = max(0, (cpuEnd - cpuStart) * 1000.0)

        return CommandBufferTimings(kernelTime: kernelDuration,
                                    gpuTime: gpuDuration,
                                    cpuTime: cpuDuration)
    }

    private static func durationMillis(start: CFTimeInterval, end: CFTimeInterval) -> Double {
        guard start > 0, end > 0, end >= start else {
            return 0
        }
        return (end - start) * 1000.0
    }
}
