@testable import VolumeRendering_iOS
import XCTest

final class ThreadgroupDispatchOptimizerTests: XCTestCase {

    func testDefaultConfigurationRespectsPipelineLimits() {
        let limits = ThreadgroupPipelineLimits(threadExecutionWidth: 40,
                                               maxTotalThreadsPerThreadgroup: 320)

        let configuration = ThreadgroupDispatchConfiguration.default(for: limits)

        XCTAssertEqual(configuration.width, 40)
        XCTAssertEqual(configuration.height, 8)
        XCTAssertEqual(configuration.threadCount, 320)
    }

    func testClampedConfigurationHonorsMaxThreadBudget() {
        let limits = ThreadgroupPipelineLimits(threadExecutionWidth: 16,
                                               maxTotalThreadsPerThreadgroup: 64)
        let oversized = ThreadgroupDispatchConfiguration(width: 512, height: 512)

        let clamped = oversized.clamped(to: limits)

        XCTAssertNotNil(clamped)
        XCTAssertEqual(clamped?.width, 64)
        XCTAssertEqual(clamped?.height, 1)
        XCTAssertLessThanOrEqual(clamped?.threadCount ?? .max, 64)
    }

    func testCandidatesReturnUniqueConfigurationsWithinLimits() {
        let limits = ThreadgroupPipelineLimits(threadExecutionWidth: 32,
                                               maxTotalThreadsPerThreadgroup: 256)

        let candidates = ThreadgroupDispatchConfiguration.candidates(for: limits)

        XCTAssertFalse(candidates.isEmpty)
        XCTAssertEqual(Set(candidates).count, candidates.count)
        XCTAssertTrue(candidates.contains(.default(for: limits)))
        XCTAssertTrue(candidates.allSatisfy { $0.threadCount <= 256 })
    }

    func testOptimizerSelectsFastestConfigurationAndCachesResult() {
        let optimizer = ThreadgroupDispatchOptimizer()
        let limits = ThreadgroupPipelineLimits(threadExecutionWidth: 32,
                                               maxTotalThreadsPerThreadgroup: 256)
        let keyHost = DummyPipelineKey()
        let key = ObjectIdentifier(keyHost)
        var callCount = 0
        let benchmark: (ThreadgroupDispatchConfiguration) -> CommandBufferTimings? = { configuration in
            callCount += 1
            let time = Double(configuration.threadCount)
            return CommandBufferTimings(kernelTime: time / 2,
                                        gpuTime: time,
                                        cpuTime: time / 4)
        }

        let selected = optimizer.configuration(key: key,
                                               limits: limits,
                                               label: "dummy",
                                               width: 512,
                                               height: 512,
                                               benchmark: benchmark)

        XCTAssertEqual(callCount, ThreadgroupDispatchConfiguration.candidates(for: limits).count)
        XCTAssertEqual(selected, ThreadgroupDispatchConfiguration(width: 8, height: 8))

        callCount = 0
        let cached = optimizer.configuration(key: key,
                                             limits: limits,
                                             label: "dummy",
                                             width: 512,
                                             height: 512,
                                             benchmark: benchmark)
        XCTAssertEqual(callCount, 0)
        XCTAssertEqual(selected, cached)
    }

    func testInvalidateAllClearsCachedConfigurations() {
        let optimizer = ThreadgroupDispatchOptimizer()
        let limits = ThreadgroupPipelineLimits(threadExecutionWidth: 16,
                                               maxTotalThreadsPerThreadgroup: 128)
        let keyHost = DummyPipelineKey()
        let key = ObjectIdentifier(keyHost)
        var callCount = 0
        let benchmark: (ThreadgroupDispatchConfiguration) -> CommandBufferTimings? = { configuration in
            callCount += 1
            return CommandBufferTimings(kernelTime: Double(configuration.threadCount),
                                        gpuTime: Double(configuration.threadCount),
                                        cpuTime: Double(configuration.threadCount))
        }

        _ = optimizer.configuration(key: key,
                                    limits: limits,
                                    label: "dummy",
                                    width: 256,
                                    height: 256,
                                    benchmark: benchmark)
        XCTAssertGreaterThan(callCount, 0)

        optimizer.invalidateAll()
        callCount = 0

        _ = optimizer.configuration(key: key,
                                    limits: limits,
                                    label: "dummy",
                                    width: 256,
                                    height: 256,
                                    benchmark: benchmark)
        XCTAssertGreaterThan(callCount, 0)
    }
}

private final class DummyPipelineKey: NSObject {}
