@testable import VolumeRendering_iOS
import XCTest

final class VolumeHistogramCalculatorTests: XCTestCase {

    func testPlannerDefaultsBinCountWhenRequestIsZero() {
        let plan = HistogramKernelPlanner.makePlan(channelCount: 3,
                                                   requestedBins: 0,
                                                   defaultBinCount: 512,
                                                   maxThreadgroupMemoryLength: Int.max)

        XCTAssertEqual(plan.bins, 512)
        XCTAssertEqual(plan.channelCount, 3)
        XCTAssertEqual(plan.bufferLength, 3 * 512 * MemoryLayout<UInt32>.stride)
    }

    func testPlannerSelectsThreadgroupKernelWhenFitsInSharedMemory() {
        let maxSharedMemory = 16 * 1024
        let plan = HistogramKernelPlanner.makePlan(channelCount: 4,
                                                   requestedBins: 128,
                                                   defaultBinCount: 512,
                                                   maxThreadgroupMemoryLength: maxSharedMemory)

        XCTAssertTrue(plan.usesThreadgroupMemory)
        XCTAssertEqual(plan.kernelName, "computeHistogramThreadgroup")
        XCTAssertLessThanOrEqual(plan.bufferLength, maxSharedMemory)
    }

    func testPlannerFallsBackToLegacyKernelWhenSharedMemoryInsufficient() {
        let maxSharedMemory = 2048
        let plan = HistogramKernelPlanner.makePlan(channelCount: 4,
                                                   requestedBins: 512,
                                                   defaultBinCount: 512,
                                                   maxThreadgroupMemoryLength: maxSharedMemory)

        XCTAssertFalse(plan.usesThreadgroupMemory)
        XCTAssertEqual(plan.kernelName, "computeHistogramLegacy")
        XCTAssertGreaterThan(plan.bufferLength, maxSharedMemory)
    }

    func testDefaultThreadgroupConfigurationMatchesPipelineLimits() {
        let limits = ThreadgroupPipelineLimits(threadExecutionWidth: 20,
                                               maxTotalThreadsPerThreadgroup: 200)
        let configuration = ThreadgroupDispatchConfiguration.default(for: limits)

        XCTAssertEqual(configuration.width, 20)
        XCTAssertEqual(configuration.height, 10)
        XCTAssertEqual(configuration.threadCount, 200)
    }
}
