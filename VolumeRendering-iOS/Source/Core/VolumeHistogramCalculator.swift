//
//  VolumeHistogramCalculator.swift
//  VolumeRendering-iOS
//
//  GPU-backed histogram calculator used for auto-windowing workflows.
//

import Foundation
import Metal

struct HistogramDispatchPlan {
    let bins: Int
    let channelCount: Int
    let bufferLength: Int
    let usesThreadgroupMemory: Bool
    let kernelName: String
}

enum HistogramKernelPlanner {
    static func makePlan(channelCount: Int,
                         requestedBins: Int,
                         defaultBinCount: Int,
                         maxThreadgroupMemoryLength: Int) -> HistogramDispatchPlan {
        let bins = VolumeHistogramCalculator.clampBinCount(requestedBins > 0 ? requestedBins : defaultBinCount)
        let clampedChannels = max(1, min(channelCount, 4))
        let bufferLength = clampedChannels * bins * MemoryLayout<UInt32>.stride
        let usesThreadgroupMemory = bufferLength <= max(0, maxThreadgroupMemoryLength)
        let kernelName = usesThreadgroupMemory ? "computeHistogramThreadgroup" : "computeHistogramLegacy"

        return HistogramDispatchPlan(bins: bins,
                                     channelCount: clampedChannels,
                                     bufferLength: bufferLength,
                                     usesThreadgroupMemory: usesThreadgroupMemory,
                                     kernelName: kernelName)
    }
}

final class VolumeHistogramCalculator {
    enum HistogramError: Error {
        case pipelineUnavailable
        case commandQueueUnavailable
        case commandBufferCreationFailed
        case encoderCreationFailed
        case bufferAllocationFailed
    }

    private let device: MTLDevice
    private let featureFlags: FeatureFlags
    private let commandQueue: MTLCommandQueue
    private var pipelineCache: [String: MTLComputePipelineState] = [:]

    init?(device: MTLDevice, featureFlags: FeatureFlags) {
        self.device = device
        self.featureFlags = featureFlags
        guard let queue = device.makeCommandQueue() else {
            return nil
        }
        queue.label = MTL_label.calculate_histogram
        self.commandQueue = queue
    }

    static func clampBinCount(_ value: Int) -> Int {
        return max(64, min(value, 4096))
    }

    func computeHistogram(for texture: MTLTexture,
                          channelCount: Int,
                          voxelMin: Int32,
                          voxelMax: Int32,
                          bins requestedBins: Int = 0,
                          completion: @escaping (Result<[[UInt32]], Error>) -> Void) {
        var plan = HistogramKernelPlanner.makePlan(channelCount: channelCount,
                                                   requestedBins: requestedBins,
                                                   defaultBinCount: AppConfig.HISTOGRAM_BIN_COUNT,
                                                   maxThreadgroupMemoryLength: device.maxThreadgroupMemoryLength)

        var pipeline = pipelineState(named: plan.kernelName)

        if pipeline == nil, plan.usesThreadgroupMemory {
            plan = HistogramKernelPlanner.makePlan(channelCount: channelCount,
                                                   requestedBins: requestedBins,
                                                   defaultBinCount: AppConfig.HISTOGRAM_BIN_COUNT,
                                                   maxThreadgroupMemoryLength: -1)
            pipeline = pipelineState(named: plan.kernelName)
        }

        guard let resolvedPipeline = pipeline else {
            completion(.failure(HistogramError.pipelineUnavailable))
            return
        }

        guard let histogramBuffer = device.makeBuffer(length: plan.bufferLength, options: .storageModeShared) else {
            completion(.failure(HistogramError.bufferAllocationFailed))
            return
        }
        histogramBuffer.label = MTL_label.histogramBuffer
        memset(histogramBuffer.contents(), 0, plan.bufferLength)

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            completion(.failure(HistogramError.commandBufferCreationFailed))
            return
        }
        commandBuffer.label = MTL_label.calculate_histogram

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            completion(.failure(HistogramError.encoderCreationFailed))
            return
        }
        encoder.label = MTL_label.calculate_histogram

        encoder.setComputePipelineState(resolvedPipeline)
        encoder.setTexture(texture, index: 0)

        var encodedChannelCount = UInt8(plan.channelCount)
        var encodedBins = UInt32(plan.bins)
        var encodedVoxelMin = Int16(voxelMin)
        var encodedVoxelMax = Int16(voxelMax)

        encoder.setBytes(&encodedChannelCount, length: MemoryLayout<UInt8>.stride, index: 0)
        encoder.setBytes(&encodedBins, length: MemoryLayout<UInt32>.stride, index: 1)
        encoder.setBytes(&encodedVoxelMin, length: MemoryLayout<Int16>.stride, index: 2)
        encoder.setBytes(&encodedVoxelMax, length: MemoryLayout<Int16>.stride, index: 3)
        encoder.setBuffer(histogramBuffer, offset: 0, index: 4)

        if plan.usesThreadgroupMemory {
            encoder.setThreadgroupMemoryLength(plan.bufferLength, index: 0)
        }

        let configuration = ThreadgroupDispatchConfiguration.default(for: ThreadgroupPipelineLimits(pipeline: resolvedPipeline))
        let threadsPerThreadgroup = configuration.threadsPerThreadgroup
        let threadsPerGrid = MTLSize(width: texture.width,
                                     height: texture.height,
                                     depth: max(texture.depth, 1))

        if featureFlags.contains(.nonUniformThreadgroups) {
            encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        } else {
            let groups = MTLSize(width: (threadsPerGrid.width + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
                                 height: (threadsPerGrid.height + threadsPerThreadgroup.height - 1) / threadsPerThreadgroup.height,
                                 depth: (threadsPerGrid.depth + threadsPerThreadgroup.depth - 1) / threadsPerThreadgroup.depth)
            encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: threadsPerThreadgroup)
        }

        encoder.endEncoding()

        commandBuffer.addCompletedHandler { buffer in
            if let error = buffer.error {
                completion(.failure(error))
                return
            }

            var results: [[UInt32]] = []
            results.reserveCapacity(plan.channelCount)

            for channel in 0..<plan.channelCount {
                let offset = channel * plan.bins * MemoryLayout<UInt32>.stride
                let pointer = histogramBuffer.contents().advanced(by: offset).assumingMemoryBound(to: UInt32.self)
                let channelHistogram = Array(UnsafeBufferPointer(start: pointer, count: plan.bins))
                results.append(channelHistogram)
            }
            completion(.success(results))
        }

        commandBuffer.commit()
    }
}

private extension VolumeHistogramCalculator {
    func pipelineState(named functionName: String) -> MTLComputePipelineState? {
        if let cached = pipelineCache[functionName] {
            return cached
        }

        guard let function = device.makeDefaultLibrary()?.makeFunction(name: functionName) else {
            Logger.log("Função Metal \(functionName) não encontrada.", level: .error, category: "Histogram")
            return nil
        }
        function.label = MTL_label.calculate_histogram

        let descriptor = MTLComputePipelineDescriptor()
        descriptor.label = MTL_label.calculate_histogram
        descriptor.computeFunction = function

        do {
            let pipeline = try device.makeComputePipelineState(descriptor: descriptor, options: [], reflection: nil)
            pipelineCache[functionName] = pipeline
            return pipeline
        } catch {
            Logger.log("Falha ao criar pipeline de histograma: \(error.localizedDescription)",
                       level: .error,
                       category: "Histogram")
            return nil
        }
    }
}
