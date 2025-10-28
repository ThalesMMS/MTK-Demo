import Foundation
import Metal
import simd

struct VolumeCameraParameters {
    var modelMatrix: simd_float4x4
    var inverseModelMatrix: simd_float4x4
    var inverseViewProjectionMatrix: simd_float4x4
    var cameraPositionLocal: SIMD3<Float>
}

private struct CameraUniforms {
    var modelMatrix: simd_float4x4
    var inverseModelMatrix: simd_float4x4
    var inverseViewProjectionMatrix: simd_float4x4
    var cameraPositionLocal: SIMD3<Float>
    var frameIndex: UInt32
    var padding: UInt32 = 0
}

final class VolumeComputeRenderer {
    private enum RendererError: Error {
        case commandQueueUnavailable
        case functionNotFound(String)
        case outputTextureCreationFailed
        case argumentEncoderUnavailable
    }

    private let device: MTLDevice
    private let featureFlags: FeatureFlags
    private let supportsNonUniformThreadgroups: Bool
    private let commandQueue: MTLCommandQueue
    private let pipelineState: MTLComputePipelineState
    private let dispatchOptimizer = ThreadgroupDispatchOptimizer()

    private var argumentManager: ArgumentEncoderManager?
    private var cameraBuffer: MTLBuffer?
    private var frameIndex: UInt32 = 0
    private var renderingParameters = RenderingParameters()
    private var optionFlags: UInt16 = 0
    private var quaternion = SIMD4<Float>(0, 0, 0, 1)
    private var targetViewSize: UInt16 = 0
    private var earlyTerminationThreshold: Float = 0.99
    private var channelIntensities = SIMD4<Float>(1, 0, 0, 0)
    private var channelTransferTextures: [MTLTexture?] = Array(repeating: nil, count: 4)
    private let fallbackTransferTexture: MTLTexture?
    private var toneBuffers: [MTLBuffer?] = Array(repeating: nil, count: 4)
    private weak var currentVolumeTexture: MTLTexture?
    private(set) var lastRenderTimings: CommandBufferTimings?
    private let toneSampleCount = ToneCurveModel.sampleCount
    private var jitterAmount: Float = 0.0
    private var adaptiveThreshold: Float = 0.1
    private let toneArgumentIndices: [ArgumentEncoderManager.ArgumentIndex] = [
        .toneBufferCh1, .toneBufferCh2, .toneBufferCh3, .toneBufferCh4
    ]
    private struct ArgumentBindingMetadata {
        let role: String
        let fragmentSlot: Int?
    }

    private let argumentBindingMetadata: [ArgumentEncoderManager.ArgumentIndex: ArgumentBindingMetadata] = [
        .mainTexture: .init(role: "Volume density texture3D", fragmentSlot: 0),
        .renderParams: .init(role: "RenderingParameters constant buffer", fragmentSlot: 4),
        .outputTexture: .init(role: "Output render target", fragmentSlot: nil),
        .toneBufferCh1: .init(role: "Tone LUT channel 1", fragmentSlot: 2),
        .toneBufferCh2: .init(role: "Tone LUT channel 2", fragmentSlot: 3),
        .toneBufferCh3: .init(role: "Tone LUT channel 3", fragmentSlot: 4),
        .toneBufferCh4: .init(role: "Tone LUT channel 4", fragmentSlot: 5),
        .optionValue: .init(role: "Rendering option flags", fragmentSlot: nil),
        .quaternion: .init(role: "Volume orientation quaternion", fragmentSlot: nil),
        .targetViewSize: .init(role: "Target view size", fragmentSlot: nil),
        .sampler: .init(role: "Volume sampling state", fragmentSlot: 2),
        .pointSetCountBuffer: .init(role: "Annotation point count", fragmentSlot: nil),
        .pointSetSelectedBuffer: .init(role: "Annotation selected index", fragmentSlot: nil),
        .pointCoordsBuffer: .init(role: "Annotation point coordinates", fragmentSlot: nil),
        .legacyOutputBuffer: .init(role: "Compatibility output buffer", fragmentSlot: nil),
        .transferTextureCh1: .init(role: "Transfer function channel 1", fragmentSlot: 3),
        .transferTextureCh2: .init(role: "Transfer function channel 2", fragmentSlot: 4),
        .transferTextureCh3: .init(role: "Transfer function channel 3", fragmentSlot: 5),
        .transferTextureCh4: .init(role: "Transfer function channel 4", fragmentSlot: 6)
    ]
    private static let optionAdaptiveMask: UInt16 = 1 << 2
    private static let optionDensityDebugMask: UInt16 = 1 << 3

    init(device: MTLDevice, featureFlags: FeatureFlags) throws {
        self.device = device
        self.featureFlags = featureFlags
        self.supportsNonUniformThreadgroups = featureFlags.contains(.nonUniformThreadgroups)

        guard let commandQueue = device.makeCommandQueue() else {
            throw RendererError.commandQueueUnavailable
        }
        commandQueue.label = "VolumeComputeQueue"
        self.commandQueue = commandQueue

        guard let function = device.makeDefaultLibrary()?.makeFunction(name: "volume_compute") else {
            throw RendererError.functionNotFound("volume_compute")
        }
        pipelineState = try device.makeComputePipelineState(function: function)
        argumentManager = ArgumentEncoderManager(device: device, mtlFunction: function)
        guard let argumentManager, argumentManager.argumentBuffer != nil else {
            throw RendererError.argumentEncoderUnavailable
        }

        fallbackTransferTexture = VolumeComputeRenderer.makeFallbackTransferTexture(device: device)
        if fallbackTransferTexture == nil {
            Logger.log("Falha ao criar textura fallback de transferência; canais desativados usarão textura neutra.",
                       level: .warn,
                       category: "VolumeCompute")
        }

        Logger.log("VolumeComputeRenderer inicializado (nonUniformThreadgroups=\(supportsNonUniformThreadgroups))",
                   level: .debug,
                   category: "VolumeCompute")
    }

    func setEarlyTerminationThreshold(_ value: Float) {
        let clamped = max(0.0, min(value, 0.9999))
        guard abs(clamped - earlyTerminationThreshold) > 0.0001 else { return }
        earlyTerminationThreshold = clamped
    }

    func updateChannelIntensities(_ intensities: SIMD4<Float>) {
        let clamped = SIMD4<Float>(max(0, intensities.x),
                                   max(0, intensities.y),
                                   max(0, intensities.z),
                                   max(0, intensities.w))
        channelIntensities = clamped
    }

    func updateTransferTextures(_ textures: [MTLTexture?]) {
        let indices: [ArgumentEncoderManager.ArgumentIndex] = [
            .transferTextureCh1,
            .transferTextureCh2,
            .transferTextureCh3,
            .transferTextureCh4
        ]

        for index in 0..<min(textures.count, channelTransferTextures.count) {
            channelTransferTextures[index] = textures[index]
            if let argumentManager, index < indices.count {
                argumentManager.markAsNeedsUpdate(argumentIndex: indices[index])
            }
        }
    }

    func setAdaptiveEnabled(_ enabled: Bool) {
        if enabled {
            optionFlags |= Self.optionAdaptiveMask
        } else {
            optionFlags &= ~Self.optionAdaptiveMask
        }
    }

    func isAdaptiveEnabled() -> Bool {
        (optionFlags & Self.optionAdaptiveMask) != 0
    }

    func setAdaptiveThreshold(_ value: Float) {
        let clamped = max(value, 0.0)
        guard abs(clamped - adaptiveThreshold) > 1e-4 else { return }
        adaptiveThreshold = clamped
        argumentManager?.markAsNeedsUpdate(argumentIndex: .renderParams)
    }

    func currentAdaptiveThreshold() -> Float {
        adaptiveThreshold
    }

    func setJitterAmount(_ value: Float) {
        let clamped = max(0.0, min(value, 1.0))
        guard abs(clamped - jitterAmount) > 1e-4 else { return }
        jitterAmount = clamped
        argumentManager?.markAsNeedsUpdate(argumentIndex: .renderParams)
    }

    func currentJitterAmount() -> Float {
        jitterAmount
    }

    func setDensityDebugEnabled(_ enabled: Bool) {
        if enabled {
            optionFlags |= Self.optionDensityDebugMask
        } else {
            optionFlags &= ~Self.optionDensityDebugMask
        }
    }

    func refreshCameraUniforms(_ camera: VolumeCameraParameters) {
        let cameraUniforms = makeCameraUniforms(for: camera)
        updateCameraBuffer(with: cameraUniforms)
    }

    func updateClipBounds(xMin: Float,
                          xMax: Float,
                          yMin: Float,
                          yMax: Float,
                          zMin: Float,
                          zMax: Float) {
        let epsilon: Float = 1e-4

        func sanitize(_ minimum: Float, _ maximum: Float) -> (Float, Float) {
            let clampedMin = max(0.0, min(minimum, 1.0))
            let clampedMax = max(0.0, min(maximum, 1.0))
            var minOut = min(clampedMin, clampedMax)
            var maxOut = max(clampedMin, clampedMax)

            if maxOut - minOut < epsilon {
                if maxOut >= 1.0 {
                    minOut = max(0.0, maxOut - epsilon)
                } else {
                    maxOut = min(1.0, minOut + epsilon)
                }
            }

            return (minOut, maxOut)
        }

        let (sxMin, sxMax) = sanitize(xMin, xMax)
        let (syMin, syMax) = sanitize(yMin, yMax)
        let (szMin, szMax) = sanitize(zMin, zMax)

        let changed =
            abs(renderingParameters.trimXMin - sxMin) > epsilon ||
            abs(renderingParameters.trimXMax - sxMax) > epsilon ||
            abs(renderingParameters.trimYMin - syMin) > epsilon ||
            abs(renderingParameters.trimYMax - syMax) > epsilon ||
            abs(renderingParameters.trimZMin - szMin) > epsilon ||
            abs(renderingParameters.trimZMax - szMax) > epsilon

        if !changed {
            return
        }

        renderingParameters.trimXMin = sxMin
        renderingParameters.trimXMax = sxMax
        renderingParameters.trimYMin = syMin
        renderingParameters.trimYMax = syMax
        renderingParameters.trimZMin = szMin
        renderingParameters.trimZMax = szMax

        argumentManager?.markAsNeedsUpdate(argumentIndex: .renderParams)
    }

    func updateClipPlanes(_ plane0: SIMD4<Float>,
                          _ plane1: SIMD4<Float>,
                          _ plane2: SIMD4<Float>) {
        let epsilon: Float = 1e-5
        let current0 = renderingParameters.clipPlane0
        let current1 = renderingParameters.clipPlane1
        let current2 = renderingParameters.clipPlane2

        let changed = simd_length(current0 - plane0) > epsilon ||
                      simd_length(current1 - plane1) > epsilon ||
                      simd_length(current2 - plane2) > epsilon

        if !changed {
            return
        }

        renderingParameters.clipPlane0 = plane0
        renderingParameters.clipPlane1 = plane1
        renderingParameters.clipPlane2 = plane2

        argumentManager?.markAsNeedsUpdate(argumentIndex: .renderParams)
    }

    func updateClipBoxQuaternion(_ quaternion: simd_quatf) {
        let epsilon: Float = 1e-4
        let normalized: simd_quatf

        if simd_length(quaternion.vector) <= epsilon {
            normalized = simd_quatf()
        } else {
            normalized = simd_normalize(quaternion)
        }

        let newVector = normalized.vector
        let current = renderingParameters.clipBoxQuaternion
        if simd_length(current - newVector) <= epsilon {
            return
        }

        renderingParameters.clipBoxQuaternion = newVector
        argumentManager?.markAsNeedsUpdate(argumentIndex: .renderParams)
    }

    func render(volume: MTLTexture,
                uniforms: VolumeCubeMaterial.Uniforms,
                camera: VolumeCameraParameters,
                outputSize: CGSize) -> MTLTexture? {
        guard outputSize.width > 0,
              outputSize.height > 0 else {
            Logger.log("Render abortado: outputSize inválido \(outputSize.width)x\(outputSize.height).",
                       level: .debug,
                       category: "VolumeCompute")
            return nil
        }

        guard let argumentManager else {
            Logger.log("Argument encoder indisponível para a renderização compute.",
                       level: .error,
                       category: "VolumeCompute")
            return nil
        }

        let width = max(1, Int(ceil(outputSize.width)))
        let height = max(1, Int(ceil(outputSize.height)))

        targetViewSize = UInt16(clamping: width)
        updateRenderingParameters(with: uniforms,
                                  camera: camera,
                                  volumeTexture: volume,
                                  outputWidth: width,
                                  outputHeight: height)

        if currentVolumeTexture !== volume {
            argumentManager.markAsNeedsUpdate(argumentIndex: .mainTexture)
            currentVolumeTexture = volume
        }

        let transferTextures = gatherTransferTextures()

        logRenderingParametersState(prefix: "Encoding render params")
        argumentManager.encodeTexture(texture: volume, argumentIndex: .mainTexture)
        argumentManager.encode(&renderingParameters, argumentIndex: .renderParams)

        argumentManager.encodeOutputTexture(width: width, height: height)
        guard let outputTexture = argumentManager.outputTexture else {
            Logger.log("Falha ao preparar textura de saída para renderização compute.",
                       level: .error,
                       category: "VolumeCompute")
            return nil
        }

        ensureToneBuffers()
        for (buffer, index) in zip(toneBuffers, toneArgumentIndices) {
            if let buffer {
                argumentManager.encode(buffer, argumentIndex: index)
            }
        }

        argumentManager.encode(&optionFlags, argumentIndex: .optionValue)
        debugLogOptionFlags()

        var quaternionCopy = quaternion
        argumentManager.encode(&quaternionCopy, argumentIndex: .quaternion)

        argumentManager.encode(&targetViewSize, argumentIndex: .targetViewSize)
        argumentManager.encodeSampler(filter: .linear)

        var pointCount: UInt16 = 0
        argumentManager.encode(&pointCount, argumentIndex: .pointSetCountBuffer)

        var selectedPoint: UInt16 = 0
        argumentManager.encode(&selectedPoint, argumentIndex: .pointSetSelectedBuffer)

        let defaultPoints = defaultPointCoordinates()
        argumentManager.encodeArray(defaultPoints,
                                    argumentIndex: .pointCoordsBuffer,
                                    capacity: defaultPoints.count)

        let transferIndices: [ArgumentEncoderManager.ArgumentIndex] = [
            .transferTextureCh1,
            .transferTextureCh2,
            .transferTextureCh3,
            .transferTextureCh4
        ]
        for (index, argumentIndex) in transferIndices.enumerated() {
            if index < transferTextures.count {
                argumentManager.encodeTexture(texture: transferTextures[index],
                                              argumentIndex: argumentIndex)
            }
        }

        let cameraUniforms = makeCameraUniforms(for: camera)
        updateCameraBuffer(with: cameraUniforms)

        let dispatchConfiguration = resolveThreadgroupConfiguration(width: outputTexture.width,
                                                                    height: outputTexture.height,
                                                                    argumentManager: argumentManager,
                                                                    volumeTexture: volume,
                                                                    transferTextures: transferTextures,
                                                                    outputTexture: outputTexture)

        let centralRayInfo = makeCentralRayDebugInfo(cameraUniforms: cameraUniforms,
                                                     width: outputTexture.width,
                                                     height: outputTexture.height)
        logPreDispatchDiagnostics(uniforms: renderingParameters.material,
                                  outputTexture: outputTexture,
                                  rayInfo: centralRayInfo)

        if renderingParameters.material.voxelMinValue >= renderingParameters.material.voxelMaxValue {
            Logger.log("HU window inválido: voxelMinValue=\(renderingParameters.material.voxelMinValue) " +
                       "≥ voxelMaxValue=\(renderingParameters.material.voxelMaxValue).",
                       level: .warn,
                       category: "VolumeCompute")
            assert(renderingParameters.material.voxelMinValue < renderingParameters.material.voxelMaxValue,
                   "HU window invalida o volume inteiro")
        }

        if let rayInfo = centralRayInfo, rayInfo.tExit <= rayInfo.tEnter {
            let message = String(format: "Central ray com interseção degenerada: tEnter=%.4f ≥ tExit=%.4f.",
                                 rayInfo.tEnter,
                                 rayInfo.tExit)
            Logger.log(message, level: .warn, category: "VolumeCompute")
            assert(rayInfo.tExit > rayInfo.tEnter, "Interseção inválida: tEnter ≥ tExit")
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            return nil
        }
        commandBuffer.label = "VolumeComputeCommandBuffer"

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            commandBuffer.commit()
            return nil
        }
        encoder.label = "VolumeComputeEncoder"
#if DEBUG
        if AppConfig.IS_DEBUG_MODE {
            debugLogBindingsBeforeEncode(argumentManager: argumentManager,
                                         volumeTexture: volume,
                                         transferTextures: transferTextures,
                                         outputTexture: outputTexture)
        }
#endif
        prepareEncoder(encoder,
                       argumentManager: argumentManager,
                       volumeTexture: volume,
                       transferTextures: transferTextures,
                       outputTexture: outputTexture)

        dispatch(configuration: dispatchConfiguration,
                 width: outputTexture.width,
                 height: outputTexture.height,
                 encoder: encoder,
                 argumentManager: argumentManager)

        encoder.endEncoding()

        CommandBufferProfiler.captureTimes(for: commandBuffer,
                                           label: "VolumeCompute",
                                           category: "VolumeCompute")

        let cpuStart = CFAbsoluteTimeGetCurrent()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        let cpuEnd = CFAbsoluteTimeGetCurrent()
        lastRenderTimings = commandBuffer.timings(cpuStart: cpuStart, cpuEnd: cpuEnd)

        if commandBuffer.status == .error {
            if let error = commandBuffer.error {
                Logger.log("VolumeComputeRenderer falhou: \(error.localizedDescription)",
                           level: .error,
                           category: "VolumeCompute")
            } else {
                Logger.log("VolumeComputeRenderer falhou sem detalhes adicionais.",
                           level: .error,
                           category: "VolumeCompute")
            }
            return nil
        }

        frameIndex &+= 1
        return outputTexture
    }

    func readback(texture: MTLTexture) -> Data? {
        let bytesPerPixel = 4
        let width = texture.width
        let height = texture.height
        let bytesPerRow = width * bytesPerPixel
        let bytesPerImage = bytesPerRow * height

        guard let buffer = device.makeBuffer(length: bytesPerImage, options: .storageModeShared) else {
            Logger.log("Falha ao criar buffer para readback da textura compute.",
                       level: .error,
                       category: "VolumeCompute")
            return nil
        }

        guard let blitCommandBuffer = commandQueue.makeCommandBuffer(),
              let blitEncoder = blitCommandBuffer.makeBlitCommandEncoder() else {
            Logger.log("Falha ao preparar blit encoder para readback da textura compute.",
                       level: .error,
                       category: "VolumeCompute")
            return nil
        }

        let size = MTLSize(width: width, height: height, depth: 1)
        blitEncoder.copy(from: texture,
                         sourceSlice: 0,
                         sourceLevel: 0,
                         sourceOrigin: MTLOriginMake(0, 0, 0),
                         sourceSize: size,
                         to: buffer,
                         destinationOffset: 0,
                         destinationBytesPerRow: bytesPerRow,
                         destinationBytesPerImage: bytesPerImage)
        blitEncoder.endEncoding()

        blitCommandBuffer.commit()
        blitCommandBuffer.waitUntilCompleted()

        if blitCommandBuffer.status == .error {
            if let error = blitCommandBuffer.error {
                Logger.log("Readback da textura compute falhou: \(error.localizedDescription)",
                           level: .error,
                           category: "VolumeCompute")
            }
            return nil
        }

        let pointer = buffer.contents()
        return Data(bytes: pointer, count: bytesPerImage)
    }

    func updateToneCurve(channel: Int, values: [Float]) {
        ensureToneBuffers()
        guard toneBuffers.indices.contains(channel),
              let buffer = toneBuffers[channel] else {
            return
        }

        let stride = MemoryLayout<Float>.stride
        let capacity = buffer.length / stride
        let copyCount = min(values.count, capacity)

        if copyCount > 0 {
            values.withUnsafeBytes { rawBuffer in
                if let baseAddress = rawBuffer.baseAddress {
                    memcpy(buffer.contents(), baseAddress, copyCount * stride)
                }
            }
        }

        if copyCount < capacity {
            let pointer = buffer.contents().bindMemory(to: Float.self, capacity: capacity)
            let fillValue = copyCount > 0 ? values[copyCount - 1] : 0
            for index in copyCount..<capacity {
                pointer[index] = fillValue
            }
        }

        if let argumentManager, channel < toneArgumentIndices.count {
            argumentManager.markAsNeedsUpdate(argumentIndex: toneArgumentIndices[channel])
        }
    }

    func updateToneCurves(_ curves: [[Float]]) {
        for (index, curve) in curves.enumerated() {
            updateToneCurve(channel: index, values: curve)
        }
    }
}

extension VolumeComputeRenderer {
    static func computeRayDirection(cameraLocal01: SIMD3<Float>,
                                    pixelLocal01: SIMD3<Float>) -> SIMD3<Float> {
        let delta = pixelLocal01 - cameraLocal01
        let length = simd_length(delta)
        guard length > 1e-6 else {
            return SIMD3<Float>(repeating: 0)
        }
        return delta / length
    }

    static func rayBoxIntersection(rayOrigin: SIMD3<Float>,
                                    rayDirection: SIMD3<Float>,
                                    boxMin: SIMD3<Float>,
                                    boxMax: SIMD3<Float>) -> SIMD2<Float> {
        let epsilon: Float = 1e-6
        let safeDirection = SIMD3<Float>(
            abs(rayDirection.x) < epsilon ? (rayDirection.x >= 0 ? epsilon : -epsilon) : rayDirection.x,
            abs(rayDirection.y) < epsilon ? (rayDirection.y >= 0 ? epsilon : -epsilon) : rayDirection.y,
            abs(rayDirection.z) < epsilon ? (rayDirection.z >= 0 ? epsilon : -epsilon) : rayDirection.z
        )

        let tMin = (boxMin - rayOrigin) / safeDirection
        let tMax = (boxMax - rayOrigin) / safeDirection
        let t1 = simd_min(tMin, tMax)
        let t2 = simd_max(tMin, tMax)
        let tNear = max(max(t1.x, t1.y), t1.z)
        let tFar = min(min(t2.x, t2.y), t2.z)
        return SIMD2<Float>(tNear, tFar)
    }
}

private extension VolumeComputeRenderer {
    struct CentralRayDebugInfo {
        let cameraLocal01: SIMD3<Float>
        let rayDirection: SIMD3<Float>
        let tEnter: Float
        let tExit: Float

        var intersectsUnitCube: Bool {
            tExit > max(tEnter, 0.0)
        }
    }

    func logPreDispatchDiagnostics(uniforms: VolumeCubeMaterial.Uniforms,
                                   outputTexture: MTLTexture,
                                   rayInfo: CentralRayDebugInfo?) {
        let dimsDescription = "\(uniforms.dimX)x\(uniforms.dimY)x\(uniforms.dimZ)"
        let methodDescription = methodDescription(for: uniforms.method)
        let huWindow = "[\(uniforms.voxelMinValue), \(uniforms.voxelMaxValue)]"
        Logger.log("Dispatch compute → dim=\(dimsDescription), output=\(outputTexture.width)x\(outputTexture.height), " +
                   "method=\(methodDescription), voxelRange=\(huWindow), HUWindow=\(huWindow)",
                   level: .debug,
                   category: "VolumeCompute")

        guard let rayInfo else {
            Logger.log("Central ray diagnostics indisponíveis (falha ao projetar pixel central).",
                       level: .debug,
                       category: "VolumeCompute")
            return
        }

        let intersectsTag = rayInfo.intersectsUnitCube ? "true" : "false"
        let message = String(format: "Central ray cameraLocal01=(%.3f, %.3f, %.3f) dir=(%.3f, %.3f, %.3f) " +
                                     "tEnter=%.4f tExit=%.4f intersects=%@",
                                 rayInfo.cameraLocal01.x,
                                 rayInfo.cameraLocal01.y,
                                 rayInfo.cameraLocal01.z,
                                 rayInfo.rayDirection.x,
                                 rayInfo.rayDirection.y,
                                 rayInfo.rayDirection.z,
                                 rayInfo.tEnter,
                                 rayInfo.tExit,
                                 intersectsTag)
        Logger.log(message, level: .debug, category: "VolumeCompute")

        if !rayInfo.intersectsUnitCube {
            Logger.log("Central ray não cruza o volume unitário [0,1]^3; verificar matriz de câmera/modelo.",
                       level: .warn,
                       category: "VolumeCompute")
        }
    }

    func methodDescription(for methodId: Int32) -> String {
        if let method = VolumeCubeMaterial.Method.allCases.first(where: { $0.idInt32 == methodId }) {
            return method.rawValue
        }
        return "unknown(\(methodId))"
    }

    func makeCentralRayDebugInfo(cameraUniforms: CameraUniforms,
                                 width: Int,
                                 height: Int) -> CentralRayDebugInfo? {
        guard width > 0, height > 0 else {
            return nil
        }

        let centerPixelX = max(0, width / 2)
        let centerPixelY = max(0, height / 2)
        let widthF = Float(width)
        let heightF = Float(height)
        let x = (Float(centerPixelX) + 0.5) / widthF
        let flippedY = Float(height - 1 - centerPixelY)
        let y = (flippedY + 0.5) / heightF

        let ndc = SIMD2<Float>(x * 2.0 - 1.0, y * 2.0 - 1.0)
        let clipFar = SIMD4<Float>(ndc.x, ndc.y, 1.0, 1.0)

        var worldFar = cameraUniforms.inverseViewProjectionMatrix * clipFar

        guard abs(worldFar.w) > Float.leastNonzeroMagnitude else {
            return nil
        }

        worldFar /= worldFar.w

        let localFar4 = cameraUniforms.inverseModelMatrix * worldFar
        guard abs(localFar4.w) > Float.leastNonzeroMagnitude else {
            return nil
        }

        let localFar = localFar4.xyz / localFar4.w + SIMD3<Float>(repeating: 0.5)
        let cameraLocal01 = cameraUniforms.cameraPositionLocal + SIMD3<Float>(repeating: 0.5)
        let directionVector = localFar - cameraLocal01
        let directionLength = simd_length(directionVector)

        guard directionLength > Float.leastNonzeroMagnitude else {
            return nil
        }

        let rayDirection = directionVector / directionLength
        let intersection = intersectUnitCubeRay(origin: cameraLocal01, direction: rayDirection)
        let tEnter = max(intersection.x, 0.0)
        let tExit = intersection.y

        return CentralRayDebugInfo(cameraLocal01: cameraLocal01,
                                   rayDirection: rayDirection,
                                   tEnter: tEnter,
                                   tExit: tExit)
    }

    func intersectUnitCubeRay(origin: SIMD3<Float>, direction: SIMD3<Float>) -> SIMD2<Float> {
        let boxMin = SIMD3<Float>(repeating: 0.0)
        let boxMax = SIMD3<Float>(repeating: 1.0)
        let tMin = (boxMin - origin) / direction
        let tMax = (boxMax - origin) / direction
        let t1 = simd_min(tMin, tMax)
        let t2 = simd_max(tMin, tMax)
        let tNear = max(t1.x, max(t1.y, t1.z))
        let tFar = min(t2.x, min(t2.y, t2.z))
        return SIMD2<Float>(tNear, tFar)
    }

    func prepareEncoder(_ encoder: MTLComputeCommandEncoder,
                        argumentManager: ArgumentEncoderManager,
                        volumeTexture: MTLTexture,
                        transferTextures: [MTLTexture],
                        outputTexture: MTLTexture) {
        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(argumentManager.argumentBuffer, offset: 0, index: 0)
        if let cameraBuffer {
            encoder.setBuffer(cameraBuffer, offset: 0, index: 1)
            encoder.useResource(cameraBuffer, usage: .read)
        }

        encoder.useResource(argumentManager.argumentBuffer, usage: .read)
        encoder.useResource(volumeTexture, usage: .read)
        for texture in transferTextures {
            encoder.useResource(texture, usage: .read)
        }
        for buffer in toneBuffers {
            if let buffer {
                encoder.useResource(buffer, usage: .read)
            }
        }
        encoder.useResource(outputTexture, usage: .write)
    }

#if DEBUG
    private func debugLogBindingsBeforeEncode(argumentManager: ArgumentEncoderManager,
                                              volumeTexture: MTLTexture,
                                              transferTextures: [MTLTexture],
                                              outputTexture: MTLTexture) {
        var lines: [String] = []
        lines.append("[VolumeCompute] Bindings before encoder.set*: argument buffer index 0 -> \(debugDescription(for: argumentManager.argumentBuffer))")

        if let cameraBuffer {
            lines.append("[VolumeCompute] Bindings before encoder.set*: camera buffer index 1 -> \(debugDescription(for: cameraBuffer))")
        } else {
            lines.append("[VolumeCompute] Bindings before encoder.set*: camera buffer index 1 -> nil")
        }

        let orderedIndices = argumentBindingMetadata.keys.sorted { $0.rawValue < $1.rawValue }

        for argumentIndex in orderedIndices {
            guard let metadata = argumentBindingMetadata[argumentIndex] else { continue }
            let resourceDescription: String
            switch argumentIndex {
            case .mainTexture:
                resourceDescription = debugDescription(for: volumeTexture)
            case .outputTexture:
                resourceDescription = debugDescription(for: outputTexture)
            case .transferTextureCh1, .transferTextureCh2, .transferTextureCh3, .transferTextureCh4:
                let texturePosition = argumentIndex.rawValue - ArgumentEncoderManager.ArgumentIndex.transferTextureCh1.rawValue
                if texturePosition >= 0 && texturePosition < transferTextures.count {
                    resourceDescription = debugDescription(for: transferTextures[texturePosition])
                } else {
                    resourceDescription = "missing"
                }
            case .sampler:
                resourceDescription = debugSamplerDescription(argumentManager.sampler)
            default:
                if let buffer = argumentManager.debugBoundBuffer(for: argumentIndex) {
                    resourceDescription = debugDescription(for: buffer)
                } else {
                    resourceDescription = "nil"
                }
            }

            let needsUpdate = argumentManager.debugNeedsUpdateState(for: argumentIndex)
            let needsUpdateString: String
            if let needsUpdate {
                needsUpdateString = needsUpdate ? "needsUpdate=true" : "needsUpdate=false"
            } else {
                needsUpdateString = "needsUpdate=?"
            }

            let fragmentInfo: String
            if let slot = metadata.fragmentSlot {
                fragmentInfo = ", fragment slot \(slot)"
            } else {
                fragmentInfo = ""
            }

            lines.append("  [\(argumentIndex.rawValue)] \(metadata.role) -> \(resourceDescription) (\(needsUpdateString)\(fragmentInfo))")
        }

        Logger.log(lines.joined(separator: "\n"), level: .debug, category: "VolumeCompute")
    }

    private func debugDescription(for resource: MTLResource?) -> String {
        guard let resource else { return "nil" }
        let label = resource.label ?? "<no label>"
        return "\(type(of: resource)):\(label)"
    }

    private func debugSamplerDescription(_ sampler: MTLSamplerState?) -> String {
        guard let sampler else { return "nil" }
        let label = sampler.label ?? "<no label>"
        return "MTLSamplerState:\(label)"
    }
#endif

    func dispatch(configuration: ThreadgroupDispatchConfiguration,
                  width: Int,
                  height: Int,
                  encoder: MTLComputeCommandEncoder,
                  argumentManager: ArgumentEncoderManager) {
        let threadsPerThreadgroup = configuration.threadsPerThreadgroup
        if AppConfig.IS_DEBUG_MODE {
            let threadgroupSummary = "\(threadsPerThreadgroup.width)x\(threadsPerThreadgroup.height)x\(threadsPerThreadgroup.depth)"
            let argumentSummary = argumentManager.debugStateSummary()
            Logger.log(
                "Dispatch compute: target=\(width)x\(height) threadsPerGroup=\(threadgroupSummary) nonUniform=\(supportsNonUniformThreadgroups) | ArgEncoder: \(argumentSummary)",
                level: .debug,
                category: "VolumeCompute"
            )
        }
        if supportsNonUniformThreadgroups {
            let threads = MTLSize(width: width, height: height, depth: 1)
            encoder.dispatchThreads(threads, threadsPerThreadgroup: threadsPerThreadgroup)
        } else {
            let threadgroups = configuration.threadgroupsPerGrid(forWidth: width, height: height)
            encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        }
    }

    func resolveThreadgroupConfiguration(width: Int,
                                         height: Int,
                                         argumentManager: ArgumentEncoderManager,
                                         volumeTexture: MTLTexture,
                                         transferTextures: [MTLTexture],
                                         outputTexture: MTLTexture) -> ThreadgroupDispatchConfiguration {
        let fallback = ThreadgroupDispatchConfiguration.default(for: pipelineState)

        guard let configuration = dispatchOptimizer.configuration(
            for: pipelineState,
            width: width,
            height: height,
            benchmark: { candidate in
                benchmarkThreadgroup(candidate,
                                     width: width,
                                     height: height,
                                     argumentManager: argumentManager,
                                     volumeTexture: volumeTexture,
                                     transferTextures: transferTextures,
                                     outputTexture: outputTexture)
            }
        ) else {
            return fallback
        }

        return configuration
    }

    func benchmarkThreadgroup(_ configuration: ThreadgroupDispatchConfiguration,
                              width: Int,
                              height: Int,
                              argumentManager: ArgumentEncoderManager,
                              volumeTexture: MTLTexture,
                              transferTextures: [MTLTexture],
                              outputTexture: MTLTexture) -> CommandBufferTimings? {
        let bufferLabel = "VolumeComputeBenchmark-\(configuration.description)"
        guard let benchmarkBuffer = commandQueue.makeCommandBuffer() else {
            Logger.log("Falha ao criar command buffer para benchmark \(configuration.description).",
                       level: .warn,
                       category: "ThreadgroupDispatch")
            return nil
        }
        benchmarkBuffer.label = bufferLabel

        guard let encoder = benchmarkBuffer.makeComputeCommandEncoder() else {
            benchmarkBuffer.commit()
            Logger.log("Falha ao criar command encoder para benchmark \(configuration.description).",
                       level: .warn,
                       category: "ThreadgroupDispatch")
            return nil
        }
        encoder.label = "VolumeComputeBenchmarkEncoder-\(configuration.description)"

        prepareEncoder(encoder,
                       argumentManager: argumentManager,
                       volumeTexture: volumeTexture,
                       transferTextures: transferTextures,
                       outputTexture: outputTexture)

        dispatch(configuration: configuration,
                 width: width,
                 height: height,
                 encoder: encoder,
                 argumentManager: argumentManager)

        encoder.endEncoding()

        let cpuStart = CFAbsoluteTimeGetCurrent()
        benchmarkBuffer.commit()
        benchmarkBuffer.waitUntilCompleted()
        let cpuEnd = CFAbsoluteTimeGetCurrent()

        if let error = benchmarkBuffer.error {
            Logger.log("Dispatch benchmark falhou para \(configuration.description): \(error.localizedDescription)",
                       level: .warn,
                       category: "ThreadgroupDispatch")
            return nil
        }

        return benchmarkBuffer.timings(cpuStart: cpuStart, cpuEnd: cpuEnd)
    }

    func updateRenderingParameters(with uniforms: VolumeCubeMaterial.Uniforms,
                                   camera: VolumeCameraParameters,
                                   volumeTexture: MTLTexture,
                                   outputWidth: Int,
                                   outputHeight: Int) {
        var adjustedUniforms = uniforms

        let textureDim = SIMD3<Int32>(Int32(volumeTexture.width),
                                      Int32(volumeTexture.height),
                                      Int32(volumeTexture.depth))

        if adjustedUniforms.dimX != textureDim.x ||
            adjustedUniforms.dimY != textureDim.y ||
            adjustedUniforms.dimZ != textureDim.z {
            if AppConfig.IS_DEBUG_MODE {
                Logger.log("Uniform dims (\(adjustedUniforms.dimX), \(adjustedUniforms.dimY), \(adjustedUniforms.dimZ)) " +
                           "≠ texture dims (\(textureDim.x), \(textureDim.y), \(textureDim.z)); substituindo pelos valores da textura.",
                           level: .warn,
                           category: "ComputeDiagnostics")
            }
            adjustedUniforms.dimX = textureDim.x
            adjustedUniforms.dimY = textureDim.y
            adjustedUniforms.dimZ = textureDim.z
        }

        renderingParameters.material = adjustedUniforms
        renderingParameters.scale = 1.0
        renderingParameters.zScale = 1.0
        renderingParameters.sliceMax = UInt16(clamping: volumeTexture.depth)
        renderingParameters.sliceNo = min(renderingParameters.sliceNo, renderingParameters.sliceMax)
        renderingParameters.trimXMin = min(max(renderingParameters.trimXMin, 0.0), 1.0)
        renderingParameters.trimXMax = min(max(renderingParameters.trimXMax, 0.0), 1.0)
        renderingParameters.trimYMin = min(max(renderingParameters.trimYMin, 0.0), 1.0)
        renderingParameters.trimYMax = min(max(renderingParameters.trimYMax, 0.0), 1.0)
        renderingParameters.trimZMin = min(max(renderingParameters.trimZMin, 0.0), 1.0)
        renderingParameters.trimZMax = min(max(renderingParameters.trimZMax, 0.0), 1.0)
        renderingParameters.viewSize = targetViewSize
        let quality = max(Float(adjustedUniforms.renderingQuality), 1.0)
        let minimumStep: Float = 1.0 / 4096.0
        renderingParameters.renderingStep = max(1.0 / quality, minimumStep)
        renderingParameters.earlyTerminationThreshold = earlyTerminationThreshold
        renderingParameters.adaptiveGradientThreshold = adaptiveThreshold
        renderingParameters.jitterAmount = jitterAmount
        renderingParameters.intensityRatio = channelIntensities
        renderingParameters.light = 1.0
        renderingParameters.shade = 1.0
        renderingParameters.dicomOrientationRow = SIMD4<Float>(1, 0, 0, 0)
        renderingParameters.dicomOrientationColumn = SIMD4<Float>(0, 1, 0, 0)
        renderingParameters.dicomOrientationNormal = SIMD4<Float>(0, 0, 1, 0)
        renderingParameters.dicomOrientationActive = 0
        renderingParameters.renderingMethod = UInt8(clamping: Int(adjustedUniforms.method))
        renderingParameters.backgroundColor = SIMD3<Float>(repeating: 0)
        renderingParameters.material.renderingQuality = max(adjustedUniforms.renderingQuality, 1)

        logRenderingParametersState(prefix: "Render params updated")
        if AppConfig.IS_DEBUG_MODE {
            let dims = "\(adjustedUniforms.dimX)x\(adjustedUniforms.dimY)x\(adjustedUniforms.dimZ)"
            let methodId = Int(adjustedUniforms.method)
            let methodName = VolumeCubeMaterial.Method.allCases.first(where: { $0.idInt32 == adjustedUniforms.method })?.rawValue ?? "unknown"
            let trimSummary = String(
                format: "X=[%.3f, %.3f] Y=[%.3f, %.3f] Z=[%.3f, %.3f]",
                Double(renderingParameters.trimXMin),
                Double(renderingParameters.trimXMax),
                Double(renderingParameters.trimYMin),
                Double(renderingParameters.trimYMax),
                Double(renderingParameters.trimZMin),
                Double(renderingParameters.trimZMax)
            )
            let clipPlanesActive: [Bool] = [
                simd_length(renderingParameters.clipPlane0) > 1e-5,
                simd_length(renderingParameters.clipPlane1) > 1e-5,
                simd_length(renderingParameters.clipPlane2) > 1e-5
            ]
            let clipPlaneSummary = clipPlanesActive.enumerated().map { index, isActive in
                "P\(index)=\(isActive)"
            }.joined(separator: ",")
            let clipBoxDefault = SIMD4<Float>(0, 0, 0, 1)
            let clipBoxActive = simd_length(renderingParameters.clipBoxQuaternion - clipBoxDefault) > 1e-4
            let windowRange = "[\(adjustedUniforms.voxelMinValue), \(adjustedUniforms.voxelMaxValue)]"
            let datasetRange = "[\(adjustedUniforms.datasetMinValue), \(adjustedUniforms.datasetMaxValue)]"
            Logger.log(
                "Render params: dims=\(dims) windowRange=\(windowRange) datasetRange=\(datasetRange) method=#\(methodId) (\(methodName)) trims=\(trimSummary) clipPlanes={\(clipPlaneSummary)} clipBox=\(clipBoxActive)",
                level: .debug,
                category: "VolumeCompute"
            )
        }
    }

    func makeCameraUniforms(for camera: VolumeCameraParameters) -> CameraUniforms {
        CameraUniforms(modelMatrix: camera.modelMatrix,
                       inverseModelMatrix: camera.inverseModelMatrix,
                       inverseViewProjectionMatrix: camera.inverseViewProjectionMatrix,
                       cameraPositionLocal: camera.cameraPositionLocal,
                       frameIndex: frameIndex)
    }

    func updateCameraBuffer(with uniforms: CameraUniforms) {
        var uniforms = uniforms
        let size = MemoryLayout<CameraUniforms>.stride
        if cameraBuffer == nil || cameraBuffer?.length != size {
            cameraBuffer = device.makeBuffer(length: size, options: .storageModeShared)
            cameraBuffer?.label = "VolumeComputeCameraUniforms"
        }
        guard let buffer = cameraBuffer else { return }
        memcpy(buffer.contents(), &uniforms, size)
    }

    func ensureToneBuffers() {
        let expectedLength = MemoryLayout<Float>.stride * toneSampleCount
        let capacity = toneSampleCount

        for index in 0..<toneBuffers.count {
            let needsCreation = toneBuffers[index] == nil || toneBuffers[index]?.length != expectedLength
            if needsCreation {
                toneBuffers[index] = device.makeBuffer(length: expectedLength,
                                                       options: [.storageModeShared])
                toneBuffers[index]?.label = "ToneBuffer_\(index + 1)"
                if let pointer = toneBuffers[index]?.contents().bindMemory(to: Float.self, capacity: capacity) {
                    let maxIndex = max(capacity - 1, 1)
                    for sample in 0..<capacity {
                        pointer[sample] = Float(sample) / Float(maxIndex)
                    }
                }
                if let argumentManager, index < toneArgumentIndices.count {
                    argumentManager.markAsNeedsUpdate(argumentIndex: toneArgumentIndices[index])
                }
            }
        }
    }

    func logRenderingParametersState(prefix: String) {
        guard AppConfig.IS_DEBUG_MODE else { return }

        let material = renderingParameters.material
        let dims = (material.dimX, material.dimY, material.dimZ)
        let voxelRange = (material.voxelMinValue, material.voxelMaxValue)
        let stepSize = renderingParameters.renderingStep
        let options = optionFlags
        let early = renderingParameters.earlyTerminationThreshold

        let message = String(format: "%@: dims=%dx%dx%d voxelRange=[%d,%d] step=%.6f options=0x%04X early=%.4f",
                             prefix,
                             dims.0,
                             dims.1,
                             dims.2,
                             voxelRange.0,
                             voxelRange.1,
                             Double(stepSize),
                             Int(options),
                             Double(early))

        Logger.log(message,
                   level: .debug,
                   category: "ComputeDiagnostics")
    }

    func debugLogOptionFlags() {
        guard AppConfig.IS_DEBUG_MODE else { return }

        let hexValue = String(format: "0x%04X", optionFlags)
        var activeFlags: [String] = []

        if (optionFlags & Self.optionAdaptiveMask) != 0 {
            activeFlags.append("adaptive")
        }
        if (optionFlags & Self.optionDensityDebugMask) != 0 {
            activeFlags.append("densityDebug")
        }

        let description = activeFlags.isEmpty ? "none" : activeFlags.joined(separator: ", ")
        Logger.log("VolumeCompute optionFlags=\(hexValue) [\(description)]",
                   level: .debug,
                   category: "VolumeCompute")
    }

    func defaultPointCoordinates() -> [float3] {
        [float3(0, 0, 0), float3(0, 0, 0)]
    }

    func transferTexture(forChannel index: Int) -> MTLTexture? {
        if index < channelTransferTextures.count, let texture = channelTransferTextures[index] {
            return texture
        }
        return fallbackTransferTexture
    }

    func gatherTransferTextures() -> [MTLTexture] {
        var textures: [MTLTexture] = []
        for index in 0..<channelTransferTextures.count {
            if let texture = transferTexture(forChannel: index) {
                textures.append(texture)
            }
        }
        if textures.count < channelTransferTextures.count, let fallback = fallbackTransferTexture {
            while textures.count < channelTransferTextures.count {
                textures.append(fallback)
            }
        }
        return textures
    }

    static func makeFallbackTransferTexture(device: MTLDevice) -> MTLTexture? {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float,
                                                                  width: 2,
                                                                  height: 1,
                                                                  mipmapped: false)
        descriptor.usage = [.shaderRead]
        descriptor.storageMode = .shared

        guard let texture = device.makeTexture(descriptor: descriptor) else {
            return nil
        }

        let zeros = [SIMD4<Float>](repeating: SIMD4<Float>(repeating: 0), count: descriptor.width * descriptor.height)
        zeros.withUnsafeBytes { rawBuffer in
            if let baseAddress = rawBuffer.baseAddress {
                texture.replace(region: MTLRegionMake2D(0, 0, descriptor.width, descriptor.height),
                                mipmapLevel: 0,
                                withBytes: baseAddress,
                                bytesPerRow: MemoryLayout<SIMD4<Float>>.stride * descriptor.width)
            }
        }
        texture.label = "TransferTextureFallback"
        return texture
    }
}
