//
//  MPRPlaneMaterial.swift
//  VolumeRendering-iOS
//
//  Keeps the legacy compute-driven MPR material alive for the demo while its logic transitions into MTK.
//  Thales Matheus Mendonça Santos — October 2025
//

import Metal
import SceneKit
import simd
import UIKit
import CoreImage

final class MPRPlaneMaterial: SCNMaterial {

    enum BlendMode: Int32, CaseIterable {
        case single = 0, mip = 1, minip = 2, mean = 3
    }

    private enum Axis {
        case axial, coronal, sagittal, oblique
    }

    struct Uniforms: sizeable {
        // Atenção à ordem/alinhamento (pad para 16 bytes p/ float3).
        var voxelMinValue: Int32 = -1024
        var voxelMaxValue: Int32 =  3071
        var datasetMinValue: Int32 = -1024
        var datasetMaxValue: Int32 =  3071
        var blendMode: Int32 = BlendMode.single.rawValue
        var numSteps: Int32  = 1
        var slabHalf: Float  = 0
        var _pad0: float3    = .zero

        var planeOrigin: float3 = .zero
        var _pad1: Float = 0
        var planeX: float3 = float3(1, 0, 0)
        var _pad2: Float = 0
        var planeY: float3 = float3(0, 1, 0)
        var overlayOpacity: Float = 0

        var useTFMpr: Int32 = 1
        var overlayEnabled: Int32 = 0
        var overlayChannel: Int32 = 0
        var overlayColor: float3 = float3(1, 0, 0)
        var _pad7: Float = 0
    }

    // MARK: - State

    private let device: MTLDevice
    private let computeRenderer: MPRComputeRenderer
    private let featureFlags: FeatureFlags
    private var textureFactory: VolumeTextureFactory
    private let uploadQueue: MTLCommandQueue
    private var uploadToken = UUID()
    private var pendingCompletion: (() -> Void)?

    private(set) var dimension: int3 = int3(1, 1, 1)
    private(set) var resolution: float3 = float3(1, 1, 1)

    private var volumeTexture: MTLTexture?
    private var transferTexture: MTLTexture?
    private var maskTexture: MTLTexture?
    private var uniforms = Uniforms()
    private var outputSize = SIMD2<Int>(1, 1)
    private var currentAxis: Axis = .axial
    private var needsRender = true

    private var lastRenderedTexture: MTLTexture?
    private let ciContext: CIContext
    private let colorSpace = CGColorSpaceCreateDeviceRGB()
    private var collapsedDimensionWarningIssued = false

    // MARK: - Init

    init(device: MTLDevice, featureFlags: FeatureFlags) {
        self.device = device
        self.featureFlags = featureFlags
        self.textureFactory = VolumeTextureFactory(part: .none, featureFlags: featureFlags)
        guard let queue = device.makeCommandQueue() else {
            fatalError("Não foi possível criar command queue para upload MPR.")
        }
        queue.label = "MPRTextureUploadQueue"
        self.uploadQueue = queue
        do {
            self.computeRenderer = try MPRComputeRenderer(device: device, featureFlags: featureFlags)
        } catch {
            fatalError("Failed to initialize MPR compute renderer: \(error)")
        }
        self.ciContext = CIContext(mtlDevice: device)
        super.init()
        configureMaterialDefaults()
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }

    // MARK: - Public API (dataset & parameters)

    func setPart(device: MTLDevice, part: VolumeCubeMaterial.BodyPart, completion: (() -> Void)? = nil) {
        apply(factory: VolumeTextureFactory(part: part, featureFlags: featureFlags),
              device: device,
              completion: completion)
    }

    func setDataset(device: MTLDevice, dataset: VolumeDataset, completion: (() -> Void)? = nil) {
        apply(factory: VolumeTextureFactory(dataset: dataset, featureFlags: featureFlags),
              device: device,
              completion: completion)
    }

    func setDataset(dimension: int3, resolution: float3) {
        self.dimension = dimension
        self.resolution = resolution
        updateOutputSize(for: currentAxis)
        markDirty()
        collapsedDimensionWarningIssued = false
    }

    func setHU(min minHU: Int32, max maxHU: Int32) {
        let datasetRange = textureFactory.dataset.intensityRange
        let lowerBound = Swift.min(minHU, maxHU)
        let upperBound = Swift.max(minHU, maxHU)
        uniforms.voxelMinValue = Swift.max(datasetRange.lowerBound, lowerBound)
        uniforms.voxelMaxValue = Swift.min(datasetRange.upperBound, upperBound)
        markDirty()
    }

    func setBlend(_ mode: BlendMode) {
        uniforms.blendMode = mode.rawValue
        markDirty()
    }

    func setSlab(thicknessInVoxels: Int, steps: Int) {
        let normal = normalize(cross(uniforms.planeX, uniforms.planeY))
        let dims = float3(Float(max(dimension.x, 1)),
                          Float(max(dimension.y, 1)),
                          Float(max(dimension.z, 1)))
        let normalInVoxels = float3(normal.x * dims.x,
                                    normal.y * dims.y,
                                    normal.z * dims.z)
        let denom = max(length(normalInVoxels), 1e-6)
        uniforms.slabHalf = 0.5 * Float(max(thicknessInVoxels, 0)) / denom
        uniforms.numSteps = Int32(max(1, steps))
        markDirty()
    }

    func setAxial(slice k: Int) {
        let kz = max(0, min(Int(dimension.z) - 1, k))
        let z = (Float(kz) + 0.5) / max(1, Float(dimension.z))
        uniforms.planeOrigin = float3(0, 0, z)
        uniforms.planeX = float3(1, 0, 0)
        uniforms.planeY = float3(0, 1, 0)
        currentAxis = .axial
        updateOutputSize(for: currentAxis)
        markDirty()
    }

    func setSagittal(column i: Int) {
        let ix = max(0, min(Int(dimension.x) - 1, i))
        let x = (Float(ix) + 0.5) / max(1, Float(dimension.x))
        uniforms.planeOrigin = float3(x, 0, 0)
        uniforms.planeX = float3(0, 1, 0)
        uniforms.planeY = float3(0, 0, 1)
        currentAxis = .sagittal
        updateOutputSize(for: currentAxis)
        markDirty()
    }

    func setCoronal(row j: Int) {
        let jy = max(0, min(Int(dimension.y) - 1, j))
        let y = (Float(jy) + 0.5) / max(1, Float(dimension.y))
        uniforms.planeOrigin = float3(0, y, 0)
        uniforms.planeX = float3(1, 0, 0)
        uniforms.planeY = float3(0, 0, 1)
        currentAxis = .coronal
        updateOutputSize(for: currentAxis)
        markDirty()
    }

    func setOblique(origin: float3, axisU: float3, axisV: float3) {
        uniforms.planeOrigin = origin
        uniforms.planeX = axisU
        uniforms.planeY = axisV
        currentAxis = .oblique
        updateOutputSize(for: currentAxis)
        markDirty()
    }

    func setUseTF(_ on: Bool) {
        uniforms.useTFMpr = on ? 1 : 0
        markDirty()
    }

    func setTransferFunction(_ texture: MTLTexture) {
        transferTexture = texture
        markDirty()
    }

    func setOverlayMask(_ texture: MTLTexture?) {
        maskTexture = texture
        uniforms.overlayEnabled = texture == nil ? 0 : 1
        markDirty()
    }

    func setOverlayOpacity(_ value: Float) {
        uniforms.overlayOpacity = max(0, min(1, value))
        markDirty()
    }

    func setOverlayColor(_ color: SIMD3<Float>) {
        let clamped = SIMD3<Float>(x: max(0, min(1, color.x)),
                                   y: max(0, min(1, color.y)),
                                   z: max(0, min(1, color.z)))
        uniforms.overlayColor = float3(clamped)
        markDirty()
    }

    func setOverlayChannel(_ channel: Int32) {
        uniforms.overlayChannel = channel
        markDirty()
    }

    func renderIfNeeded() {
        guard needsRender,
              outputSize.x > 0, outputSize.y > 0,
              let volumeTexture = volumeTexture
        else { return }

#if DEBUG
        if dimension == int3(1, 1, 1),
           let sceneMeta = SceneViewController.Instance.currentDatasetMeta(),
           sceneMeta.dimension != int3(1, 1, 1) {
            if !collapsedDimensionWarningIssued {
                Logger.log("MPR dataset dimension collapsed to 1³ while SceneViewController reports \(sceneMeta.dimension)",
                           level: .warn,
                           category: "MPRCompute")
                collapsedDimensionWarningIssued = true
            }
            needsRender = true
            return
        } else if dimension != int3(1, 1, 1) {
            collapsedDimensionWarningIssued = false
        }
#endif

        uniforms.overlayEnabled = (maskTexture != nil) ? 1 : 0
        var params = uniforms
        if transferTexture == nil {
            params.useTFMpr = 0
        }

        guard let texture = computeRenderer.render(volume: volumeTexture,
                                                    transfer: transferTexture,
                                                    mask: maskTexture,
                                                    uniforms: &params,
                                                    outputSize: outputSize) else {
            return
        }

        uniforms = params
        diffuse.contents = texture
        emission.contents = texture
        needsRender = false
        lastRenderedTexture = texture
    }

    func currentCGImage() -> CGImage? {
        renderIfNeeded()
        guard let texture = lastRenderedTexture,
              let ciImage = CIImage(mtlTexture: texture, options: [.colorSpace: colorSpace]) else {
            Logger.log("Falha ao criar CIImage para textura MPR (pixelFormat=\(String(describing: lastRenderedTexture?.pixelFormat)))",
                       level: .warn,
                       category: "MPRCompute")
            return nil
        }
        let rect = CGRect(x: 0, y: 0, width: texture.width, height: texture.height)
        return ciContext.createCGImage(ciImage, from: rect)
    }

    func currentSliceIndex(for axis: Int) -> Int {
        switch axis {
        case 0:
            return index(from: uniforms.planeOrigin.x, count: Int(dimension.x))
        case 1:
            return index(from: uniforms.planeOrigin.y, count: Int(dimension.y))
        default:
            return index(from: uniforms.planeOrigin.z, count: Int(dimension.z))
        }
    }

    private func index(from normalized: Float, count: Int) -> Int {
        guard count > 0 else { return 0 }
        let value = normalized * Float(count)
        let idx = Int(round(value - 0.5))
        return max(0, min(count - 1, idx))
    }

    // MARK: - Overrides (to retain compatibility with legacy KVC access)

    override func setValue(_ value: Any?, forKey key: String) {
        switch key {
        case "volume":
            if let property = value as? SCNMaterialProperty,
               let texture = property.contents as? MTLTexture {
                volumeTexture = texture
                markDirty()
            }
        case "transferColor":
            if let property = value as? SCNMaterialProperty,
               let texture = property.contents as? MTLTexture {
                transferTexture = texture
                markDirty()
            }
        default:
            super.setValue(value, forKey: key)
        }
    }

    // MARK: - Private helpers

    private func configureMaterialDefaults() {
        lightingModel = .constant
        isDoubleSided = true
        writesToDepthBuffer = false
        readsFromDepthBuffer = false
        cullMode = .back

        diffuse.wrapS = .clamp
        diffuse.wrapT = .clamp
        diffuse.magnificationFilter = .linear
        diffuse.minificationFilter = .linear
        diffuse.mipFilter = .none
        diffuse.contents = UIColor.black

        let emissionProperty = emission
        emissionProperty.wrapS = .clamp
        emissionProperty.wrapT = .clamp
        emissionProperty.magnificationFilter = .linear
        emissionProperty.minificationFilter = .linear
        emissionProperty.mipFilter = .none
        emissionProperty.contents = UIColor.black
    }

    private func markDirty() {
        needsRender = true
    }

    private func updateOutputSize(for axis: Axis) {
        switch axis {
        case .axial:
            outputSize = SIMD2(Int(dimension.x), Int(dimension.y))
        case .coronal:
            outputSize = SIMD2(Int(dimension.x), Int(dimension.z))
        case .sagittal:
            outputSize = SIMD2(Int(dimension.y), Int(dimension.z))
        case .oblique:
            let maxDim = max(Int(dimension.x), max(Int(dimension.y), Int(dimension.z)))
            outputSize = SIMD2(maxDim, maxDim)
        }
        outputSize = SIMD2(max(outputSize.x, 1), max(outputSize.y, 1))
    }

    private func apply(factory: VolumeTextureFactory,
                       device: MTLDevice,
                       completion: (() -> Void)?) {
        textureFactory = factory
        pendingCompletion = completion
        let token = UUID()
        uploadToken = token

        factory.generate(device: device, commandQueue: uploadQueue) { [weak self] texture in
            guard let self else { return }
            guard self.uploadToken == token else {
                self.pendingCompletion = nil
                return
            }

            guard let texture else {
                Logger.log("Falha ao gerar textura 3D para MPR.",
                           level: .error,
                           category: "MPRDataset")
                self.pendingCompletion?()
                self.pendingCompletion = nil
                return
            }

            self.dimension = factory.dimension
            self.resolution = factory.resolution

            let range = factory.dataset.intensityRange
            self.uniforms.datasetMinValue = range.lowerBound
            self.uniforms.datasetMaxValue = range.upperBound
            self.uniforms.voxelMinValue = range.lowerBound
            self.uniforms.voxelMaxValue = range.upperBound

            self.volumeTexture = texture
            self.updateOutputSize(for: self.currentAxis)
            self.markDirty()
            self.pendingCompletion?()
            self.pendingCompletion = nil
        }
    }
}

// MARK: - Compute backend

private final class MPRComputeRenderer {

    private enum RendererError: Error {
        case libraryNotFound
        case functionNotFound(name: String)
        case commandQueueUnavailable
        case argumentEncoderUnavailable
    }

    private enum ArgumentIndex: Int {
        case volume = 0
        case params = 1
        case output = 2
        case transfer = 3
        case sampler = 4
        case mask = 5
    }

    private let device: MTLDevice
    private let featureFlags: FeatureFlags
    private let usesArgumentBuffers: Bool
    private let supportsNonUniformThreadgroups: Bool
    private let commandQueue: MTLCommandQueue
    private let pipelineState: MTLComputePipelineState
    private let argumentEncoder: MTLArgumentEncoder?
    private var argumentBuffer: MTLBuffer?
    private var parametersBuffer: MTLBuffer?
    private var outputTexture: MTLTexture?
    private let sampler: MTLSamplerState

    init(device: MTLDevice, featureFlags: FeatureFlags) throws {
        self.device = device
        self.featureFlags = featureFlags
        self.usesArgumentBuffers = featureFlags.contains(.argumentBuffers)
        self.supportsNonUniformThreadgroups = featureFlags.contains(.nonUniformThreadgroups)

        guard let library = device.makeDefaultLibrary() else {
            throw RendererError.libraryNotFound
        }
        let functionName = usesArgumentBuffers ? "mpr_compute" : "mpr_compute_legacy"
        guard let function = library.makeFunction(name: functionName) else {
            throw RendererError.functionNotFound(name: functionName)
        }
        pipelineState = try device.makeComputePipelineState(function: function)

        guard let queue = device.makeCommandQueue() else {
            throw RendererError.commandQueueUnavailable
        }
        commandQueue = queue
        commandQueue.label = "MPRComputeQueue"

        sampler = try MPRComputeRenderer.makeSampler(device: device,
                                                     supportsArgumentBuffers: usesArgumentBuffers)

        if usesArgumentBuffers {
            let encoder = function.makeArgumentEncoder(bufferIndex: 0)
            guard let buffer = device.makeBuffer(length: encoder.encodedLength,
                                                 options: [.storageModeShared]) else {
                throw RendererError.argumentEncoderUnavailable
            }
            encoder.setArgumentBuffer(buffer, offset: 0)
            encoder.setSamplerState(sampler, index: ArgumentIndex.sampler.rawValue)
            argumentEncoder = encoder
            argumentBuffer = buffer
        } else {
            argumentEncoder = nil
            argumentBuffer = nil
        }

        Logger.log("MPRComputeRenderer configurado (argumentBuffers=\(usesArgumentBuffers), nonUniformThreadgroups=\(supportsNonUniformThreadgroups))",
                   level: .debug,
                   category: "MPRCompute")
    }

    func render(volume: MTLTexture,
                transfer: MTLTexture?,
                mask: MTLTexture?,
                uniforms: inout MPRPlaneMaterial.Uniforms,
                outputSize: SIMD2<Int>) -> MTLTexture? {

        guard let output = ensureOutputTexture(width: outputSize.x, height: outputSize.y) else {
            return nil
        }

        updateParameterBuffer(with: &uniforms)
        guard let paramsBuffer = parametersBuffer else { return nil }

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            return nil
        }
        commandBuffer.label = usesArgumentBuffers ? "MPRCompute" : "MPRComputeLegacy"

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return nil
        }
        encoder.label = "MPRComputeEncoder"
        encoder.setComputePipelineState(pipelineState)

        if usesArgumentBuffers {
            guard let argumentEncoder = argumentEncoder,
                  let argumentBuffer = argumentBuffer else {
                Logger.log("Argument buffer indisponível para renderização MPR.",
                           level: .error,
                           category: "MPRCompute")
                encoder.endEncoding()
                commandBuffer.commit()
                commandBuffer.waitUntilCompleted()
                return nil
            }

            argumentEncoder.setTexture(volume, index: ArgumentIndex.volume.rawValue)
            argumentEncoder.setBuffer(paramsBuffer, offset: 0, index: ArgumentIndex.params.rawValue)
            argumentEncoder.setTexture(output, index: ArgumentIndex.output.rawValue)

            if let transferTexture = transfer {
                argumentEncoder.setTexture(transferTexture, index: ArgumentIndex.transfer.rawValue)
            } else {
                argumentEncoder.setTexture(nil, index: ArgumentIndex.transfer.rawValue)
            }

            if let maskTexture = mask {
                argumentEncoder.setTexture(maskTexture, index: ArgumentIndex.mask.rawValue)
            } else {
                argumentEncoder.setTexture(nil, index: ArgumentIndex.mask.rawValue)
            }

            encoder.setBuffer(argumentBuffer, offset: 0, index: 0)
            encoder.useResource(argumentBuffer, usage: .read)
        } else {
            encoder.setTexture(volume, index: 0)
            encoder.setBuffer(paramsBuffer, offset: 0, index: 0)
            encoder.setTexture(output, index: 1)
            encoder.setSamplerState(sampler, index: 0)

            if let transferTexture = transfer {
                encoder.setTexture(transferTexture, index: 2)
            } else {
                encoder.setTexture(nil, index: 2)
            }

            if let maskTexture = mask {
                encoder.setTexture(maskTexture, index: 3)
            } else {
                encoder.setTexture(nil, index: 3)
            }
        }

        encoder.useResource(volume, usage: .read)
        encoder.useResource(output, usage: .write)
        encoder.useResource(paramsBuffer, usage: .read)
        if let transferTexture = transfer {
            encoder.useResource(transferTexture, usage: .read)
        }
        if let maskTexture = mask {
            encoder.useResource(maskTexture, usage: .read)
        }

        let threadWidth = pipelineState.threadExecutionWidth
        let maxTotalThreads = pipelineState.maxTotalThreadsPerThreadgroup
        let threadHeight = max(1, maxTotalThreads / threadWidth)
        let threadsPerThreadgroup = MTLSize(width: threadWidth, height: threadHeight, depth: 1)
        let grid = MTLSize(width: output.width, height: output.height, depth: 1)

        if supportsNonUniformThreadgroups {
            encoder.dispatchThreads(grid, threadsPerThreadgroup: threadsPerThreadgroup)
        } else {
            let threadgroups = MTLSize(width: (grid.width + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
                                       height: (grid.height + threadsPerThreadgroup.height - 1) / threadsPerThreadgroup.height,
                                       depth: 1)
            encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        }
        encoder.endEncoding()

        CommandBufferProfiler.captureTimes(for: commandBuffer,
                                           label: commandBuffer.label ?? "MPRCompute",
                                           category: "MPRCompute")

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if commandBuffer.status == .error {
            if let error = commandBuffer.error {
                Logger.log("MPRComputeRenderer command buffer failed: \(error.localizedDescription)",
                           level: .error,
                           category: "MPRCompute")
            } else {
                Logger.log("MPRComputeRenderer command buffer failed sem detalhes adicionais.",
                           level: .error,
                           category: "MPRCompute")
            }
            return nil
        }

#if DEBUG
        if AppConfig.IS_DEBUG_MODE,
           output.storageMode == .shared,
           output.pixelFormat == .bgra8Unorm {
            let sampleX = max(0, min(output.width / 2, output.width - 1))
            let sampleY = max(0, min(output.height / 2, output.height - 1))
            var pixel = SIMD4<UInt8>(repeating: 0)
            withUnsafeMutableBytes(of: &pixel) { bytes in
                guard let baseAddress = bytes.baseAddress else { return }
                let region = MTLRegionMake2D(sampleX, sampleY, 1, 1)
                output.getBytes(baseAddress,
                                bytesPerRow: MemoryLayout<SIMD4<UInt8>>.stride,
                                from: region,
                                mipmapLevel: 0)
            }
            let normalized = SIMD4<Float>(Float(pixel.x) / 255.0,
                                          Float(pixel.y) / 255.0,
                                          Float(pixel.z) / 255.0,
                                          Float(pixel.w) / 255.0)
            Logger.log(String(format: "MPR sample BGRA (%d,%d) -> (%.3f, %.3f, %.3f, %.3f)",
                               sampleX,
                               sampleY,
                               normalized.x,
                               normalized.y,
                               normalized.z,
                               normalized.w),
                       level: .debug,
                       category: "MPRCompute")
        }
#endif

        return output
    }

    private func updateParameterBuffer(with uniforms: inout MPRPlaneMaterial.Uniforms) {
        if parametersBuffer == nil {
            parametersBuffer = device.makeBuffer(length: MPRPlaneMaterial.Uniforms.stride,
                                                 options: [.storageModeShared])
            parametersBuffer?.label = "MPRParametersBuffer"
        }
        guard let buffer = parametersBuffer else { return }
        memcpy(buffer.contents(), &uniforms, MPRPlaneMaterial.Uniforms.stride)
    }

    private func ensureOutputTexture(width: Int, height: Int) -> MTLTexture? {
        if let texture = outputTexture,
           texture.width == width,
           texture.height == height {
            return texture
        }

        // SceneKit + Core Image expect BGRA ordering when we publish the MPR slice.
        // Using BGRA8 ensures the texture can back both on-screen rendering and TIFF export
        // via CIImage (which refuses RGBA8 inputs).
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm,
                                                                  width: width,
                                                                  height: height,
                                                                  mipmapped: false)
        descriptor.usage = [.shaderWrite, .shaderRead]
        // SceneKit precisa consumir essa textura como material; manter storage compartilhado garante visibilidade.
        descriptor.storageMode = .shared

        guard let texture = device.makeTexture(descriptor: descriptor) else {
            Logger.log("Falha ao criar textura de saída MPR \(width)x\(height) em storage shared.",
                       level: .error,
                       category: "MPRCompute")
            return nil
        }

        texture.label = "MPRComputeOutputShared_\(width)x\(height)"
        outputTexture = texture
        return texture
    }

    private static func makeSampler(device: MTLDevice,
                                    supportsArgumentBuffers: Bool) throws -> MTLSamplerState {
        let descriptor = MTLSamplerDescriptor()
        descriptor.minFilter = .linear
        descriptor.magFilter = .linear
        descriptor.mipFilter = .notMipmapped
        descriptor.sAddressMode = .clampToZero
        descriptor.tAddressMode = .clampToZero
        descriptor.rAddressMode = .clampToZero
        descriptor.normalizedCoordinates = true
        descriptor.supportArgumentBuffers = supportsArgumentBuffers
        guard let sampler = device.makeSamplerState(descriptor: descriptor) else {
            throw RendererError.argumentEncoderUnavailable
        }
        return sampler
    }
}
