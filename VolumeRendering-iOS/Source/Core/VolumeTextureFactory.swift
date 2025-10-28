import Metal
import simd

final class VolumeTextureFactory {
    private(set) var dataset: VolumeDataset
    private let featureFlags: FeatureFlags
    private var heap: MTLHeap?

    init(dataset: VolumeDataset, featureFlags: FeatureFlags) {
        self.dataset = dataset
        self.featureFlags = featureFlags
    }

    convenience init(part: VolumeCubeMaterial.BodyPart, featureFlags: FeatureFlags) {
        self.init(dataset: VolumeTextureFactory.dataset(for: part), featureFlags: featureFlags)
    }

    var resolution: float3 { dataset.spacing }
    var dimension: int3 { dataset.dimensions }
    var scale: float3 { dataset.scale }
    var orientation: simd_float3x3 { dataset.orientation }
    var transform: simd_float4x4 {
        let scale = dataset.scale
        let orientation = dataset.orientation
        let basisX = orientation.columns.0 * scale.x
        let basisY = orientation.columns.1 * scale.y
        let basisZ = orientation.columns.2 * scale.z
        let translation = SIMD3<Float>(repeating: 0)
        return simd_float4x4(columns: (
            SIMD4<Float>(basisX, 0),
            SIMD4<Float>(basisY, 0),
            SIMD4<Float>(basisZ, 0),
            SIMD4<Float>(translation, 1)
        ))
    }

    func update(dataset: VolumeDataset) {
        self.dataset = dataset
    }

    func generate(device: MTLDevice,
                  commandQueue: MTLCommandQueue?,
                  completion: @escaping (MTLTexture?) -> Void) {
        let descriptor = baseDescriptor()
        if descriptor.storageMode == .private && !featureFlags.contains(.heapAllocations) {
            descriptor.storageMode = .shared
        }

        let textureLabel = "VolumeTexture_\(descriptor.width)x\(descriptor.height)x\(descriptor.depth)"
        let heapLabel = "VolumeTextureHeap_\(descriptor.width)x\(descriptor.height)x\(descriptor.depth)"

        guard let texture = makeTexture(device: device, descriptor: descriptor) else {
            Logger.log("Falha ao criar textura 3D \(descriptor.width)x\(descriptor.height)x\(descriptor.depth)",
                       level: .error,
                       category: "VolumeTextureFactory")
            DispatchQueue.main.async {
                completion(nil)
            }
            return
        }

        texture.label = textureLabel
        heap?.label = heapLabel

        let bytesPerVoxel = dataset.pixelFormat.bytesPerVoxel
        let bytesPerRow = bytesPerVoxel * descriptor.width
        let bytesPerImage = bytesPerRow * descriptor.height
        let totalBytes = dataset.data.count

        Logger.log("Upload de textura \(texture.label ?? "volume") format=\(descriptor.pixelFormat) dim=\(descriptor.width)x\(descriptor.height)x\(descriptor.depth) bytesPerImage=\(bytesPerImage)",
                   level: .debug,
                   category: "VolumeTextureFactory")

        if descriptor.storageMode == .private {
            guard totalBytes > 0 else {
                DispatchQueue.main.async { completion(texture) }
                return
            }

            guard let queue = commandQueue ?? device.makeCommandQueue() else {
                Logger.log("Falha ao criar command queue para upload de textura.",
                           level: .error,
                           category: "VolumeTextureFactory")
                DispatchQueue.main.async { completion(nil) }
                return
            }

            guard let stagingBuffer = device.makeBuffer(length: totalBytes,
                                                        options: [.storageModeShared]) else {
                Logger.log("Falha ao criar staging buffer de upload (\(totalBytes) bytes).",
                           level: .error,
                           category: "VolumeTextureFactory")
                DispatchQueue.main.async { completion(nil) }
                return
            }

            dataset.data.withUnsafeBytes { buffer in
                if let baseAddress = buffer.baseAddress {
                    memcpy(stagingBuffer.contents(), baseAddress, totalBytes)
                }
            }

            guard let commandBuffer = queue.makeCommandBuffer(),
                  let blitEncoder = commandBuffer.makeBlitCommandEncoder() else {
                Logger.log("Falha ao preparar command buffer para upload.",
                           level: .error,
                           category: "VolumeTextureFactory")
                DispatchQueue.main.async { completion(nil) }
                return
            }

            let copySize = MTLSize(width: descriptor.width,
                                   height: descriptor.height,
                                   depth: descriptor.depth)

            blitEncoder.copy(from: stagingBuffer,
                             sourceOffset: 0,
                             sourceBytesPerRow: bytesPerRow,
                             sourceBytesPerImage: bytesPerImage,
                             sourceSize: copySize,
                             to: texture,
                             destinationSlice: 0,
                             destinationLevel: 0,
                             destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
            blitEncoder.endEncoding()

            commandBuffer.addCompletedHandler { _ in
                let retainedBuffer = stagingBuffer // ensure lifetime until completion
                _ = retainedBuffer
                DispatchQueue.main.async {
                    completion(texture)
                }
            }

            commandBuffer.commit()
        } else {
            dataset.data.withUnsafeBytes { buffer in
                guard let baseAddress = buffer.baseAddress else { return }
                texture.replace(region: MTLRegionMake3D(0, 0, 0,
                                                        descriptor.width,
                                                        descriptor.height,
                                                        descriptor.depth),
                                mipmapLevel: 0,
                                slice: 0,
                                withBytes: baseAddress,
                                bytesPerRow: bytesPerRow,
                                bytesPerImage: bytesPerImage)
            }

            DispatchQueue.main.async {
                completion(texture)
            }
        }
    }
}

private extension VolumeTextureFactory {
    func baseDescriptor() -> MTLTextureDescriptor {
        let descriptor = MTLTextureDescriptor()
        descriptor.textureType = .type3D
        descriptor.pixelFormat = dataset.pixelFormat.metalPixelFormat
        descriptor.usage = .shaderRead
        descriptor.width = Int(dataset.dimensions.x)
        descriptor.height = Int(dataset.dimensions.y)
        descriptor.depth = Int(dataset.dimensions.z)
        descriptor.storageMode = featureFlags.contains(.heapAllocations) ? .private : .shared
        descriptor.cpuCacheMode = .defaultCache
        return descriptor
    }

    func makeTexture(device: MTLDevice, descriptor: MTLTextureDescriptor) -> MTLTexture? {
        // Heaps require private storage; when using shared textures (e.g. to allow CPU writes)
        // fall back to allocating directly on the device.
        guard descriptor.storageMode == .private else {
            return device.makeTexture(descriptor: descriptor)
        }

        guard featureFlags.contains(.heapAllocations) else {
            let fallbackDescriptor = (descriptor.copy() as? MTLTextureDescriptor) ?? descriptor
            fallbackDescriptor.storageMode = .shared
            return device.makeTexture(descriptor: fallbackDescriptor)
        }

        let sizeAndAlign = device.heapTextureSizeAndAlign(descriptor: descriptor)
        let alignment = max(1, sizeAndAlign.align)
        let requiredSize = ((sizeAndAlign.size + alignment - 1) / alignment) * alignment

        if let heap = heap {
            let available = heap.maxAvailableSize(alignment: alignment)
            if available >= requiredSize && heap.storageMode == descriptor.storageMode {
                return heap.makeTexture(descriptor: descriptor)
            }
        }

        let heapDescriptor = MTLHeapDescriptor()
        heapDescriptor.storageMode = descriptor.storageMode
        heapDescriptor.cpuCacheMode = descriptor.cpuCacheMode
        heapDescriptor.resourceOptions = [.storageModePrivate]
        heapDescriptor.size = max(requiredSize, 1 << 20)

        guard let newHeap = device.makeHeap(descriptor: heapDescriptor) else {
            return nil
        }

        heap = newHeap
        newHeap.label = "VolumeTextureHeap"
        return newHeap.makeTexture(descriptor: descriptor)
    }

    static func dataset(for part: VolumeCubeMaterial.BodyPart) -> VolumeDataset {
        switch part {
        case .none, .dicom:
            return placeholderDataset()
        }
    }

    static func placeholderDataset() -> VolumeDataset {
        let data = Data(count: VolumePixelFormat.int16Signed.bytesPerVoxel)
        return VolumeDataset(data: data,
                             dimensions: int3(1, 1, 1),
                             spacing: float3(1, 1, 1),
                             pixelFormat: .int16Signed,
                             intensityRange: (-1024)...3071)
    }
}
