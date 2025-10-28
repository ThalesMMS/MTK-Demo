//
//  ArgumentEncoderManager.swift
//  VolumeRendering-iOS
//
//  Maintains the legacy argument buffer wiring for Metal volume shaders while the app migrates to MTK surfaces.
//  Thales Matheus Mendonça Santos — October 2025
//

import Foundation
import Metal
import simd

class ArgumentEncoderManager {
    private let device: MTLDevice
    private let mtlFunction: MTLFunction

    // The argument encoder
    private var argumentEncoder: MTLArgumentEncoder!
    public var argumentBuffer: MTLBuffer!

    private var buffers = [Int: MTLBuffer]()

    // A dictionary to track if a value is updated
    private var needsUpdate = [Int: Bool]()

    var currentPxByteSize: Int = 0
    var currentOutputWidth: Int = 0
    var currentOutputHeight: Int = 0
    var outputTexture: MTLTexture?
    private var readbackTexture: MTLTexture?
    private var legacyOutputBuffer: MTLBuffer?
    private var compatibilityPxByteSize: Int = 0
    private let outputPixelFormat: MTLPixelFormat = .bgra8Unorm
    
    var sampler: MTLSamplerState?
    var currentSamplerFilter: MTLSamplerMinMagFilter = .nearest
    
    enum ArgumentIndex: Int, CaseIterable, CustomStringConvertible {
        var description: String{
            switch self{
            case .mainTexture: return "mainTexture"
            case .renderParams: return "renderParams"
            case .outputTexture: return "output texture"
            case .toneBufferCh1: return "tone buffer 1"
            case .toneBufferCh2: return "tone buffer 2"
            case .toneBufferCh3: return "tone buffer 3"
            case .toneBufferCh4: return "tone buffer 4"
            case .optionValue: return "option value"
            case .quaternion: return "quaternion"
            case .sampler: return "sampler"
            case .targetViewSize: return "targetViewSize"
            case .pointSetCountBuffer: return "pointSet count"
            case .pointSetSelectedBuffer: return "pointSet selector"
            case .pointCoordsBuffer: return "pointSet coords"
            case .legacyOutputBuffer: return "output pixel buffer (compat)"
            case .transferTextureCh1: return "transfer function ch1"
            case .transferTextureCh2: return "transfer function ch2"
            case .transferTextureCh3: return "transfer function ch3"
            case .transferTextureCh4: return "transfer function ch4"
        }
        }

        case mainTexture = 0
        case renderParams = 1
        case outputTexture = 2
        case toneBufferCh1 = 3
        case toneBufferCh2 = 4
        case toneBufferCh3 = 5
        case toneBufferCh4 = 6
        case optionValue = 7
        case quaternion = 8
        case targetViewSize = 9
        case sampler = 10
        case pointSetCountBuffer = 11
        case pointSetSelectedBuffer = 12
        case pointCoordsBuffer = 13
        case legacyOutputBuffer = 14
        case transferTextureCh1 = 15
        case transferTextureCh2 = 16
        case transferTextureCh3 = 17
        case transferTextureCh4 = 18

        static func validateShaderLayout(file: StaticString = #file, line: UInt = #line) {
            assert(allCases.count == 19, "RenderingArguments defines 19 resources in volume_compute.metal", file: file, line: line)
            assert(ArgumentIndex.mainTexture.rawValue == 0, "volume_compute.metal expects volumeTexture at index 0", file: file, line: line)
            assert(ArgumentIndex.renderParams.rawValue == 1, "volume_compute.metal expects RenderingParameters at index 1", file: file, line: line)
            assert(ArgumentIndex.outputTexture.rawValue == 2, "volume_compute.metal expects outputTexture at index 2", file: file, line: line)
            assert(ArgumentIndex.toneBufferCh1.rawValue == 3, "volume_compute.metal expects toneBufferCh1 at index 3", file: file, line: line)
            assert(ArgumentIndex.toneBufferCh2.rawValue == 4, "volume_compute.metal expects toneBufferCh2 at index 4", file: file, line: line)
            assert(ArgumentIndex.toneBufferCh3.rawValue == 5, "volume_compute.metal expects toneBufferCh3 at index 5", file: file, line: line)
            assert(ArgumentIndex.toneBufferCh4.rawValue == 6, "volume_compute.metal expects toneBufferCh4 at index 6", file: file, line: line)
            assert(ArgumentIndex.optionValue.rawValue == 7, "volume_compute.metal expects optionValue at index 7", file: file, line: line)
            assert(ArgumentIndex.quaternion.rawValue == 8, "volume_compute.metal expects quaternion at index 8", file: file, line: line)
            assert(ArgumentIndex.targetViewSize.rawValue == 9, "volume_compute.metal expects targetViewSize at index 9", file: file, line: line)
            assert(ArgumentIndex.sampler.rawValue == 10, "volume_compute.metal expects sampler at index 10", file: file, line: line)
            assert(ArgumentIndex.pointSetCountBuffer.rawValue == 11, "volume_compute.metal expects pointSetCount at index 11", file: file, line: line)
            assert(ArgumentIndex.pointSetSelectedBuffer.rawValue == 12, "volume_compute.metal expects pointSelectedIndex at index 12", file: file, line: line)
            assert(ArgumentIndex.pointCoordsBuffer.rawValue == 13, "volume_compute.metal expects pointSet coordinates at index 13", file: file, line: line)
            assert(ArgumentIndex.legacyOutputBuffer.rawValue == 14, "volume_compute.metal expects legacyOutputBuffer at index 14", file: file, line: line)
            assert(ArgumentIndex.transferTextureCh1.rawValue == 15, "volume_compute.metal expects transferTextureCh1 at index 15", file: file, line: line)
            assert(ArgumentIndex.transferTextureCh2.rawValue == 16, "volume_compute.metal expects transferTextureCh2 at index 16", file: file, line: line)
            assert(ArgumentIndex.transferTextureCh3.rawValue == 17, "volume_compute.metal expects transferTextureCh3 at index 17", file: file, line: line)
            assert(ArgumentIndex.transferTextureCh4.rawValue == 18, "volume_compute.metal expects transferTextureCh4 at index 18", file: file, line: line)
        }
    }

    init(device: MTLDevice, mtlFunction: MTLFunction) {
        self.device = device
        self.mtlFunction = mtlFunction

        ArgumentIndex.validateShaderLayout()

        // Create the argument encoder when the manager is initialized
        let encoder = mtlFunction.makeArgumentEncoder(bufferIndex: 0)
        self.argumentEncoder = encoder
        self.argumentEncoder.label = "Argument Encoder"
        
        let argumentBufferLength = argumentEncoder.encodedLength
        
        // create argument buffer
        guard let argumentBuffer = device.makeBuffer(length: argumentBufferLength,
                                                     options: [.cpuCacheModeWriteCombined, .storageModeShared]) else {
            Logger.log("Falha ao criar argument buffer (len=\(argumentBufferLength)).",
                       level: .error,
                       category: "ArgumentEncoder")
            return
        }
        
        self.argumentBuffer = argumentBuffer
        self.argumentBuffer.label = MTL_label.argumentBuffer
        
        self.argumentEncoder.setArgumentBuffer(argumentBuffer, offset: 0)
        
        for argumentIndex in ArgumentIndex.allCases {
            needsUpdate[argumentIndex.rawValue] = true
        }
    }

    func encodeTexture(texture: MTLTexture, argumentIndex: ArgumentIndex){
        let index = argumentIndex.rawValue
        
        if needsUpdate[index] == true{
            argumentEncoder.setTexture(texture, index: index)
            
            needsUpdate[index] = false
            
            if(AppConfig.IS_DEBUG_MODE == true){
                print("arg texture index:\(index) (\(argumentIndex.description)), \(String(describing: type(of: texture))), set")
            }

        }else{
            if(AppConfig.IS_DEBUG_MODE == true){
                print("arg texture index:\(index) (\(argumentIndex.description)), \(String(describing: type(of: texture))), reuse")
            }
        }
    }
    
    func encodeSampler(filter: MTLSamplerMinMagFilter){
        if(self.sampler == nil){
            if(AppConfig.IS_DEBUG_MODE == true){
                print("arg sampler index:\(ArgumentIndex.sampler.rawValue), \(String(describing: type(of: sampler))), created")
            }
            self.sampler = makeSampler(filter: filter)
            self.currentSamplerFilter = filter
            
            if let sampler {
                argumentEncoder.setSamplerState(sampler, index: ArgumentIndex.sampler.rawValue)
            } else {
                Logger.log("Falha ao criar sampler para filtro \(filter).",
                           level: .error,
                           category: "ArgumentEncoder")
            }
            
        }else{
            if(self.currentSamplerFilter != filter){
                // when sampler description has changed
                self.sampler = makeSampler(filter: filter)
                self.currentSamplerFilter = filter
                if(AppConfig.IS_DEBUG_MODE == true){
                    print("arg sampler index:\(ArgumentIndex.sampler.rawValue) (\(ArgumentIndex.sampler.description)), \(String(describing: type(of: sampler))), recreated because filter was changed")
                }
                if let sampler {
                    argumentEncoder.setSamplerState(sampler, index: ArgumentIndex.sampler.rawValue)
                } else {
                    Logger.log("Falha ao recriar sampler para filtro \(filter).",
                               level: .error,
                               category: "ArgumentEncoder")
                }
                
            }else{
                if(AppConfig.IS_DEBUG_MODE == true){
                    print("arg sampler index:\(ArgumentIndex.sampler.rawValue) (\(ArgumentIndex.sampler.description)), \(String(describing: type(of: sampler))), reuse")
                }
            }
        }
    }
    
    // Encodes a struct to the specified argument index
    func encode<T>(_ value: inout T, argumentIndex: ArgumentIndex, capacity: Int = 1) {
        let index = argumentIndex.rawValue
        
        let size = MemoryLayout<T>.stride * capacity
        
        if buffers[index] == nil {
            if(AppConfig.IS_DEBUG_MODE == true){
                print("arg buffer index:\(index) (\(argumentIndex.description)), \(String(describing: T.self)), LayoutSize:\(MemoryLayout<T>.stride), size:\(size) created -> \(value)")
            }
            buffers[index] = device.makeBuffer(length: size, options: [.cpuCacheModeWriteCombined, .storageModeShared])
            buffers[index]?.label = argumentIndex.description
            
            let pointer = buffers[index]!.contents().bindMemory(to: T.self, capacity: capacity)
            pointer.pointee = value
            
            argumentEncoder.setBuffer(buffers[index], offset: 0, index: index)
            needsUpdate[index] = false
            
        }else{
            if(AppConfig.IS_DEBUG_MODE == true){
                print("arg buffer index:\(index) (\(argumentIndex.description)), \(String(describing: T.self)) reuse")
            }
            
        }
        var shouldUpdate = needsUpdate[index] ?? true
        if !shouldUpdate {
            let existingPointer = buffers[index]!.contents()
            withUnsafeBytes(of: value) { rawValueBytes in
                if let baseAddress = rawValueBytes.baseAddress {
                    shouldUpdate = memcmp(existingPointer, baseAddress, size) != 0
                }
            }
        }

        if shouldUpdate {
            let pointer = buffers[index]!.contents().bindMemory(to: T.self, capacity: capacity)
            pointer.pointee = value

            argumentEncoder.setBuffer(buffers[index], offset: 0, index: index)

            if(AppConfig.IS_DEBUG_MODE == true){
                print("arg buffer index:\(index) (\(argumentIndex.description)), \(String(describing: T.self)) update because value change was detected -> \(value)")
            }

            needsUpdate[index] = false
        }
    }
    
    // Encodes a struct to the specified argument index
    func encodeArray(_ value: [float3], argumentIndex: ArgumentIndex, capacity: Int = 1) {
        let index = argumentIndex.rawValue
        // If a buffer for this index doesn't exist, create one
        if (buffers[index] == nil) {
            let size = MemoryLayout<float3>.stride * capacity
            if(AppConfig.IS_DEBUG_MODE == true){
                print("arg buffer index:\(index), \(String(describing: float3.self)), LayoutSize:\(MemoryLayout<float3>.stride), size:\(size) created")
            }
            buffers[index] = device.makeBuffer(bytes: value, length: size, options: [.cpuCacheModeWriteCombined, .storageModeShared])
            buffers[index]?.label = argumentIndex.description
            argumentEncoder.setBuffer(buffers[index], offset: 0, index: index)
            needsUpdate[index] = false
            
        }else{
            if(AppConfig.IS_DEBUG_MODE == true){
                print("arg buffer index:\(index) (\(argumentIndex.description)), \(String(describing: float3.self)) reuse")
            }
            
        }

        if needsUpdate[index] == true {
            let size = MemoryLayout<float3>.stride * capacity
            if(AppConfig.IS_DEBUG_MODE == true){
                print("arg buffer index:\(index) (\(argumentIndex.description)), \(String(describing: float3.self)), LayoutSize:\(MemoryLayout<float3>.stride), size:\(size) created")
            }
            buffers[index] = device.makeBuffer(bytes: value, length: size, options: [.cpuCacheModeWriteCombined, .storageModeShared])
            buffers[index]?.label = argumentIndex.description
            argumentEncoder.setBuffer(buffers[index], offset: 0, index: index)
            needsUpdate[index] = true
        }
    }
    
    func encode(_ buffer: MTLBuffer?, argumentIndex: ArgumentIndex) {
        let index = argumentIndex.rawValue
        
        if(needsUpdate[index] == true){
            buffers[index] = buffer
            argumentEncoder.setBuffer(buffer, offset: 0, index: index)
            needsUpdate[index] = false
            if(AppConfig.IS_DEBUG_MODE == true){
                print("arg buffer index:\(index) (\(argumentIndex.description)), \(String(describing: type(of: buffer))), set")
            }
            
        }else{
            if(AppConfig.IS_DEBUG_MODE == true){
                print("arg buffer index:\(index) (\(argumentIndex.description)), \(String(describing: type(of: buffer))), reuse")
            }
        }
    }
    
    func encodeOutputTexture(width: Int, height: Int) {
        let index = ArgumentIndex.outputTexture.rawValue

        currentOutputWidth = width
        currentOutputHeight = height
        currentPxByteSize = width * height * 4

        let needsRecreation: Bool

        if let texture = outputTexture {
            needsRecreation = texture.width != width || texture.height != height
        } else {
            needsRecreation = true
        }

        if needsRecreation {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: outputPixelFormat,
                width: width,
                height: height,
                mipmapped: false
            )
            descriptor.usage = [.shaderWrite, .shaderRead]
            // SceneKit samples this texture after the compute pass, so keep it GPU/CPU shareable.
            descriptor.storageMode = .shared

            outputTexture = device.makeTexture(descriptor: descriptor)
            outputTexture?.label = MTL_label.outputTexture
            needsUpdate[index] = true
        }

        if needsUpdate[index] == true, let outputTexture {
            argumentEncoder.setTexture(outputTexture, index: index)
            needsUpdate[index] = false

            if AppConfig.IS_DEBUG_MODE {
                print("arg texture index:\(index) (\(ArgumentIndex.outputTexture.description)), output texture set @ \(width)x\(height)")
            }
        }

        encodeCompatibilityBuffer(width: width, height: height)
    }

    private func encodeCompatibilityBuffer(width: Int, height: Int) {
        let index = ArgumentIndex.legacyOutputBuffer.rawValue

        let requiredBytes = width * height * 3
        if legacyOutputBuffer == nil || compatibilityPxByteSize != requiredBytes {
            legacyOutputBuffer = device.makeBuffer(length: MemoryLayout<UInt8>.stride * requiredBytes)
            legacyOutputBuffer?.label = MTL_label.outputPixelBuffer
            needsUpdate[index] = true
            compatibilityPxByteSize = requiredBytes
        }

        if needsUpdate[index] == true {
            argumentEncoder.setBuffer(legacyOutputBuffer, offset: 0, index: index)
            buffers[index] = legacyOutputBuffer
            needsUpdate[index] = false

            if AppConfig.IS_DEBUG_MODE {
                print("arg buffer index:\(index) (\(ArgumentIndex.legacyOutputBuffer.description)), compatibility buffer set @ \(requiredBytes) bytes")
            }
        }
    }

    func makeReadbackTexture(drawingViewSize: Int) -> MTLTexture? {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: outputPixelFormat,
            width: drawingViewSize,
            height: drawingViewSize,
            mipmapped: false
        )
        descriptor.storageMode = .shared

        if let texture = readbackTexture,
           texture.width == drawingViewSize,
           texture.height == drawingViewSize {
            return texture
        }

        readbackTexture = device.makeTexture(descriptor: descriptor)
        readbackTexture?.label = "tex.output.readback"
        return readbackTexture
    }

    // Provides a way to mark a buffer as updated
    func markAsNeedsUpdate(argumentIndex: ArgumentIndex) {
        needsUpdate[argumentIndex.rawValue] = true
        if(AppConfig.IS_DEBUG_MODE == true){
            print("arg buffer index:\(argumentIndex.rawValue) markes as Needs Updata")
        }
    }

    // Provides a way to get the buffer associated with an index (if needed)
    func getBuffer(argumentIndex: ArgumentIndex) -> MTLBuffer? {
        return buffers[argumentIndex.rawValue]
    }

#if DEBUG
    func debugNeedsUpdateState(for argumentIndex: ArgumentIndex) -> Bool? {
        needsUpdate[argumentIndex.rawValue]
    }

    func debugBoundBuffer(for argumentIndex: ArgumentIndex) -> MTLBuffer? {
        buffers[argumentIndex.rawValue]
    }
#endif

    private func makeSampler(filter: MTLSamplerMinMagFilter) -> MTLSamplerState? {
        let descriptor = MTLSamplerDescriptor()
        descriptor.minFilter = filter
        descriptor.magFilter = filter
        descriptor.mipFilter = .notMipmapped
        descriptor.sAddressMode = .clampToZero
        descriptor.tAddressMode = .clampToZero
        descriptor.rAddressMode = .clampToZero
        descriptor.normalizedCoordinates = true
        descriptor.supportArgumentBuffers = true
        descriptor.lodMinClamp = 0
        descriptor.lodMaxClamp = Float.greatestFiniteMagnitude
        descriptor.label = filter == .linear ? "sampler.linear" : "sampler.nearest"
        return device.makeSamplerState(descriptor: descriptor)
    }

    func debugStateSummary() -> String {
        var components: [String] = []
        components.append("output=\(currentOutputWidth)x\(currentOutputHeight)")
        components.append("pixelBytes=\(currentPxByteSize)")
        if let outputTexture {
            components.append("outputTexture=\(outputTexture.width)x\(outputTexture.height)")
        } else {
            components.append("outputTexture=nil")
        }

        if let sampler {
            let samplerLabel = sampler.label ?? (currentSamplerFilter == .linear ? "sampler.linear" : "sampler.nearest")
            components.append("sampler=\(samplerLabel)")
        } else {
            components.append("sampler=nil")
        }

        let dirtyArguments = needsUpdate.compactMap { entry -> String? in
            guard entry.value else { return nil }
            if let argument = ArgumentIndex(rawValue: entry.key) {
                return argument.description
            }
            return "#\(entry.key)"
        }.sorted()
        let dirtySummary = dirtyArguments.isEmpty ? "none" : dirtyArguments.joined(separator: ",")
        components.append("dirty=\(dirtySummary)")

        return components.joined(separator: " | ")
    }
}
