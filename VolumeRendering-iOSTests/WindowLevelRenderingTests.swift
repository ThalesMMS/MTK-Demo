@testable import VolumeRendering_iOS
import Metal
import SceneKit
import simd
import XCTest

@MainActor
final class WindowLevelRenderingTests: XCTestCase {
    private let outputSize = CGSize(width: 64, height: 64)

    func testVolumeComputeRespectsWindowLevelWithTransferFunction() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal indisponível neste dispositivo de teste.")
        }

        let featureFlags = FeatureFlags.evaluate(for: device)
        guard featureFlags.contains(.argumentBuffers) else {
            throw XCTSkip("Argument buffers não suportados – ignorando teste de compute.")
        }

        // Dataset preenchido com HU=0 para permitir contraste controlado via window/level.
        let dataset = makeFilledDataset(size: 16, value: 0, intensityRange: -1024...3071)

        let material = VolumeCubeMaterial(device: device, featureFlags: featureFlags)
        let uploadExpectation = expectation(description: "Upload dataset compute")
        material.setDataset(device: device, dataset: dataset) {
            uploadExpectation.fulfill()
        }
        wait(for: [uploadExpectation], timeout: 10.0)

        material.setMethod(method: .dvr)
        material.setLighting(on: false)
        material.setStep(step: Float(outputSize.width))

        let transferTexture = try makeLinearTransferTexture(device: device)
        material.setTransferFunctionTexture(transferTexture)

        guard let volumeTexture = material.currentVolumeTexture() else {
            XCTFail("Textura de volume não disponível após upload.")
            return
        }

        guard let tfTexture = material.currentTransferFunctionTexture() else {
            XCTFail("Textura da transfer function indisponível.")
            return
        }

        let (scene, volumeNode) = makeScene(with: material)
        let cameraNode = makeCameraNode(for: volumeNode)
        guard let cameraParameters = makeCameraParameters(cameraNode: cameraNode, volumeNode: volumeNode) else {
            XCTFail("Falha ao construir parâmetros de câmera.")
            _ = scene
            return
        }
        _ = scene // mantêm referências vivas

        let computeRenderer = try VolumeComputeRenderer(device: device, featureFlags: featureFlags)
        computeRenderer.updateTransferTextures(Array(repeating: tfTexture, count: 4))
        computeRenderer.updateChannelIntensities(SIMD4<Float>(1, 0, 0, 0))

        // Janela abrangendo o HU=0 (espera-se pixel brilhante).
        material.setHuWindow(minHU: -150, maxHU: 150)
        let brightUniforms = normalizedUniforms(from: material, fallbackQuality: Int(outputSize.width))
        guard let brightTexture = computeRenderer.render(volume: volumeTexture,
                                                         uniforms: brightUniforms,
                                                         camera: cameraParameters,
                                                         outputSize: outputSize),
              let brightData = computeRenderer.readback(texture: brightTexture) else {
            XCTFail("Falha ao renderizar cenário iluminado.")
            return
        }
        let brightPixel = sampleCenterPixel(fromBGRA: brightData,
                                            width: Int(outputSize.width),
                                            height: Int(outputSize.height))

        // Janela deslocada para HU altos (pixel deve apagar).
        material.setHuWindow(minHU: 400, maxHU: 600)
        let darkUniforms = normalizedUniforms(from: material, fallbackQuality: Int(outputSize.width))
        guard let darkTexture = computeRenderer.render(volume: volumeTexture,
                                                       uniforms: darkUniforms,
                                                       camera: cameraParameters,
                                                       outputSize: outputSize),
              let darkData = computeRenderer.readback(texture: darkTexture) else {
            XCTFail("Falha ao renderizar cenário obscurecido.")
            return
        }
        let darkPixel = sampleCenterPixel(fromBGRA: darkData,
                                          width: Int(outputSize.width),
                                          height: Int(outputSize.height))

        XCTAssertGreaterThan(rgbAverage(brightPixel), 0.05, "Pixel renderizado deveria manter luminância sob janela cobrindo o HU.")
        XCTAssertLessThan(rgbAverage(darkPixel), 0.01, "Pixel não deveria acumular intensidade quando HU está fora da janela.")
        XCTAssertGreaterThan(brightPixel.w, darkPixel.w + 0.1, "Alpha do pixel iluminado deve exceder substancialmente o escuro.")
    }

    func testMPRComputeRespectsWindowLevelWithTransferFunction() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal indisponível neste dispositivo de teste.")
        }

        let featureFlags = FeatureFlags.evaluate(for: device)
        guard featureFlags.contains(.argumentBuffers) else {
            throw XCTSkip("Argument buffers não suportados – ignorando teste de MPR.")
        }

        let dataset = makeFilledDataset(size: 16, value: 0, intensityRange: -1024...3071)

        let mprMaterial = MPRPlaneMaterial(device: device, featureFlags: featureFlags)
        let uploadExpectation = expectation(description: "Upload dataset MPR")
        mprMaterial.setDataset(device: device, dataset: dataset) {
            uploadExpectation.fulfill()
        }
        wait(for: [uploadExpectation], timeout: 10.0)

        mprMaterial.setAxial(slice: Int(dataset.dimensions.z / 2))
        let mprTransfer = try makeLinearTransferTexture(device: device)
        mprMaterial.setTransferFunction(mprTransfer)
        mprMaterial.setUseTF(true)

        // Render com janela abrangente.
        mprMaterial.setHU(min: -150, max: 150)
        mprMaterial.renderIfNeeded()
        guard let brightTexture = mprMaterial.diffuse.contents as? MTLTexture else {
            XCTFail("Textura MPR (bright) não disponível.")
            return
        }
        let brightData = readTextureBGRA(brightTexture)
        let brightPixel = sampleCenterPixel(fromBGRA: brightData,
                                            width: brightTexture.width,
                                            height: brightTexture.height)

        // Render com janela deslocada.
        mprMaterial.setHU(min: 400, max: 600)
        mprMaterial.renderIfNeeded()
        guard let darkTexture = mprMaterial.diffuse.contents as? MTLTexture else {
            XCTFail("Textura MPR (dark) não disponível.")
            return
        }
        let darkData = readTextureBGRA(darkTexture)
        let darkPixel = sampleCenterPixel(fromBGRA: darkData,
                                          width: darkTexture.width,
                                          height: darkTexture.height)

        XCTAssertGreaterThan(rgbAverage(brightPixel), 0.05, "Pixel MPR deveria manter luminância quando HU está dentro da janela.")
        XCTAssertLessThan(rgbAverage(darkPixel), 0.01, "Pixel MPR deveria apagar quando HU sai da janela.")
        XCTAssertGreaterThan(brightPixel.w, darkPixel.w + 0.1, "Alpha MPR brilhante precisa exceder claramente o escuro.")
    }
}

// MARK: - Helpers

private extension WindowLevelRenderingTests {
    enum HelperError: Error {
        case transferTextureCreation
    }
    func makeFilledDataset(size: Int, value: Int16, intensityRange: ClosedRange<Int32>) -> VolumeDataset {
        precondition(size > 0)
        let voxelCount = size * size * size
        var data = Data(count: voxelCount * MemoryLayout<Int16>.size)
        data.withUnsafeMutableBytes { pointer in
            guard let base = pointer.bindMemory(to: Int16.self).baseAddress else { return }
            for index in 0..<voxelCount {
                base[index] = value
            }
        }

        let spacing = float3(repeating: 1.0 / Float(size))
        let dimensions = int3(Int32(size), Int32(size), Int32(size))
        return VolumeDataset(data: data,
                             dimensions: dimensions,
                             spacing: spacing,
                             pixelFormat: .int16Signed,
                             intensityRange: intensityRange)
    }

    func makeLinearTransferTexture(device: MTLDevice, sampleCount: Int = 256) throws -> MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float,
                                                                  width: sampleCount,
                                                                  height: 1,
                                                                  mipmapped: false)
        descriptor.usage = [.shaderRead]
        descriptor.storageMode = .shared

        guard let texture = device.makeTexture(descriptor: descriptor) else {
            XCTFail("Falha ao criar textura linear de transferência.")
            throw HelperError.transferTextureCreation
        }
        texture.label = "LinearTF"

        var samples = [SIMD4<Float>](repeating: .zero, count: sampleCount)
        for index in 0..<sampleCount {
            let t = Float(index) / Float(max(sampleCount - 1, 1))
            samples[index] = SIMD4<Float>(repeating: t)
        }

        samples.withUnsafeBytes { rawBuffer in
            texture.replace(region: MTLRegionMake2D(0, 0, sampleCount, 1),
                            mipmapLevel: 0,
                            withBytes: rawBuffer.baseAddress!,
                            bytesPerRow: MemoryLayout<SIMD4<Float>>.stride * sampleCount)
        }

        return texture
    }

    func makeScene(with material: VolumeCubeMaterial) -> (SCNScene, SCNNode) {
        let scene = SCNScene()
        let box = SCNBox(width: 1, height: 1, length: 1, chamferRadius: 0)
        box.firstMaterial = material

        let node = SCNNode(geometry: box)
        node.simdTransform = material.transform
        scene.rootNode.addChildNode(node)
        return (scene, node)
    }

    func makeCameraNode(for volumeNode: SCNNode) -> SCNNode {
        let camera = SCNCamera()
        camera.usesOrthographicProjection = false
        camera.zNear = 0.01
        camera.zFar = 10.0
        camera.fieldOfView = 40

        let node = SCNNode()
        node.camera = camera
        let radius = max(volumeNode.boundingSphere.radius, 0.5)
        node.position = SCNVector3(0, 0, radius * 3)
        node.look(at: SCNVector3Zero)
        volumeNode.parent?.addChildNode(node)
        return node
    }

    func makeCameraParameters(cameraNode: SCNNode, volumeNode: SCNNode) -> VolumeCameraParameters? {
        guard let camera = cameraNode.camera else { return nil }

        let modelMatrix = volumeNode.simdWorldTransform
        let inverseModelMatrix = simd_inverse(modelMatrix)
        let projectionMatrix = simd_float4x4(camera.projectionTransform)
        let viewMatrix = simd_inverse(cameraNode.simdWorldTransform)
        let viewProjection = projectionMatrix * viewMatrix
        let inverseViewProjection = simd_inverse(viewProjection)

        let cameraWorld = cameraNode.simdWorldPosition
        let cameraLocal4 = inverseModelMatrix * SIMD4<Float>(cameraWorld, 1.0)
        let cameraLocal = cameraLocal4.xyz / max(cameraLocal4.w, 1e-5)

        return VolumeCameraParameters(modelMatrix: modelMatrix,
                                      inverseModelMatrix: inverseModelMatrix,
                                      inverseViewProjectionMatrix: inverseViewProjection,
                                      cameraPositionLocal: cameraLocal)
    }

    func normalizedUniforms(from material: VolumeCubeMaterial, fallbackQuality: Int) -> VolumeCubeMaterial.Uniforms {
        var uniforms = material.snapshotUniforms()
        if uniforms.renderingQuality <= 0 {
            uniforms.renderingQuality = Int32(fallbackQuality)
        }
        return uniforms
    }

    func sampleCenterPixel(fromBGRA data: Data, width: Int, height: Int) -> SIMD4<Float> {
        precondition(data.count == width * height * 4)
        let x = width / 2
        let y = height / 2
        let index = (y * width + x) * 4

        return data.withUnsafeBytes { pointer -> SIMD4<Float> in
            let base = pointer.bindMemory(to: UInt8.self).baseAddress!
            let b = Float(base[index + 0]) / 255.0
            let g = Float(base[index + 1]) / 255.0
            let r = Float(base[index + 2]) / 255.0
            let a = Float(base[index + 3]) / 255.0
            return SIMD4<Float>(r, g, b, a)
        }
    }

    func readTextureBGRA(_ texture: MTLTexture) -> Data {
        let width = texture.width
        let height = texture.height
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        var data = Data(count: bytesPerRow * height)
        data.withUnsafeMutableBytes { pointer in
            guard let baseAddress = pointer.baseAddress else { return }
            texture.getBytes(baseAddress,
                             bytesPerRow: bytesPerRow,
                             from: MTLRegionMake2D(0, 0, width, height),
                             mipmapLevel: 0)
        }
        return data
    }

    func rgbAverage(_ pixel: SIMD4<Float>) -> Float {
        (pixel.x + pixel.y + pixel.z) / 3.0
    }
}
