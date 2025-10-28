import XCTest
import Metal
import SceneKit
import simd
import UIKit
@testable import VolumeRendering_iOS

@MainActor
final class RendererRegressionTests: XCTestCase {
    private let viewSize: Int = 192
    private lazy var viewportSize = CGSize(width: viewSize, height: viewSize)

    private struct RegressionResult {
        let dataset: String
        let transfer: String
        let angleIndex: Int
        let rmse: Double
        let timings: CommandBufferTimings?
    }

    private struct DatasetDefinition {
        let name: String
        let dataset: VolumeDataset
    }

    private struct TransferDefinition {
        let name: String
        let texture: MTLTexture
    }

    func testComputeMatchesFragmentRenderer() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal indisponível neste dispositivo de teste.")
        }

        let featureFlags = FeatureFlags.evaluate(for: device)
        guard featureFlags.contains(.argumentBuffers) else {
            throw XCTSkip("Argument buffers não suportados – ignorando regressão compute.")
        }

        let datasets = try makeDatasets()
        let transfers = try makeTransferFunctions(device: device)
        let quaternions = makeRotationSuite()

        let computeRenderer = try VolumeComputeRenderer(device: device, featureFlags: featureFlags)
        let renderer = SCNRenderer(device: device, options: nil)

        var results: [RegressionResult] = []

        for datasetDefinition in datasets {
            let material = VolumeCubeMaterial(device: device, featureFlags: featureFlags)
            let datasetExpectation = expectation(description: "Dataset upload \(datasetDefinition.name)")
            material.setDataset(device: device, dataset: datasetDefinition.dataset) {
                datasetExpectation.fulfill()
            }
            wait(for: [datasetExpectation], timeout: 60.0)
            material.setMethod(method: .dvr)
            material.setLighting(on: false)
            material.setStep(step: Float(viewSize))

            let (scene, volumeNode) = makeScene(with: material)
            let cameraNode = makeCameraNode(for: volumeNode)
            renderer.scene = scene
            renderer.pointOfView = cameraNode

            for transfer in transfers {
                material.setTransferFunctionTexture(transfer.texture)
                material.setShift(device: device, shift: 0)

                guard let transferTexture = material.currentTransferFunctionTexture() else {
                    XCTFail("Falha ao recuperar textura da transfer function para testes")
                    continue
                }

                for (index, quaternion) in quaternions.enumerated() {
                    apply(quaternion: quaternion, to: volumeNode, baseTransform: material.transform)
                    cameraNode.look(at: SCNVector3Zero)

                    guard let volumeTexture = material.currentVolumeTexture() else {
                        XCTFail("Falha ao recuperar textura de volume para dataset \(datasetDefinition.name)")
                        continue
                    }

                    let uniforms = normalizedUniforms(from: material)
                    guard let cameraParameters = makeCameraParameters(cameraNode: cameraNode, volumeNode: volumeNode) else {
                        XCTFail("Falha ao criar parâmetros de câmera")
                        continue
                    }

                    guard let fragmentData = renderFragment(using: renderer, size: viewportSize) else {
                        XCTFail("Snapshot SceneKit retornou nulo")
                        continue
                    }

                    let transfersForCompute = [transferTexture, transferTexture, transferTexture, transferTexture]
                    computeRenderer.updateTransferTextures(transfersForCompute)
                    computeRenderer.updateChannelIntensities(SIMD4<Float>(1, 0, 0, 0))

                    guard let computeTexture = computeRenderer.render(volume: volumeTexture,
                                                                       uniforms: uniforms,
                                                                       camera: cameraParameters,
                                                                       outputSize: viewportSize) else {
                        XCTFail("Renderer compute retornou textura nula")
                        continue
                    }

                    guard let computeData = computeRenderer.readback(texture: computeTexture) else {
                        XCTFail("Falha ao realizar readback da textura compute")
                        continue
                    }

                    let rmse = computeRMSE(lhs: computeData, rhs: fragmentData)
                    let configuration = RegressionResult(dataset: datasetDefinition.name,
                                                          transfer: transfer.name,
                                                          angleIndex: index,
                                                          rmse: rmse,
                                                          timings: computeRenderer.lastRenderTimings)
                    results.append(configuration)

                    XCTAssertLessThanOrEqual(rmse, 0.01, "RMSE excedeu tolerância para \(datasetDefinition.name)-\(transfer.name)-angle\(index)")
                }
            }
        }

        attachMetrics(for: results)
    }
}

// MARK: - Dataset & Transfer Helpers

private extension RendererRegressionTests {
    private func makeDatasets() throws -> [DatasetDefinition] {
        let sphere = makeSphereDataset(name: "sphere", size: 128, radius: 40, insideHU: 1500, outsideHU: -900)
        let layered = makeLayeredDataset(name: "layered", size: 128)
        let anisotropic = makeAnisotropicDataset(name: "anisotropic")

        var datasets = [sphere, layered, anisotropic]
        let dicom = try loadDicomDataset()
        datasets.append(dicom)
        return datasets
    }

    private func makeSphereDataset(name: String, size: Int, radius: Int, insideHU: Int16, outsideHU: Int16) -> DatasetDefinition {
        var data = Data(count: size * size * size * MemoryLayout<Int16>.size)
        let center = SIMD3<Float>(Float(size) / 2)
        let radiusSq = Float(radius * radius)

        data.withUnsafeMutableBytes { pointer in
            guard let base = pointer.bindMemory(to: Int16.self).baseAddress else { return }
            for z in 0..<size {
                for y in 0..<size {
                    for x in 0..<size {
                        let index = z * size * size + y * size + x
                        let position = SIMD3<Float>(Float(x), Float(y), Float(z))
                        let distanceSq = simd_length_squared(position - center)
                        base[index] = distanceSq <= radiusSq ? insideHU : outsideHU
                    }
                }
            }
        }

        let minValue = Int32(min(insideHU, outsideHU))
        let maxValue = Int32(max(insideHU, outsideHU))
        let dataset = VolumeDataset(data: data,
                                    dimensions: int3(Int32(size), Int32(size), Int32(size)),
                                    spacing: SIMD3<Float>(repeating: 1.0 / Float(size)),
                                    pixelFormat: .int16Signed,
                                    intensityRange: minValue...maxValue)
        return DatasetDefinition(name: name, dataset: dataset)
    }

    private func loadDicomDataset() throws -> DatasetDefinition {
        let importResult = try loadDicomImportResult(at: DicomTestSupport.seriesURL())
        return DatasetDefinition(name: "dicom-series", dataset: importResult.dataset)
    }

    private func makeLayeredDataset(name: String, size: Int) -> DatasetDefinition {
        var data = Data(count: size * size * size * MemoryLayout<Int16>.size)
        var minValue = Int16.max
        var maxValue = Int16.min

        data.withUnsafeMutableBytes { pointer in
            guard let base = pointer.bindMemory(to: Int16.self).baseAddress else { return }
            for z in 0..<size {
                let layerValue = Int16(-900 + (z % 16) * 120)
                for y in 0..<size {
                    for x in 0..<size {
                        let index = z * size * size + y * size + x
                        let checker = ((x / 16) + (y / 16)) % 2 == 0
                        let value = checker ? layerValue : -1024
                        base[index] = value
                        minValue = min(minValue, value)
                        maxValue = max(maxValue, value)
                    }
                }
            }
        }

        let dataset = VolumeDataset(data: data,
                                    dimensions: int3(Int32(size), Int32(size), Int32(size)),
                                    spacing: SIMD3<Float>(repeating: 1.0 / Float(size)),
                                    pixelFormat: .int16Signed,
                                    intensityRange: Int32(minValue)...Int32(maxValue))
        return DatasetDefinition(name: name, dataset: dataset)
    }

    private func makeAnisotropicDataset(name: String) -> DatasetDefinition {
        let width = 96
        let height = 128
        let depth = 48
        let voxelCount = width * height * depth
        var data = Data(count: voxelCount * MemoryLayout<Int16>.size)
        var minValue = Int16.max
        var maxValue = Int16.min

        data.withUnsafeMutableBytes { pointer in
            guard let base = pointer.bindMemory(to: Int16.self).baseAddress else { return }
            for z in 0..<depth {
                let normalizedZ = Float(z) / Float(max(depth - 1, 1))
                for y in 0..<height {
                    let normalizedY = Float(y) / Float(max(height - 1, 1))
                    for x in 0..<width {
                        let normalizedX = Float(x) / Float(max(width - 1, 1))
                        let index = z * width * height + y * width + x
                        let huValue = -950.0
                            + Double(normalizedZ) * 1800.0
                            + Double(normalizedY) * 400.0
                            + Double(normalizedX) * 250.0
                        let intValue = Int16(clamping: Int(lround(huValue)))
                        base[index] = intValue
                        minValue = min(minValue, intValue)
                        maxValue = max(maxValue, intValue)
                    }
                }
            }
        }

        let spacing = SIMD3<Float>(0.0015, 0.0025, 0.0040)
        let dataset = VolumeDataset(data: data,
                                    dimensions: int3(Int32(width), Int32(height), Int32(depth)),
                                    spacing: spacing,
                                    pixelFormat: .int16Signed,
                                    intensityRange: Int32(minValue)...Int32(maxValue))
        return DatasetDefinition(name: name, dataset: dataset)
    }

    private func makeTransferFunctions(device: MTLDevice) throws -> [TransferDefinition] {
        return try [
            TransferDefinition(name: "linear", texture: makeTransferTexture(device: device) { t in
                let color = SIMD3<Float>(repeating: t)
                return SIMD4<Float>(color, t)
            }),
            TransferDefinition(name: "inverse", texture: makeTransferTexture(device: device) { t in
                let color = SIMD3<Float>(repeating: 1.0 - t)
                return SIMD4<Float>(color, t)
            }),
            TransferDefinition(name: "gamma", texture: makeTransferTexture(device: device) { t in
                let gamma = powf(t, 1.8)
                return SIMD4<Float>(SIMD3<Float>(repeating: gamma), gamma)
            })
        ]
    }

    private func makeTransferTexture(device: MTLDevice, ramp: (Float) -> SIMD4<Float>) throws -> MTLTexture {
        let width = 512
        let height = 2
        var pixels = [SIMD4<Float>](repeating: .zero, count: width * height)
        for x in 0..<width {
            let t = Float(x) / Float(width - 1)
            let value = ramp(t)
            for y in 0..<height {
                pixels[x + y * width] = value
            }
        }

        let descriptor = MTLTextureDescriptor()
        descriptor.pixelFormat = .rgba32Float
        descriptor.width = width
        descriptor.height = height
        descriptor.usage = .shaderRead

        guard let texture = device.makeTexture(descriptor: descriptor) else {
            throw NSError(domain: "RendererRegressionTests", code: 1, userInfo: [NSLocalizedDescriptionKey: "Não foi possível criar textura de transfer function"])
        }

        texture.replace(region: MTLRegionMake2D(0, 0, width, height),
                        mipmapLevel: 0,
                        slice: 0,
                        withBytes: pixels,
                        bytesPerRow: MemoryLayout<SIMD4<Float>>.stride * width,
                        bytesPerImage: MemoryLayout<SIMD4<Float>>.stride * width * height)
        return texture
    }
}

// MARK: - Scene / Camera Helpers

private extension RendererRegressionTests {
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

    func apply(quaternion: simd_quatf, to node: SCNNode, baseTransform: simd_float4x4) {
        let rotationMatrix = simd_float4x4(quaternion)
        node.simdTransform = matrix_multiply(rotationMatrix, baseTransform)
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

    func renderFragment(using renderer: SCNRenderer, size: CGSize) -> Data? {
        let image = renderer.snapshot(atTime: 0, with: size, antialiasingMode: .none)
        guard let cgImage = image.cgImage else { return nil }
        return makeBGRAData(from: cgImage)
    }
}

// MARK: - Math & Metrics

private extension RendererRegressionTests {
    func makeRotationSuite() -> [simd_quatf] {
        var quaternions: [simd_quatf] = []
        let angles = stride(from: 0.0, to: Float.pi * 2.0, by: Float.pi / 5.0)
        for yaw in angles {
            for pitch in [0.0, Float.pi / 4.0] {
                let qx = simd_quatf(angle: pitch, axis: SIMD3<Float>(1, 0, 0))
                let qy = simd_quatf(angle: yaw, axis: SIMD3<Float>(0, 1, 0))
                quaternions.append(qy * qx)
            }
        }
        return Array(quaternions.prefix(10))
    }

    func computeRMSE(lhs: Data, rhs: Data) -> Double {
        precondition(lhs.count == rhs.count)
        var total: Double = 0
        let count = lhs.count

        lhs.withUnsafeBytes { lPtr in
            rhs.withUnsafeBytes { rPtr in
                guard let lBase = lPtr.bindMemory(to: UInt8.self).baseAddress,
                      let rBase = rPtr.bindMemory(to: UInt8.self).baseAddress else { return }
                for index in 0..<count {
                    let diff = Double(Int(lBase[index]) - Int(rBase[index]))
                    total += diff * diff
                }
            }
        }

        return sqrt(total / Double(count)) / 255.0
    }

    func makeBGRAData(from image: CGImage) -> Data? {
        let width = image.width
        let height = image.height
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        var data = Data(count: bytesPerRow * height)

        let bitmapInfo = CGBitmapInfo.byteOrder32Little.union(CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue))

        let success = data.withUnsafeMutableBytes { pointer -> Bool in
            guard let context = CGContext(data: pointer.baseAddress,
                                          width: width,
                                          height: height,
                                          bitsPerComponent: 8,
                                          bytesPerRow: bytesPerRow,
                                          space: CGColorSpaceCreateDeviceRGB(),
                                          bitmapInfo: bitmapInfo.rawValue) else {
                return false
            }
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }

        return success ? data : nil
    }

    func normalizedUniforms(from material: VolumeCubeMaterial) -> VolumeCubeMaterial.Uniforms {
        var uniforms = material.snapshotUniforms()
        if uniforms.renderingQuality <= 0 {
            uniforms.renderingQuality = Int32(viewSize)
        }
        return uniforms
    }

    private func attachMetrics(for results: [RegressionResult]) {
        let grouped = Dictionary(grouping: results) { $0.dataset }
        for (dataset, entries) in grouped {
            let rmseValues = entries.map { $0.rmse }
            let averageRMSE = rmseValues.reduce(0, +) / Double(rmseValues.count)
            let maxRMSE = rmseValues.max() ?? 0

            var summary = "Average RMSE: \(String(format: "%.4f", averageRMSE)), Max RMSE: \(String(format: "%.4f", maxRMSE))\n"

            let kernelTimings = entries.compactMap { $0.timings?.kernelTime }
            if !kernelTimings.isEmpty {
                let minKernel = kernelTimings.min() ?? 0
                let avgKernel = kernelTimings.reduce(0, +) / Double(kernelTimings.count)
                let sorted = kernelTimings.sorted()
                let p95Index = Int(Double(sorted.count - 1) * 0.95)
                let p95Kernel = sorted[min(p95Index, sorted.count - 1)]
                summary += "Kernel time (ms) – min: \(String(format: "%.3f", minKernel)), avg: \(String(format: "%.3f", avgKernel)), p95: \(String(format: "%.3f", p95Kernel))"
            }

            let attachment = XCTAttachment(string: summary)
            attachment.name = "\(dataset)-metrics"
            add(attachment)
        }
    }
}
