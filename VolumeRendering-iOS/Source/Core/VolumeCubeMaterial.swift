import SceneKit
import SwiftUI

/// SceneKit material que hosteia o shader Metal de volume (SR/DVR/MIP/MinIP/AIP).
final class VolumeCubeMaterial: SCNMaterial {

    // MARK: - Enums

    enum Preset: String, CaseIterable, Identifiable {
        case ct_arteries
        case ct_entire
        case ct_lung
        case ct_bone
        case ct_cardiac
        case ct_liver_vasculature
        case mr_t2_brain
        case ct_chest_contrast
        case ct_soft_tissue
        case ct_pulmonary_arteries
        case ct_fat
        case mr_angio

        var id: RawValue { rawValue }

        var displayName: String {
            switch self {
            case .ct_arteries: return "CT Arteries"
            case .ct_entire: return "CT Entire"
            case .ct_lung: return "CT Lung"
            case .ct_bone: return "CT Bone"
            case .ct_cardiac: return "CT Cardiac"
            case .ct_liver_vasculature: return "CT Liver Vasculature"
            case .mr_t2_brain: return "MR T2 Brain"
            case .ct_chest_contrast: return "CT Chest Contrast"
            case .ct_soft_tissue: return "CT Soft Tissue"
            case .ct_pulmonary_arteries: return "CT Pulmonary Arteries"
            case .ct_fat: return "CT Fat"
            case .mr_angio: return "MR Angio"
            }
        }
    }

    /// Métodos de renderização. Os IDs precisam casar com o `switch` do shader.
    enum Method: String, CaseIterable, Identifiable {
        var id: RawValue { rawValue }
        var idInt32: Int32 {
            switch self {
            case .surf:  return 0
            case .dvr:   return 1
            case .mip:   return 2
            case .minip: return 3
            case .avg:   return 4
            }
        }
        case surf, dvr, mip, minip, avg
    }

    enum BodyPart: String, CaseIterable, Identifiable {
        case none, dicom

        var id: RawValue { rawValue }

        var displayName: String {
            switch self {
            case .none:
                return "None"
            case .dicom:
                return "DICOM"
            }
        }
    }

    // MARK: - Uniforms (deve casar com struct Uniforms em volumerendering.metal)

    struct Uniforms: sizeable {
        // Flags como Int32 para alinhamento/portabilidade Swift/MSL
        var isLightingOn: Int32 = 1
        var isBackwardOn: Int32 = 0

        var method: Int32 = Method.dvr.idInt32
        var renderingQuality: Int32 = 512

        // HU window normalização [min..max]
        var voxelMinValue: Int32 = -1024
        var voxelMaxValue: Int32 =  3071

        // Faixa HU completa do dataset (mantida separada do window)
        var datasetMinValue: Int32 = -1024
        var datasetMaxValue: Int32 =  3071

        // Gating para projeções (MIP/MinIP/AIP)
        var densityFloor: Float = 0.02
        var densityCeil:  Float = 1.00

        // Gating por HU nativo (quando habilitado)
        var gateHuMin: Int32 = -900
        var gateHuMax: Int32 = -500
        var useHuGate: Int32 = 0

        // Dimensão real do volume (passada ao shader p/ gradiente correto)
        var dimX: Int32 = 1
        var dimY: Int32 = 1
        var dimZ: Int32 = 1

        // Aplicar TF nas projeções?
        var useTFProj: Int32 = 0

        // Padding para múltiplos de 16B (evita surpresas de alinhamento)
        var _pad0: Int32 = 0
        var _pad1: Int32 = 0
        var _pad2: Int32 = 0
    }

    // MARK: - Estado

    private var uniforms = Uniforms()
    private let uniformsKey    = "uniforms"       // [[ buffer(4) ]]
    private let dicomKey       = "dicom"          // [[ texture(0) ]]
    private let tfKey          = "transferColor"  // [[ texture(3) ]]

    private let featureFlags: FeatureFlags
    private(set) var textureGenerator: VolumeTextureFactory
    var tf: TransferFunction?
    private let device: MTLDevice
    private let uploadQueue: MTLCommandQueue
    private var uploadToken = UUID()
    private var pendingCompletion: (() -> Void)?

    /// Escala do cubo (SceneKit) = voxel spacing * dimensão (mantém proporção anatômica)
    var scale: float3 { textureGenerator.scale }
    var transform: simd_float4x4 { textureGenerator.transform }

    // MARK: - Init

    init(device: MTLDevice, featureFlags: FeatureFlags) {
        self.featureFlags = featureFlags
        self.device = device
        self.textureGenerator = VolumeTextureFactory(part: .none, featureFlags: featureFlags)
        guard let queue = device.makeCommandQueue() else {
            fatalError("Não foi possível criar command queue para upload de volume.")
        }
        queue.label = "VolumeTextureUploadQueue"
        self.uploadQueue = queue
        super.init()

        let program = SCNProgram()
        program.vertexFunctionName   = "volume_vertex"
        program.fragmentFunctionName = "volume_fragment"
        self.program = program

        // Flags de material primeiro
        cullMode = .front
        writesToDepthBuffer = true

        // Inicializa sem dados
        setPart(device: device, part: .none, completion: nil)

        // TF default com verificação - REMOVER este bloco problemático
        // if Bundle.main.url(forResource: "ct_arteries", withExtension: "tf") != nil {
        //     setPreset(device: device, preset: .ct_arteries)
        //     setShift(device: device, shift: 0)
        // }

        // Empurra uniforms iniciais
        pushUniforms()
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    // MARK: - Helpers

    private func pushUniforms() {
        var u = uniforms
        // stride garante padding correto para Metal
        let buffer = NSData(bytes: &u, length: Uniforms.stride)
        setValue(buffer, forKey: uniformsKey)
    }

    private func setDicomTexture(_ texture: MTLTexture) {
        let prop = SCNMaterialProperty(contents: texture as Any)
        setValue(prop, forKey: dicomKey)
    }

    func setTransferFunctionTexture(_ texture: MTLTexture) {
        let prop = SCNMaterialProperty(contents: texture as Any)
        setValue(prop, forKey: tfKey)
    }

    // MARK: - Snapshots

    func currentVolumeTexture() -> MTLTexture? {
        let prop = value(forKey: dicomKey) as? SCNMaterialProperty
        return prop?.contents as? MTLTexture
    }

    func currentTransferFunctionTexture() -> MTLTexture? {
        let prop = value(forKey: tfKey) as? SCNMaterialProperty
        return prop?.contents as? MTLTexture
    }

    func snapshotUniforms() -> Uniforms {
        uniforms
    }

    var datasetMeta: (dimension: int3, resolution: float3) {
        (textureGenerator.dimension, textureGenerator.resolution)
    }

    // MARK: - API de controle

    func setMethod(method: Method) {
        uniforms.method = method.idInt32
        pushUniforms()
    }

    /// Injeta o volume e atualiza dimX/Y/Z (para gradiente correto no shader).
    func setPart(device: MTLDevice, part: BodyPart, completion: (() -> Void)? = nil) {
        apply(factory: VolumeTextureFactory(part: part, featureFlags: featureFlags),
              device: device,
              completion: completion)
    }

    func setDataset(device: MTLDevice, dataset: VolumeDataset, completion: (() -> Void)? = nil) {
        apply(factory: VolumeTextureFactory(dataset: dataset, featureFlags: featureFlags),
              device: device,
              completion: completion)
    }

    func setPreset(device: MTLDevice, preset: Preset) {
        Logger.log("Tentando carregar preset: \(preset.rawValue)", level: .debug, category: "TransferFunction")

        guard let url = Bundle.main.url(forResource: preset.rawValue, withExtension: "tf") else {
            Logger.log("Arquivo não encontrado: \(preset.rawValue).tf", level: .error, category: "TransferFunction")
            if let urls = Bundle.main.urls(forResourcesWithExtension: "tf", subdirectory: nil) {
                let available = urls.map { $0.lastPathComponent }.joined(separator: ", ")
                Logger.log("Transfer functions disponíveis: \(available)", level: .info, category: "TransferFunction")
            }
            return
        }

        Logger.log("Transfer function encontrada em \(url.lastPathComponent)", level: .info, category: "TransferFunction")
        tf = TransferFunction.load(from: url)
    }

    func setLighting(on: Bool) {
        uniforms.isLightingOn = on ? 1 : 0
        pushUniforms()
    }

    func setStep(step: Float) {
        uniforms.renderingQuality = Int32(step)
        pushUniforms()
    }

    /// Desloca a TF (útil para presets que varrem faixas HU)
    func setShift(device: MTLDevice, shift: Float) {
        tf?.shift = shift
        guard let tf = tf else { return }
        if let tfTexture = tf.get(device: device) { // Unwrapping seguro
            setTransferFunctionTexture(tfTexture)
        }
    }

    /// Gating para MIP/MinIP/AIP em [0,1] (após normalização HU).
    func setDensityGate(floor: Float, ceil: Float) {
        uniforms.densityFloor = max(0, min(1, floor))
        uniforms.densityCeil  = max(uniforms.densityFloor, min(1, ceil))
        pushUniforms()
    }

    /// Aplica TF nas projeções (em vez de grayscale).
    func setUseTFOnProjections(_ on: Bool) {
        uniforms.useTFProj = on ? 1 : 0
        pushUniforms()
    }

    // MARK: - HU Gate controls (projections)
    func setHuGate(enabled: Bool) {
        uniforms.useHuGate = enabled ? 1 : 0
        pushUniforms()
    }

    func setHuWindow(minHU: Int32, maxHU: Int32) {
        let datasetRange = textureGenerator.dataset.intensityRange
        let lowerBound = min(minHU, maxHU)
        let upperBound = max(minHU, maxHU)
        let clampedMin = max(datasetRange.lowerBound, lowerBound)
        let clampedMax = min(datasetRange.upperBound, upperBound)

        uniforms.voxelMinValue = clampedMin
        uniforms.voxelMaxValue = clampedMax

        uniforms.gateHuMin = clampedMin
        uniforms.gateHuMax = clampedMax
        pushUniforms()
    }

}

private extension VolumeCubeMaterial {
    func apply(factory: VolumeTextureFactory,
               device: MTLDevice,
               completion: (() -> Void)?) {
        textureGenerator = factory
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
                Logger.log("Falha ao gerar textura para o volume.",
                           level: .error,
                           category: "VolumeCubeMaterial")
                self.pendingCompletion?()
                self.pendingCompletion = nil
                return
            }

            self.setDicomTexture(texture)

            Logger.log("Dataset aplicado dim=\(factory.dimension) range=\(factory.dataset.intensityRange)",
                       level: .debug,
                       category: "VolumeCubeMaterial")

            let dimension = factory.dimension
            self.uniforms.dimX = dimension.x
            self.uniforms.dimY = dimension.y
            self.uniforms.dimZ = dimension.z

            let range = factory.dataset.intensityRange
            self.uniforms.datasetMinValue = range.lowerBound
            self.uniforms.datasetMaxValue = range.upperBound
            self.uniforms.voxelMinValue = range.lowerBound
            self.uniforms.voxelMaxValue = range.upperBound

            self.pushUniforms()
            self.pendingCompletion?()
            self.pendingCompletion = nil
        }
    }
}
