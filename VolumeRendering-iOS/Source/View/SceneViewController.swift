//
//  SceneViewController.swift
//  VolumeRendering-iOS
//
//  Created by won on 2022/05/30.
//

import Foundation
import SceneKit
import Metal
import simd
import UIKit

enum InteractionTool: String, CaseIterable {
    case navigate, window, zoom
}
import UniformTypeIdentifiers

class SceneViewController: NSObject, UIGestureRecognizerDelegate, SCNSceneRendererDelegate {
    static let Instance = SceneViewController() // like Singleton

    struct DatasetMeta {
        let dimension: int3
        let resolution: float3
    }

    struct VolumeMetadata {
        let dimensions: int3
        let spacing: float3
        let origin: float3
        let orientation: simd_float3x3
    }

    struct ToneCurveChannelSnapshot {
        let index: Int
        let controlPoints: [ToneCurvePoint]
        let histogram: [UInt32]
        let presetKey: String
        let gain: Float
    }

    struct ClipBounds: Equatable {
        var xMin: Float = 0.0
        var xMax: Float = 1.0
        var yMin: Float = 0.0
        var yMax: Float = 1.0
        var zMin: Float = 0.0
        var zMax: Float = 1.0

        func sanitized(epsilon: Float = 1e-4) -> ClipBounds {
            func sanitizePair(minimum: Float, maximum: Float) -> (Float, Float) {
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

            let (sxMin, sxMax) = sanitizePair(minimum: xMin, maximum: xMax)
            let (syMin, syMax) = sanitizePair(minimum: yMin, maximum: yMax)
            let (szMin, szMax) = sanitizePair(minimum: zMin, maximum: zMax)

            return ClipBounds(xMin: sxMin,
                              xMax: sxMax,
                              yMin: syMin,
                              yMax: syMax,
                              zMin: szMin,
                              zMax: szMax)
        }
    }

    enum ClipPlanePreset: String, CaseIterable, Identifiable {
        case off, axial, sagittal, coronal, custom

        var id: String { rawValue }

        var displayName: String {
            switch self {
            case .off: return "Off"
            case .axial: return "Axial"
            case .sagittal: return "Sagittal"
            case .coronal: return "Coronal"
            case .custom: return "Custom"
            }
        }
    }

    struct ClipPlaneState: Equatable {
        var preset: ClipPlanePreset
        var offset: Float
        var hasCustomNormal: Bool
    }

    enum RenderMode: String, CaseIterable, Identifiable {
        var id: RawValue { rawValue }
        case surf, dvr, mip, minip, avg, mpr
    }

    enum RenderBackend {
        case fragmentShader
        case compute
    }

    struct ChannelState {
        var preset: VolumeCubeMaterial.Preset?
        var transferFunction: TransferFunction
        var texture: MTLTexture?
        var gain: Float
        var toneCurve: ToneCurveModel
        var histogram: [UInt32]
    }

    private struct CameraSignature {
        var modelMatrix: simd_float4x4
        var inverseViewProjectionMatrix: simd_float4x4

        func isApproximatelyEqual(to other: CameraSignature, epsilon: Float = 1e-4) -> Bool {
            return modelMatrix.isApproximatelyEqual(to: other.modelMatrix, epsilon: epsilon) &&
                   inverseViewProjectionMatrix.isApproximatelyEqual(to: other.inverseViewProjectionMatrix, epsilon: epsilon)
        }
    }

    private struct ComputeRenderDiagnostics {
        var missingRenderer = false
        var missingVolume = false
        var missingView = false
        var missingCamera = false
        var zeroViewport = false
        var renderFailure = false
    }

    var device: MTLDevice!
    private var featureFlags: FeatureFlags = []
    var root: SCNNode!
    var cameraController: SCNCameraController!
    
    var volume: SCNNode!
    var mat: VolumeCubeMaterial!
    // --- MPR ---
    private var mprNode: SCNNode?
    private var mprMat: MPRPlaneMaterial?
    private var importedVolume: DicomImportResult?
    private let dicomLoader = DicomVolumeLoader()
    private weak var sceneView: SCNView?
    private var renderBackend: RenderBackend = .fragmentShader
    private var computeRenderer: VolumeComputeRenderer?
    private var computePlaneNode: SCNNode?
    private var computeDiffuseProperty: SCNMaterialProperty?
    private var computeEmissionProperty: SCNMaterialProperty?
    private var computeViewportSize: CGSize = .zero
    private var defaultCameraNode: SCNNode?
    private var computeDiagnostics = ComputeRenderDiagnostics()
    private var computeDebugSampleCount: Int = 0
    private var computeDebugRayLogCount: Int = 0
    private var isRenderingCompute = false
    private var computeNeedsUpdate = false
    private var lastCameraSignature: CameraSignature?
    private var earlyTerminationThreshold: Float = 0.99
    private var lastComputeTexture: MTLTexture?
    private var channelStates: [ChannelState] = []
    private var histogramCalculator: VolumeHistogramCalculator?
    private var histogramRequestID: UInt64 = 0

    private var activeRenderMode: RenderMode = .dvr
    private var adaptiveOn: Bool = true
    private var adaptiveThreshold: Float = 0.1
    private var jitterAmount: Float = 0.0
    private var lastStep: Float = 512
    private var interactionFactor: Float = 0.35 // 35% dos steps durante interação
    private var currentHuWindowRange: (min: Int32, max: Int32) = (-1024, 3071)

    private var mprAxNode: SCNNode?
    private var mprCoNode: SCNNode?
    private var mprSaNode: SCNNode?

    private var clipBounds = ClipBounds()
    private var clipBoxQuaternion = simd_quatf()
    private var clipPlanePreset: ClipPlanePreset = .off
    private var clipPlaneOffset: Float = 0.0
    private var customClipPlaneNormal = SIMD3<Float>(0, 0, 1)
    private var volumeMetadata: VolumeMetadata?

    private var activeInteractionTool: InteractionTool = .navigate
    private weak var rotateGesture: UIPanGestureRecognizer?
    private weak var zoomGesture: UIPinchGestureRecognizer?
    private weak var adaptivePanGesture: UIPanGestureRecognizer?
    private weak var adaptivePinchGesture: UIPinchGestureRecognizer?
    private weak var adaptiveRotateGesture: UIRotationGestureRecognizer?
    private var initialEulerAngles = SCNVector3Zero
    private var initialScale: Float = 1.0
    private var windowPanStart: (min: Float, max: Float)?
    private var zoomPanStart: Float?

    
    override public init() { super.init() }

    // 0 = X, 1 = Y, 2 = Z (default: axial = Z)
    private var currentPlaneAxis: Int = 2

    // Conveniência para a UI: número de fatias no eixo Z
    var mprDimZ: Int {
        Int(mprMat?.dimension.z ?? 1)
    }

    // Dimensão ao longo do plano corrente (p/ step do slider)
    var mprDimCurrent: Int {
        guard let d = mprMat?.dimension else { return 1 }
        switch currentPlaneAxis {
        case 0: return Int(d.x)
        case 1: return Int(d.y)
        default: return Int(d.z)
        }
    }
    
    func setMPRPlane(_ plane: DrawOptionModel.MPRPlane) {
        guard let d = mprMat?.dimension else { return }
        switch plane {
        case .axial:
            currentPlaneAxis = 2
            setMPRPlaneAxial(slice: Int(d.z / 2))
        case .coronal:
            currentPlaneAxis = 1
            setMPRPlaneCoronal(row: Int(d.y / 2))
        case .sagittal:
            currentPlaneAxis = 0
            setMPRPlaneSagittal(column: Int(d.x / 2))
        }
    }

    // Atualiza o índice da fatia no plano atual a partir de [0..1]
    func updateSlice(normalizedValue: Float) {
        guard let d = mprMat?.dimension else { return }
        switch currentPlaneAxis {
        case 0:
            let n = max(1, d.x - 1)
            setMPRPlaneSagittal(column: Int(round(normalizedValue * Float(n))))
        case 1:
            let n = max(1, d.y - 1)
            setMPRPlaneCoronal(row: Int(round(normalizedValue * Float(n))))
        default:
            let n = max(1, d.z - 1)
            setMPRPlaneAxial(slice: Int(round(normalizedValue * Float(n))))
        }
    }

    func registerToolGestures(rotate: UIPanGestureRecognizer,
                              zoom: UIPinchGestureRecognizer) {
        rotateGesture = rotate
        zoomGesture = zoom
        updateGestureStates()
    }

    func registerAdaptiveGestures(pan: UIPanGestureRecognizer,
                                  pinch: UIPinchGestureRecognizer,
                                  rotate: UIRotationGestureRecognizer) {
        adaptivePanGesture = pan
        adaptivePinchGesture = pinch
        adaptiveRotateGesture = rotate
    }

    func setActiveTool(_ tool: InteractionTool) {
        activeInteractionTool = tool
        windowPanStart = nil
        zoomPanStart = nil
        updateGestureStates()
    }

    private func updateGestureStates() {
        rotateGesture?.isEnabled = true
        zoomGesture?.isEnabled = activeInteractionTool == .zoom
    }

    func gestureRecognizer(_ gestureRecognizer: UIGestureRecognizer,
                           shouldRecognizeSimultaneouslyWith otherGestureRecognizer: UIGestureRecognizer) -> Bool {
        let toolGestures: [UIGestureRecognizer?] = [rotateGesture, zoomGesture]
        let adaptiveGestures: [UIGestureRecognizer?] = [adaptivePanGesture, adaptivePinchGesture, adaptiveRotateGesture]

        let gestureIsTool = toolGestures.contains { $0 === gestureRecognizer }
        let gestureIsAdaptive = adaptiveGestures.contains { $0 === gestureRecognizer }
        let otherIsTool = toolGestures.contains { $0 === otherGestureRecognizer }
        let otherIsAdaptive = adaptiveGestures.contains { $0 === otherGestureRecognizer }

        if (gestureIsTool && otherIsAdaptive) || (gestureIsAdaptive && otherIsTool) {
            return true
        }

        return false
    }

    func renderer(_ renderer: SCNSceneRenderer, updateAtTime time: TimeInterval) {
        guard renderBackend == .compute else { return }
        renderComputeFrame(using: renderer)
    }

    func onAppear(_ view: SCNView) {
        sceneView = view
         // Device Metal com fallback
        guard let dev = view.device ?? MTLCreateSystemDefaultDevice() else {
            assertionFailure("Metal não disponível no dispositivo")
            return
        }
        self.device = dev
        self.featureFlags = FeatureFlags.evaluate(for: dev)
        self.histogramCalculator = VolumeHistogramCalculator(device: dev, featureFlags: featureFlags)

        Logger.log("Feature flags detectados: \(featureFlags.description)",
                   level: .info,
                   category: "Metal")

        if !featureFlags.contains(.argumentBuffers) {
            Logger.log("Argument buffers indisponíveis; usando pipeline legacy com setTexture/setBuffer.",
                       level: .warn,
                       category: "Metal")
        }
        if !featureFlags.contains(.nonUniformThreadgroups) {
            Logger.log("Non-uniform threadgroups não suportado; usando dispatchThreadgroups padrão.",
                       level: .warn,
                       category: "Metal")
        }
        if !featureFlags.contains(.heapAllocations) {
            Logger.log("Heaps/private storage indisponíveis; texturas usarão storageMode.shared.",
                       level: .warn,
                       category: "Metal")
        }

        // Cena
        view.isPlaying = true
        let scene = view.scene ?? SCNScene()
        view.scene = scene
        scene.isPaused = false
        root = scene.rootNode
        cameraController = view.defaultCameraController
        ensureDefaultCamera(in: scene, view: view)
        
        let box = SCNBox(width: 1, height: 1, length: 1, chamferRadius: 0)
        mat = VolumeCubeMaterial(device: device, featureFlags: featureFlags)
        mat.setPart(device: device, part: .none)

        // SceneViewController.onAppear(...)
        var initialTransferTexture: MTLTexture?
        if let url = Bundle.main.url(forResource: "ct_arteries", withExtension: "tf"),
           let tfTex = TransferFunction.load(from: url).get(device: device) {
            initialTransferTexture = tfTex
        } else if let tfFallback = TransferFunction().get(device: device) {
            initialTransferTexture = tfFallback
        }

        if let tfTex = initialTransferTexture {
            mat.setTransferFunctionTexture(tfTex)
        }

        setupChannelDefaults(device: device, initialTexture: initialTransferTexture)

        // Default de qualidade quando não houver lastStep do usuário
        mat.setStep(step: lastStep)
        let initialUniforms = mat.snapshotUniforms()
        currentHuWindowRange = (initialUniforms.voxelMinValue, initialUniforms.voxelMaxValue)
        volume = SCNNode(geometry: box)
        volume.geometry?.materials = [mat]
        volume.simdTransform = mat.transform
        root.addChildNode(volume)
        updateCameraPlacement()

        if featureFlags.contains(.argumentBuffers) {
            do {
                let renderer = try VolumeComputeRenderer(device: device, featureFlags: featureFlags)
                computeRenderer = renderer
                renderer.setEarlyTerminationThreshold(earlyTerminationThreshold)
                renderer.setAdaptiveEnabled(adaptiveOn)
                renderer.setAdaptiveThreshold(adaptiveThreshold)
                renderer.setJitterAmount(jitterAmount)
                renderer.setDensityDebugEnabled(AppConfig.ENABLE_DENSITY_DEBUG)
                renderBackend = .compute
                refreshToneBuffersForAllChannels()
                propagateChannelStateToRenderers()
                propagateClipStateToRenderers()
                setupComputePlaneIfNeeded(in: view)
                computeNeedsUpdate = true
                view.delegate = self
                view.isPlaying = true
                Logger.log("Renderizador compute habilitado (SceneKit→Compute).",
                           level: .info,
                           category: "VolumeCompute")
            } catch {
                renderBackend = .fragmentShader
                computeRenderer = nil
                Logger.log("Falha ao inicializar compute renderer: \(error.localizedDescription)",
                           level: .warn,
                           category: "VolumeCompute")
            }
        } else {
            renderBackend = .fragmentShader
            computeRenderer = nil
        }

        updateNodesForBackend()
        
        // for depth test
        // let node2 = SCNNode(geometry: SCNBox(width: 0.2, height: 0.2, length: 0.2, chamferRadius: 0))
        // node2.geometry?.firstMaterial?.diffuse.contents = UIColor.yellow
        // node2.position = SCNVector3Make(0.5, 0, 0.5)
        // root.addChildNode(node2)
  
        // let node3 = SCNNode(geometry: SCNSphere(radius: 0.2))
        // node3.geometry?.firstMaterial?.diffuse.contents = UIColor.green
        // node3.position = SCNVector3Make(-0.5, 0, 0.5)
        // root.addChildNode(node3)
        
        cameraController.target = volume.boundingSphere.center

        // MPR: criar mas não configurar ainda
        let plane = SCNPlane(width: 1, height: 1)
        let mpr = MPRPlaneMaterial(device: device, featureFlags: featureFlags)
        mpr.setPart(device: device, part: .none) // <-- placeholder 3D válido
        if let tfTex = mat.currentTransferFunctionTexture() ?? initialTransferTexture {
            mpr.setTransferFunction(tfTex)
        }
        let node = SCNNode(geometry: plane)
        node.geometry?.materials = [mpr]
        node.isHidden = true
        root.addChildNode(node)
        node.simdTransform = volume.simdTransform // mantém o mesmo frame do volume
        self.mprNode = node
        self.mprMat = mpr
    }

    private func syncMPRTransformWithVolume() {
        guard let mprNode = mprNode else { return }
        mprNode.simdTransform = volume.simdTransform
    }

    private func syncMPRTransferFunction() {
        if let tfTexture = mat.currentTransferFunctionTexture() {
            mprMat?.setTransferFunction(tfTexture)
        }
        notifyMPRTransferFunctionObservers()
        markComputeDirty()
    }
    
    func setMethod(method: VolumeCubeMaterial.Method) {
        mat.setMethod(method: method)
        markComputeDirty()
    }
    
    func setPart(part: VolumeCubeMaterial.BodyPart) {
        if part == .dicom {
            guard let imported = importedVolume else {
                mat.setPart(device: device, part: .dicom) { [weak self] in
                    guard let self else { return }
                    self.updateMetadata(from: self.mat.textureGenerator.dataset)
                    self.postVolumeUpdate(usingMprPart: .dicom)
                }
                return
            }
            applyImportedDataset(imported.dataset)
            return
        }

        mat.setPart(device: device, part: part) { [weak self] in
            guard let self else { return }
            self.updateMetadata(from: self.mat.textureGenerator.dataset)
            self.postVolumeUpdate(usingMprPart: part)
        }
    }
    
    func setPreset(preset: VolumeCubeMaterial.Preset) {
        mat.setPreset(device: device, preset: preset)
        mat.setShift(device: device, shift: 0)
        if channelStates.isEmpty {
            setupChannelDefaults(device: device, initialTexture: mat.currentTransferFunctionTexture())
        }
        if channelStates.indices.contains(0) {
            channelStates[0].preset = preset
            if let updatedTransfer = mat.tf {
                channelStates[0].transferFunction = updatedTransfer
            }
            channelStates[0].texture = mat.currentTransferFunctionTexture()
        }
        syncMPRTransferFunction()
        propagateChannelStateToRenderers()
        markComputeDirty()
    }

    func setLighting(isOn: Bool) {
        mat.setLighting(on: isOn)
        markComputeDirty()
    }

    func setStep(step: Float) {
        let clamped = max(step, 64)
        lastStep = clamped
        mat.setStep(step: clamped)
        markComputeDirty()
    }

    func setShift(shift: Float) {
        updateChannelShift(for: 0, shift: shift)
    }

    func setChannelPreset(channel index: Int, presetKey: String) {
        if presetKey == "none" {
            updateChannelPreset(nil, for: index, resetGain: true, resetShift: true)
            return
        }

        if let preset = VolumeCubeMaterial.Preset(rawValue: presetKey) {
            let shouldResetGain: Bool
            if channelStates.indices.contains(index) {
                shouldResetGain = channelStates[index].gain <= 0.0001
            } else {
                shouldResetGain = true
            }
            updateChannelPreset(preset, for: index, resetGain: shouldResetGain, resetShift: true)
        } else {
            Logger.log("Preset desconhecido recebido: \(presetKey)",
                       level: .warn,
                       category: "TransferFunction")
        }
    }

    func setChannelGain(channel index: Int, gain: Float) {
        guard channelStates.indices.contains(index) else { return }
        channelStates[index].gain = max(0, gain)
        propagateChannelStateToRenderers()
        markComputeDirty()
    }

    func channelControlSnapshot() -> [(presetKey: String, gain: Float)] {
        guard !channelStates.isEmpty else {
            return Array(repeating: ("none", 0), count: 4)
        }
        return channelStates.enumerated().map { (_, state) in
            (state.preset?.rawValue ?? "none", state.gain)
        }
    }

    // MARK: - Tone Curve & Histogram

    func updateToneCurve(channel index: Int, controlPoints: [ToneCurvePoint]) {
        guard channelStates.indices.contains(index) else { return }
        channelStates[index].toneCurve.setControlPoints(controlPoints)
        refreshToneBuffer(for: index)
        notifyToneCurveObservers(channel: index)
        markComputeDirty()
    }

    func applyAutoWindowPreset(_ preset: ToneCurveAutoWindowPreset, channel index: Int) {
        guard channelStates.indices.contains(index) else { return }
        channelStates[index].toneCurve.applyAutoWindow(preset)
        refreshToneBuffer(for: index)
        notifyToneCurveObservers(channel: index)
        markComputeDirty()
    }

    func toneCurveControlPoints(for channel: Int) -> [ToneCurvePoint] {
        guard channelStates.indices.contains(channel) else { return [] }
        return channelStates[channel].toneCurve.currentControlPoints()
    }

    func histogram(for channel: Int) -> [UInt32] {
        guard channelStates.indices.contains(channel) else { return [] }
        return channelStates[channel].histogram
    }

    func toneCurvePresets() -> [ToneCurveAutoWindowPreset] {
        ToneCurveAutoWindowPreset.allPresets
    }

    func resetToneCurve(channel index: Int) {
        guard channelStates.indices.contains(index) else { return }
        channelStates[index].toneCurve.reset()
        refreshToneBuffer(for: index)
        notifyToneCurveObservers(channel: index)
        markComputeDirty()
    }

    func insertToneCurvePoint(channel index: Int, point: ToneCurvePoint) {
        guard channelStates.indices.contains(index) else { return }
        channelStates[index].toneCurve.insertPoint(point)
        refreshToneBuffer(for: index)
        notifyToneCurveObservers(channel: index)
        markComputeDirty()
    }

    func removeToneCurvePoint(channel index: Int, pointIndex: Int) {
        guard channelStates.indices.contains(index) else { return }
        channelStates[index].toneCurve.removePoint(at: pointIndex)
        refreshToneBuffer(for: index)
        notifyToneCurveObservers(channel: index)
        markComputeDirty()
    }

    func toneCurveSnapshot() -> [ToneCurveChannelSnapshot] {
        channelStates.enumerated().map { index, state in
            ToneCurveChannelSnapshot(index: index,
                                     controlPoints: state.toneCurve.currentControlPoints(),
                                     histogram: state.histogram,
                                     presetKey: state.preset?.rawValue ?? "none",
                                     gain: state.gain)
        }
    }

    func refreshToneBuffersForAllChannels() {
        guard let computeRenderer else { return }
        for index in 0..<channelStates.count {
            let samples = channelStates[index].toneCurve.sampledValues()
            computeRenderer.updateToneCurve(channel: index, values: samples)
        }
    }

    private func refreshToneBuffer(for channel: Int) {
        guard channelStates.indices.contains(channel),
              let computeRenderer else { return }
        let samples = channelStates[channel].toneCurve.sampledValues()
        computeRenderer.updateToneCurve(channel: channel, values: samples)
    }

    func enqueueHistogramUpdate() {
        guard let histogramCalculator,
              let volumeTexture = mat.currentVolumeTexture() else { return }

        let uniforms = mat.snapshotUniforms()
        let channelCount = max(1, channelStates.count)
        histogramRequestID &+= 1
        let requestID = histogramRequestID

        histogramCalculator.computeHistogram(for: volumeTexture,
                                             channelCount: channelCount,
                                             voxelMin: Int32(uniforms.voxelMinValue),
                                             voxelMax: Int32(uniforms.voxelMaxValue),
                                             bins: AppConfig.HISTOGRAM_BIN_COUNT) { [weak self] result in
            DispatchQueue.main.async {
                guard let self else { return }
                guard requestID == self.histogramRequestID else { return }

                switch result {
                case .success(let histograms):
                    for (index, histogram) in histograms.enumerated() where index < self.channelStates.count {
                        self.channelStates[index].histogram = histogram
                        self.channelStates[index].toneCurve.setHistogram(histogram)
                    }
                    self.notifyHistogramObservers()
                case .failure(let error):
                    Logger.log("Falha ao calcular histograma: \(error.localizedDescription)",
                               level: .warn,
                               category: "Histogram")
                }
            }
        }
    }

    func loadDicomSeries(from url: URL,
                         progress: @escaping (DicomVolumeLoader.UIProgressUpdate) -> Void,
                         completion: @escaping (Result<DicomImportResult, Error>) -> Void) {
        dicomLoader.loadVolume(from: url, progress: { update in
            switch update {
            case .partialPreview(let dataset, let fraction):
                self.applyImportedDataset(dataset)
                progress(.previewAvailable(fraction))
            case .started, .reading:
                progress(DicomVolumeLoader.uiUpdate(from: update))
            }
        }, completion: { result in
            switch result {
            case .success(let importResult):
                self.importedVolume = importResult
                self.applyImportedDataset(importResult.dataset)
                progress(.previewAvailable(1.0))
                completion(.success(importResult))
            case .failure(let error):
                completion(.failure(error))
            }
        })
    }

    // MARK: - Render mode (VR vs MPR)
    func setRenderMode(_ mode: RenderMode) {
        activeRenderMode = mode
        let isMpr = (mode == .mpr)

        if renderBackend == .fragmentShader && !isMpr {
            rebindVolumeNodeForVR()
        }

        mprNode?.isHidden = !isMpr

        if !isMpr {
            mat.cullMode = .front
            mat.setMethod(method: mapToVRMethod(mode))
            markComputeDirty()
        }

        updateNodesForBackend()
    }

    func restoreVolumeIfNeeded(in view: SCNView) {
        if view.scene == nil {
            view.scene = SCNScene()
        }

        if let sceneRoot = view.scene?.rootNode, root == nil || root !== sceneRoot {
            root = sceneRoot
        }

        if renderBackend == .fragmentShader {
            rebindVolumeNodeForVR()
        } else {
            setupComputePlaneIfNeeded(in: view)
            updateNodesForBackend()
            markComputeDirty()
        }
    }


    private func mapToVRMethod(_ mode: RenderMode) -> VolumeCubeMaterial.Method {
        switch mode {
        case .surf:  return .surf
        case .dvr:   return .dvr
        case .mip:   return .mip
        case .minip: return .minip
        case .avg:   return .avg
        case .mpr:   return .dvr // não usado; apenas para satisfazer retorno
        }
    }

     // =========================
    // MARK: - MPR Controls MVP
    // =========================
    func enableMPR(_ on: Bool) {
        mprNode?.isHidden = !on
        // Opcional: quando MPR ligar, podemos desligar o volume para não "brigar".
        // volume.isHidden = on
        updateNodesForBackend()
        if !on {
            markComputeDirty()
        }
        requestMPRDisplayRefresh()
    }

    func setMPRBlend(_ mode: MPRPlaneMaterial.BlendMode) {
        mprMat?.setBlend(mode)
        renderMPRAndPresent()
    }

    func setMPRHuWindow(min: Int32, max: Int32) {
        mprMat?.setHU(min: min, max: max)
        renderMPRAndPresent()
    }

    func setMPRSlab(thicknessInVoxels: Int, steps: Int) {
        mprMat?.setSlab(thicknessInVoxels: thicknessInVoxels, steps: steps)
        renderMPRAndPresent()
    }

    func setMPRPlaneAxial(slice k: Int) {
        currentPlaneAxis = 2
        mprMat?.setAxial(slice: k)
        renderMPRAndPresent()
    }

    func setMPRPlaneSagittal(column i: Int) {
        currentPlaneAxis = 0
        mprMat?.setSagittal(column: i)
        renderMPRAndPresent()
    }

    func setMPRPlaneCoronal(row j: Int) {
        currentPlaneAxis = 1
        mprMat?.setCoronal(row: j)
        renderMPRAndPresent()
    }

    func setDensityGate(floor: Float, ceil: Float) {
        mat.setDensityGate(floor: floor, ceil: ceil)
        markComputeDirty()
    }
    func setUseTFOnProjections(_ on: Bool) {
        mat.setUseTFOnProjections(on)
        markComputeDirty()
    }
    func setHuGate(enabled: Bool) {
        mat.setHuGate(enabled: enabled)
        markComputeDirty()
    }
    func setHuWindow(minHU: Int32, maxHU: Int32) {
        mat.setHuWindow(minHU: minHU, maxHU: maxHU)
        refreshHuWindowFromMaterial()
        markComputeDirty()
    }

    func handleEarlyTerminationShortcut(for input: String) {
        let presets: [String: Float] = ["1": 0.90, "2": 0.95, "3": 0.99]
        guard let target = presets[input] else { return }
        updateEarlyTerminationThreshold(to: target)
    }

    private func updateEarlyTerminationThreshold(to newValue: Float) {
        let clamped = max(0.0, min(newValue, 0.9999))
        guard abs(earlyTerminationThreshold - clamped) > 0.0001 else { return }

        earlyTerminationThreshold = clamped
        computeRenderer?.setEarlyTerminationThreshold(clamped)
        Logger.log(String(format: "Early termination threshold set to %.2f", clamped),
                   level: .info,
                   category: "VolumeCompute")
        markComputeDirty()
    }

    func setMPRUseTF(_ on: Bool) {
        mprMat?.setUseTF(on)
        renderMPRAndPresent()
    }

    func setMPRMaskTexture(_ texture: MTLTexture?) {
        mprMat?.setOverlayMask(texture)
        renderMPRAndPresent()
    }

    func setMPROverlayOpacity(_ value: Float) {
        mprMat?.setOverlayOpacity(value)
        renderMPRAndPresent()
    }

    func setMPROverlayColor(_ color: SIMD3<Float>) {
        mprMat?.setOverlayColor(color)
        renderMPRAndPresent()
    }

    func setMPROverlayChannel(_ channel: Int32) {
        mprMat?.setOverlayChannel(channel)
        renderMPRAndPresent()
    }

    private func handleWindowPan(with translation: CGPoint, state: UIGestureRecognizer.State) {
        guard let mat else { return }
        switch state {
        case .began:
            let current = currentHuWindowRange
            windowPanStart = (Float(current.min), Float(current.max))
        case .changed:
            guard let start = windowPanStart else {
                let current = currentHuWindowRange
                windowPanStart = (Float(current.min), Float(current.max))
                return
            }
            var datasetMin = Float(-4096)
            var datasetMax = Float(4096)
            let uniforms = mat.snapshotUniforms()
            datasetMin = Float(uniforms.datasetMinValue)
            datasetMax = Float(uniforms.datasetMaxValue)
            let startWidth = max(1.0, start.max - start.min)
            let startLevel = (start.max + start.min) * 0.5
            let levelSensitivity: Float = 1.0
            let widthSensitivity: Float = 2.0
            let deltaLevel = Float(translation.x) * levelSensitivity
            let deltaWidth = Float(-translation.y) * widthSensitivity
            var width = max(1.0, startWidth + deltaWidth)
            let maxWidth = max(1.0, datasetMax - datasetMin)
            width = min(width, maxWidth)
            var level = startLevel + deltaLevel
            let halfWidth = width * 0.5
            level = max(datasetMin + halfWidth, min(datasetMax - halfWidth, level))
            var newMin = level - halfWidth
            var newMax = level + halfWidth
            newMin = max(datasetMin, newMin)
            newMax = min(datasetMax, newMax)
            let minHU = Int32(round(newMin))
            let maxHU = Int32(round(newMax))
            guard minHU < maxHU else { return }
            setHuWindow(minHU: minHU, maxHU: maxHU)
        default:
            windowPanStart = nil
        }
    }

    @objc func handleRotateToolPan(_ gesture: UIPanGestureRecognizer) {
        guard let node = volume else { return }
        let translation = gesture.translation(in: gesture.view)
        switch activeInteractionTool {
        case .navigate:
            switch gesture.state {
            case .began:
                initialEulerAngles = node.eulerAngles
            case .changed:
                let factor: Float = 0.005
                let deltaX = Float(translation.y) * factor
                let deltaY = Float(translation.x) * factor
                node.eulerAngles.x = initialEulerAngles.x + deltaX
                node.eulerAngles.y = initialEulerAngles.y + deltaY
                syncMPRTransformWithVolume()
                markComputeDirty()
            default:
                break
            }
        case .window:
            handleWindowPan(with: translation, state: gesture.state)
        case .zoom:
            switch gesture.state {
            case .began:
                initialScale = node.scale.x
                zoomPanStart = initialScale
            case .changed:
                guard let start = zoomPanStart else {
                    zoomPanStart = node.scale.x
                    return
                }
                let sensitivity: Float = 0.003
                let delta = Float(-translation.y) * sensitivity
                let scaleFactor = powf(2.0, delta)
                let newScale = max(0.05, min(10.0, start * scaleFactor))
                node.scale = SCNVector3(x: newScale, y: newScale, z: newScale)
                syncMPRTransformWithVolume()
                markComputeDirty()
            default:
                zoomPanStart = nil
            }
        }
    }

    @objc func handleZoomToolPinch(_ gesture: UIPinchGestureRecognizer) {
        guard activeInteractionTool == .zoom, let node = volume else { return }
        switch gesture.state {
        case .began:
            initialScale = node.scale.x
        case .changed:
            let newScale = max(0.05, min(10.0, initialScale * Float(gesture.scale)))
            node.scale = SCNVector3(x: newScale, y: newScale, z: newScale)
            syncMPRTransformWithVolume()
            markComputeDirty()
        default:
            break
        }
    }
    func setAdaptive(_ on: Bool) {
        adaptiveOn = on
        computeRenderer?.setAdaptiveEnabled(on)
        markComputeDirty()
    }

    func setAdaptiveThreshold(_ value: Float) {
        let clamped = max(value, 0.0)
        guard abs(clamped - adaptiveThreshold) > 1e-4 else { return }
        adaptiveThreshold = clamped
        computeRenderer?.setAdaptiveThreshold(clamped)
        markComputeDirty()
    }

    func currentAdaptiveThreshold() -> Float {
        adaptiveThreshold
    }

    func isAdaptiveEnabled() -> Bool {
        adaptiveOn
    }

    func setJitterAmount(_ value: Float) {
        let clamped = max(0.0, min(value, 1.0))
        guard abs(clamped - jitterAmount) > 1e-4 else { return }
        jitterAmount = clamped
        computeRenderer?.setJitterAmount(clamped)
        markComputeDirty()
    }

    func currentJitterAmount() -> Float {
        jitterAmount
    }

    // --- Adaptive: reduzir steps enquanto interage ---
    @objc func handlePan(_ g: UIPanGestureRecognizer) {
        switch g.state {
        case .began:
            if adaptiveOn { mat.setStep(step: max(64, lastStep * interactionFactor)) }
        case .ended, .cancelled, .failed:
            if adaptiveOn { mat.setStep(step: lastStep) }
        default: break
        }
    }
    @objc func handlePinch(_ g: UIPinchGestureRecognizer) {
        switch g.state {
        case .began:
            if adaptiveOn { mat.setStep(step: max(64, lastStep * interactionFactor)) }
        case .ended, .cancelled, .failed:
            if adaptiveOn { mat.setStep(step: lastStep) }
        default: break
        }
    }
    @objc func handleRotate(_ g: UIRotationGestureRecognizer) {
        switch g.state {
        case .began:
            if adaptiveOn { mat.setStep(step: max(64, lastStep * interactionFactor)) }
        case .ended, .cancelled, .failed:
            if adaptiveOn { mat.setStep(step: lastStep) }
        default: break
        }
    }

    func setMPROblique(using geom: DICOMGeometry,
                    originMm: simd_float3,
                    axisUMm: simd_float3,
                    axisVMm: simd_float3)
    {
        let (o,u,v) = geom.planeWorldToTex(originW: originMm, axisUW: axisUMm, axisVW: axisVMm)
        mprMat?.setOblique(origin: float3(o), axisU: float3(u), axisV: float3(v))
        if let node = mprNode {
            node.setTransformFromBasisTex(originTex: o, UTex: u, VTex: v)
            node.isHidden = false
        }
        currentPlaneAxis = 2
        renderMPRAndPresent()
    }

    func updateAxialSlice(normalizedValue: Float) {
        guard let dim = mprMat?.dimension else { return }
        let sliceCount = Float(max(1, dim.z - 1))
        let sliceIndex = Int(round(normalizedValue * sliceCount))
        setMPRPlaneAxial(slice: sliceIndex)
    }

    func setDatasetForMPROnly(dimension: int3, resolution: float3) {
        mprMat?.setDataset(dimension: dimension, resolution: resolution)
        let scale = SCNVector3(
            resolution.x * Float(dimension.x),
            resolution.y * Float(dimension.y),
            resolution.z * Float(dimension.z)
        )
        mprNode?.scale = scale
        renderMPRAndPresent()
    }

    enum MPRExportError: Error {
        case missingMaterial
        case invalidRange
        case writerFailure
    }

    func exportMPRSeries(plane: TriPlane,
                         sliceRange: ClosedRange<Int>? = nil,
                         destination url: URL,
                         progress: ((Int, Int) -> Void)? = nil) throws {
        guard let mprMat else { throw MPRExportError.missingMaterial }

        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)

        let maxIndex: Int
        switch plane {
        case .axial: maxIndex = max(Int(mprMat.dimension.z) - 1, 0)
        case .coronal: maxIndex = max(Int(mprMat.dimension.y) - 1, 0)
        case .sagittal: maxIndex = max(Int(mprMat.dimension.x) - 1, 0)
        }

        let clampedRange: ClosedRange<Int>
        if let provided = sliceRange {
            let lower = max(0, min(provided.lowerBound, maxIndex))
            let upper = max(0, min(provided.upperBound, maxIndex))
            guard lower <= upper else { throw MPRExportError.invalidRange }
            clampedRange = lower...upper
        } else {
            clampedRange = 0...maxIndex
        }

        let previousAxis = currentPlaneAxis
        let previousIndex = mprMat.currentSliceIndex(for: previousAxis)

        let totalCount = clampedRange.upperBound - clampedRange.lowerBound + 1
        let uti = UTType.tiff.identifier as CFString

        var completed = 0
        for slice in clampedRange {
            var exportError: Error?
            autoreleasepool {
                switch plane {
                case .axial:
                    setMPRPlaneAxial(slice: slice)
                case .coronal:
                    setMPRPlaneCoronal(row: slice)
                case .sagittal:
                    setMPRPlaneSagittal(column: slice)
                }

                mprMat.renderIfNeeded()
                guard let image = mprMat.currentCGImage() else {
                    exportError = MPRExportError.writerFailure
                    return
                }

                let destinationURL = url.appendingPathComponent(String(format: "slice_%04d.tiff", slice))
                guard let destination = CGImageDestinationCreateWithURL(destinationURL as CFURL, uti, 1, nil) else {
                    exportError = MPRExportError.writerFailure
                    return
                }

                CGImageDestinationAddImage(destination, image, nil)
                if !CGImageDestinationFinalize(destination) {
                    exportError = MPRExportError.writerFailure
                }
            }

            if let error = exportError { throw error }

            completed += 1
            progress?(completed, totalCount)
        }

        switch previousAxis {
        case 0:
            setMPRPlaneSagittal(column: previousIndex)
        case 1:
            setMPRPlaneCoronal(row: previousIndex)
        default:
            setMPRPlaneAxial(slice: previousIndex)
        }
        mprMat.renderIfNeeded()
    }

    func enableTriPlanarMPR(_ on: Bool) {
        if on {
            if mprAxNode == nil { mprAxNode = makeMPRNode(axis: 2) /* axial */ }
            if mprCoNode == nil { mprCoNode = makeMPRNode(axis: 1) /* coronal */ }
            if mprSaNode == nil { mprSaNode = makeMPRNode(axis: 0) /* sagital */ }
            [mprAxNode, mprCoNode, mprSaNode].forEach { $0?.isHidden = false }
        } else {
            [mprAxNode, mprCoNode, mprSaNode].forEach { $0?.isHidden = true }
        }
    }

    private func makeMPRNode(axis: Int) -> SCNNode {
        let plane = SCNPlane(width: 1, height: 1)
        let mpr = MPRPlaneMaterial(device: device, featureFlags: featureFlags)
        // share volume+TF
        if let tex = (mat.value(forKey: "dicom") as? SCNMaterialProperty)?.contents as? MTLTexture {
            mpr.setValue(SCNMaterialProperty(contents: tex), forKey: "volume")
        } else {
            mpr.setPart(device: device, part: VolumeCubeMaterial.BodyPart.none) // fallback
        }
        if let tfTex = (mat.value(forKey: "transferColor") as? SCNMaterialProperty)?.contents as? MTLTexture {
            mpr.setTransferFunction(tfTex)
        }
        let node = SCNNode(geometry: plane)
        node.geometry?.materials = [mpr]
        node.renderingOrder = 10 + axis
        node.isHidden = false
        root.addChildNode(node)
        node.simdTransform = volume.simdTransform

        // posicionar no meio de cada eixo
        if axis == 2 { mpr.setAxial(slice: Int((mpr.dimension.z)/2)) }
        if axis == 1 { mpr.setCoronal(row:  Int((mpr.dimension.y)/2)) }
        if axis == 0 { mpr.setSagittal(column:Int((mpr.dimension.x)/2)) }

        mpr.renderIfNeeded()

        return node
    }
}

extension SceneViewController {
    func setupComputePlaneIfNeeded(in view: SCNView) {
        let pixelSize = CGSize(width: view.bounds.width * view.contentScaleFactor,
                               height: view.bounds.height * view.contentScaleFactor)

        guard computePlaneNode == nil else {
            updateComputePlaneSizeIfNeeded(viewportSize: pixelSize)
            return
        }

        let plane = SCNPlane(width: 1, height: 1)
        plane.cornerRadius = 0
        let material = SCNMaterial()
        material.lightingModel = .constant
        material.isDoubleSided = true
        material.writesToDepthBuffer = false
        material.readsFromDepthBuffer = false

        let diffuseProperty = material.diffuse
        diffuseProperty.contents = UIColor.black
        diffuseProperty.wrapS = .clamp
        diffuseProperty.wrapT = .clamp
        diffuseProperty.magnificationFilter = .linear
        diffuseProperty.minificationFilter = .linear
        diffuseProperty.mipFilter = .none

        let emissionProperty = material.emission
        emissionProperty.contents = UIColor.black
        emissionProperty.wrapS = .clamp
        emissionProperty.wrapT = .clamp
        emissionProperty.magnificationFilter = .linear
        emissionProperty.minificationFilter = .linear
        emissionProperty.mipFilter = .none
        plane.materials = [material]

        let node = SCNNode(geometry: plane)
        node.name = "VolumeComputePlane"
        node.constraints = [SCNBillboardConstraint()]
        node.simdPosition = .zero
        node.renderingOrder = 999
        root.addChildNode(node)

        computePlaneNode = node
        computeDiffuseProperty = diffuseProperty
        computeEmissionProperty = emissionProperty
        updateComputePlaneSizeIfNeeded(viewportSize: pixelSize)
    }

    func updateComputePlaneSizeIfNeeded(for view: SCNView) {
        let pixelSize = CGSize(width: view.bounds.width * view.contentScaleFactor,
                               height: view.bounds.height * view.contentScaleFactor)
        updateComputePlaneSizeIfNeeded(viewportSize: pixelSize)
    }

    private func updateComputePlaneSizeIfNeeded(viewportSize: CGSize) {
        guard viewportSize.width > 0, viewportSize.height > 0 else { return }

        if Thread.isMainThread {
            applyComputePlaneSizeUpdate(viewportSize: viewportSize)
        } else {
            DispatchQueue.main.async { [weak self] in
                self?.applyComputePlaneSizeUpdate(viewportSize: viewportSize)
            }
        }
    }

    private func applyComputePlaneSizeUpdate(viewportSize: CGSize) {
        guard Thread.isMainThread else {
            DispatchQueue.main.async { [weak self] in
                self?.applyComputePlaneSizeUpdate(viewportSize: viewportSize)
            }
            return
        }

        guard let plane = computePlaneNode?.geometry as? SCNPlane else { return }
        if computeViewportSize == viewportSize { return }
        computeViewportSize = viewportSize
        let aspect = viewportSize.width / max(viewportSize.height, 1)
        plane.height = 2.0
        plane.width = Double(aspect) * plane.height
        computeNeedsUpdate = true
    }

    private func ensureDefaultCamera(in scene: SCNScene, view: SCNView) {
        if let existing = view.pointOfView {
            Logger.log("Reusing existing pointOfView \(existing.name ?? "<unnamed>")", level: .debug, category: "ComputeDiagnostics")
            defaultCameraNode = existing
            cameraController.pointOfView = existing
            return
        }

        if let cachedNode = defaultCameraNode {
            if cachedNode.parent == nil {
                scene.rootNode.addChildNode(cachedNode)
            }
            view.pointOfView = cachedNode
            cameraController.pointOfView = cachedNode
            Logger.log("Reattached cached camera node.", level: .debug, category: "ComputeDiagnostics")
            return
        }

        let cameraNode = SCNNode()
        cameraNode.name = "VolumeDefaultCamera"
        let camera = SCNCamera()
        camera.usesOrthographicProjection = false
        camera.zNear = 0.001
        camera.zFar = 100.0
        camera.fieldOfView = 50.0
        cameraNode.camera = camera
        cameraNode.position = SCNVector3(0, 0, 2.5)
        scene.rootNode.addChildNode(cameraNode)

        defaultCameraNode = cameraNode
        view.pointOfView = cameraNode
        cameraController.pointOfView = cameraNode
        Logger.log("Created default camera node at position \(cameraNode.position)", level: .debug, category: "ComputeDiagnostics")
    }

    private func updateCameraPlacement() {
        guard let volume else { return }
        let sphere = volume.presentation.boundingSphere
        let centerWorld = volume.presentation.convertPosition(sphere.center, to: nil)
        let radius = max(Double(sphere.radius), 0.5)
        let distance = Float(radius * 2.5)
        let target = SCNVector3(centerWorld.x, centerWorld.y, centerWorld.z)

        if let cameraNode = cameraController.pointOfView ?? defaultCameraNode ?? sceneView?.pointOfView {
            cameraNode.position = SCNVector3(target.x, target.y, target.z + distance)
            cameraNode.look(at: target)
            cameraController.pointOfView = cameraNode
            Logger.log("Updated camera position \(cameraNode.position) looking at \(target)", level: .debug, category: "ComputeDiagnostics")
        }
        cameraController.target = target
    }

    private func logComputeIssue(_ keyPath: WritableKeyPath<ComputeRenderDiagnostics, Bool>, message: String) {
        if computeDiagnostics[keyPath: keyPath] { return }
        computeDiagnostics[keyPath: keyPath] = true
        Logger.log(message, level: .debug, category: "ComputeDiagnostics")
    }

    private func clearComputeIssue(_ keyPath: WritableKeyPath<ComputeRenderDiagnostics, Bool>) {
        computeDiagnostics[keyPath: keyPath] = false
    }

    func updateNodesForBackend() {
        switch renderBackend {
        case .fragmentShader:
            volume?.isHidden = false
            computePlaneNode?.isHidden = true
        case .compute:
            volume?.isHidden = true
            computePlaneNode?.isHidden = activeRenderMode == .mpr
        }
        requestMPRDisplayRefresh()
    }

    func markComputeDirty() {
        computeNeedsUpdate = true
    }

    private func renderMPRAndPresent() {
        mprMat?.renderIfNeeded()
        requestMPRDisplayRefresh()
    }

    private func requestMPRDisplayRefresh() {
        guard let view = sceneView else { return }
        if Thread.isMainThread {
            view.setNeedsDisplay()
        } else {
            DispatchQueue.main.async { view.setNeedsDisplay() }
        }
    }

    private func refreshHuWindowFromMaterial() {
        let uniforms = mat.snapshotUniforms()
        let previous = currentHuWindowRange
        currentHuWindowRange = (uniforms.voxelMinValue, uniforms.voxelMaxValue)
        mprMat?.setHU(min: currentHuWindowRange.min, max: currentHuWindowRange.max)
        renderMPRAndPresent()
        if currentHuWindowRange.min != previous.min || currentHuWindowRange.max != previous.max {
            Logger.log("HU window atualizado para [\(currentHuWindowRange.min), \(currentHuWindowRange.max)]",
                       level: .debug,
                       category: "VolumeCompute")
        }
        NotificationCenter.default.post(name: .sceneControllerHuWindowDidChange,
                                        object: self)
    }

    func clipBoundsSnapshot() -> ClipBounds {
        clipBounds.sanitized()
    }

    func updateClipBounds(xMin: Float,
                          xMax: Float,
                          yMin: Float,
                          yMax: Float,
                          zMin: Float,
                          zMax: Float) {
        applyClipBounds(ClipBounds(xMin: xMin,
                                   xMax: xMax,
                                   yMin: yMin,
                                   yMax: yMax,
                                   zMin: zMin,
                                   zMax: zMax))
    }

    func resetClipBounds() {
        let defaultBounds = ClipBounds()
        let identity = simd_quatf()

        let boxUnchanged = clipBounds == defaultBounds &&
            simd_length(clipBoxQuaternion.vector - identity.vector) <= 1e-4

        clipBounds = defaultBounds
        clipBoxQuaternion = identity
        resetClipPlanes()

        if boxUnchanged {
            return
        }

        propagateClipStateToRenderers()
        notifyClipBoundsObservers()
        markComputeDirty()
    }

    func alignClipBoxToView() {
        guard let volume,
              let cameraNode = sceneView?.pointOfView else { return }

        let volumeTransform = volume.simdWorldTransform
        let cameraTransform = cameraNode.simdWorldTransform
        let cameraInVolume = simd_mul(simd_inverse(volumeTransform), cameraTransform)
        let rotationColumns = (
            SIMD3<Float>(cameraInVolume.columns.0.x, cameraInVolume.columns.0.y, cameraInVolume.columns.0.z),
            SIMD3<Float>(cameraInVolume.columns.1.x, cameraInVolume.columns.1.y, cameraInVolume.columns.1.z),
            SIMD3<Float>(cameraInVolume.columns.2.x, cameraInVolume.columns.2.y, cameraInVolume.columns.2.z)
        )
        let rotation = simd_float3x3(columns: rotationColumns)
        let quaternion = simd_quatf(rotation)
        setClipBoxQuaternion(quaternion)
    }

    func updateClipBoxQuaternion(_ quaternion: simd_quatf) {
        setClipBoxQuaternion(quaternion)
    }

    func clipPlaneSnapshot() -> ClipPlaneState {
        ClipPlaneState(preset: clipPlanePreset,
                       offset: clipPlaneOffset,
                       hasCustomNormal: clipPlanePreset == .custom)
    }

    func setClipPlanePreset(_ preset: ClipPlanePreset) {
        guard clipPlanePreset != preset else { return }
        clipPlanePreset = preset

        if preset != .custom {
            customClipPlaneNormal = SIMD3<Float>(0, 0, 1)
        } else if simd_length(customClipPlaneNormal) < 1e-4 {
            customClipPlaneNormal = SIMD3<Float>(0, 0, 1)
        }

        propagateClipStateToRenderers()
        notifyClipPlaneObservers()
        markComputeDirty()
    }

    func setClipPlaneOffset(_ offset: Float) {
        let clamped = max(-0.5, min(offset, 0.5))
        guard abs(clamped - clipPlaneOffset) > 1e-4 else { return }
        clipPlaneOffset = clamped
        propagateClipStateToRenderers()
        notifyClipPlaneObservers()
        markComputeDirty()
    }

    func alignClipPlaneToView() {
        guard clipPlanePreset == .custom,
              let volume,
              let cameraNode = sceneView?.pointOfView else {
            return
        }

        let volumeTransform = volume.simdWorldTransform
        let cameraTransform = cameraNode.simdWorldTransform
        let cameraInVolume = simd_mul(simd_inverse(volumeTransform), cameraTransform)
        let forward = SIMD3<Float>(-cameraInVolume.columns.2.x,
                                   -cameraInVolume.columns.2.y,
                                   -cameraInVolume.columns.2.z)

        if simd_length(forward) > 1e-4 {
            customClipPlaneNormal = simd_normalize(forward)
            propagateClipStateToRenderers()
            notifyClipPlaneObservers()
            markComputeDirty()
        }
    }

    func resetClipPlanes() {
        clipPlanePreset = .off
        clipPlaneOffset = 0.0
        customClipPlaneNormal = SIMD3<Float>(0, 0, 1)
        propagateClipStateToRenderers()
        notifyClipPlaneObservers()
    }

    private func applyClipBounds(_ bounds: ClipBounds) {
        let sanitized = bounds.sanitized()
        guard sanitized != clipBounds else { return }
        clipBounds = sanitized
        propagateClipStateToRenderers()
        notifyClipBoundsObservers()
        markComputeDirty()
    }

    private func setClipBoxQuaternion(_ quaternion: simd_quatf) {
        let epsilon: Float = 1e-4
        let normalized: simd_quatf

        if simd_length(quaternion.vector) <= epsilon {
            normalized = simd_quatf()
        } else {
            normalized = simd_normalize(quaternion)
        }

        if simd_length(normalized.vector - clipBoxQuaternion.vector) <= epsilon {
            return
        }

        clipBoxQuaternion = normalized
        propagateClipStateToRenderers()
        markComputeDirty()
    }

    private func propagateClipStateToRenderers() {
        guard let computeRenderer else { return }
        let bounds = clipBounds.sanitized()
        computeRenderer.updateClipBounds(xMin: bounds.xMin,
                                         xMax: bounds.xMax,
                                         yMin: bounds.yMin,
                                         yMax: bounds.yMax,
                                         zMin: bounds.zMin,
                                         zMax: bounds.zMax)
        computeRenderer.updateClipBoxQuaternion(clipBoxQuaternion)
        let planes = clipPlaneVectors()
        computeRenderer.updateClipPlanes(planes.0, planes.1, planes.2)
    }

    private func clipPlaneVectors() -> (SIMD4<Float>, SIMD4<Float>, SIMD4<Float>) {
        let zero = SIMD4<Float>(repeating: 0)

        switch clipPlanePreset {
        case .off:
            return (zero, zero, zero)
        case .axial:
            return (makePlane(normal: SIMD3<Float>(0, 0, 1)), zero, zero)
        case .sagittal:
            return (makePlane(normal: SIMD3<Float>(1, 0, 0)), zero, zero)
        case .coronal:
            return (makePlane(normal: SIMD3<Float>(0, 1, 0)), zero, zero)
        case .custom:
            return (makePlane(normal: customClipPlaneNormal), zero, zero)
        }
    }

    private func makePlane(normal: SIMD3<Float>) -> SIMD4<Float> {
        var axis = normal
        if simd_length(axis) < 1e-4 {
            axis = SIMD3<Float>(0, 0, 1)
        }
        let normalized = simd_normalize(axis)
        return SIMD4<Float>(normalized.x, normalized.y, normalized.z, -clipPlaneOffset)
    }

    private func notifyClipPlaneObservers() {
        DispatchQueue.main.async {
            NotificationCenter.default.post(name: .sceneControllerClipPlaneDidChange, object: self)
        }
    }

    func setupChannelDefaults(device: MTLDevice, initialTexture: MTLTexture?) {
        guard channelStates.isEmpty else { return }

        let defaultPresets: [VolumeCubeMaterial.Preset?] = [.ct_arteries, nil, nil, nil]
        var states: [ChannelState] = []

        for index in 0..<defaultPresets.count {
            let preset = defaultPresets[index]
            var transferFunction = preset.flatMap { loadTransferFunction(for: $0) } ?? TransferFunction()
            var texture: MTLTexture?

            if index == 0 {
                if let matTransfer = mat.tf {
                    transferFunction = matTransfer
                }
                texture = initialTexture ?? transferFunction.get(device: device)
            } else {
                texture = transferFunction.get(device: device)
            }

            let gain: Float = index == 0 ? 1.0 : 0.0
            let toneCurve = ToneCurveModel()
            states.append(ChannelState(preset: preset,
                                       transferFunction: transferFunction,
                                       texture: texture,
                                       gain: gain,
                                       toneCurve: toneCurve,
                                       histogram: []))
        }

        channelStates = states
        applyChannelZeroTransferTexture()
        refreshToneBuffersForAllChannels()
        enqueueHistogramUpdate()
    }

    func loadTransferFunction(for preset: VolumeCubeMaterial.Preset) -> TransferFunction? {
        guard let url = Bundle.main.url(forResource: preset.rawValue, withExtension: "tf") else {
            Logger.log("Arquivo não encontrado: \(preset.rawValue).tf",
                       level: .error,
                       category: "TransferFunction")
            return nil
        }
        return TransferFunction.load(from: url)
    }

    func applyChannelZeroTransferTexture() {
        guard !channelStates.isEmpty else { return }
        if let texture = channelStates[0].texture {
            mat.setTransferFunctionTexture(texture)
        }
    }

    func propagateChannelStateToRenderers() {
        guard !channelStates.isEmpty else { return }
        let textures = channelStates.map { $0.texture }
        computeRenderer?.updateTransferTextures(textures)
        computeRenderer?.updateChannelIntensities(currentChannelIntensities())
        refreshToneBuffersForAllChannels()
    }

    func currentChannelIntensities() -> SIMD4<Float> {
        var gains = [Float](repeating: 0, count: 4)
        for (index, state) in channelStates.enumerated() where index < 4 {
            gains[index] = state.gain
        }
        return SIMD4<Float>(gains[0], gains[1], gains[2], gains[3])
    }

    func updateChannelPreset(_ preset: VolumeCubeMaterial.Preset?,
                             for channel: Int,
                             resetGain: Bool = false,
                             resetShift: Bool = false) {
        guard channelStates.indices.contains(channel) else { return }

        var state = channelStates[channel]
        state.preset = preset

        if let preset {
            if var transfer = loadTransferFunction(for: preset) {
                if resetShift {
                    transfer.shift = 0
                } else {
                    transfer.shift = state.transferFunction.shift
                }
                state.transferFunction = transfer
                if let texture = transfer.get(device: device) {
                    state.texture = texture
                } else {
                    Logger.log("Falha ao gerar textura de TF para canal \(channel)",
                               level: .error,
                               category: "TransferFunction")
                }
                if resetGain && state.gain <= 0.0001 {
                    state.gain = 1.0
                }
            }
        } else {
            var transfer = TransferFunction()
            transfer.shift = 0
            state.transferFunction = transfer
            if let texture = transfer.get(device: device) {
                state.texture = texture
            }
            if resetGain {
                state.gain = 0.0
            }
        }

        channelStates[channel] = state

        if channel == 0 {
            mat.tf = state.transferFunction
            applyChannelZeroTransferTexture()
            syncMPRTransferFunction()
        }

        propagateChannelStateToRenderers()
        markComputeDirty()
    }

    func updateChannelShift(for channel: Int, shift: Float) {
        guard channelStates.indices.contains(channel) else { return }
        channelStates[channel].transferFunction.shift = shift

        if channel == 0 {
            mat.tf = channelStates[channel].transferFunction
            mat.setShift(device: device, shift: shift)
            channelStates[channel].texture = mat.currentTransferFunctionTexture()
            if channelStates[channel].texture == nil,
               let fallbackTexture = channelStates[channel].transferFunction.get(device: device) {
                channelStates[channel].texture = fallbackTexture
            }
            syncMPRTransferFunction()
        } else {
            if let texture = channelStates[channel].transferFunction.get(device: device) {
                channelStates[channel].texture = texture
            } else {
                Logger.log("Falha ao atualizar textura do canal \(channel) após shift",
                           level: .warn,
                           category: "TransferFunction")
            }
        }

        propagateChannelStateToRenderers()
        markComputeDirty()
    }

    func renderComputeFrame(using renderer: SCNSceneRenderer) {
        guard renderBackend == .compute else { return }
        guard activeRenderMode != .mpr else { return }
        guard !isRenderingCompute else { return }

        guard let computeRenderer else {
            logComputeIssue(\.missingRenderer, message: "Compute renderer unavailable; skipping compute frame.")
            return
        }
        clearComputeIssue(\.missingRenderer)

        guard let volumeTexture = mat.currentVolumeTexture() else {
            logComputeIssue(\.missingVolume, message: "Volume texture unavailable; compute frame skipped.")
            return
        }
        clearComputeIssue(\.missingVolume)

        guard sceneView != nil else {
            logComputeIssue(\.missingView, message: "SceneView not ready; compute frame skipped.")
            return
        }
        clearComputeIssue(\.missingView)

        guard let cameraParams = makeCameraParameters(from: renderer) else {
            logComputeIssue(\.missingCamera, message: "Failed to build camera parameters; compute frame skipped.")
            return
        }
        clearComputeIssue(\.missingCamera)

        let signature = CameraSignature(modelMatrix: cameraParams.modelMatrix,
                                         inverseViewProjectionMatrix: cameraParams.inverseViewProjectionMatrix)
        if !computeNeedsUpdate,
           let previous = lastCameraSignature,
           previous.isApproximatelyEqual(to: signature) {
            return
        }
        lastCameraSignature = signature

        let viewport = renderer.currentViewport
        let viewportSize = CGSize(width: viewport.width, height: viewport.height)
        guard viewportSize.width > 0, viewportSize.height > 0 else {
            logComputeIssue(\.zeroViewport, message: "Renderer viewport is zero-sized; skipping compute frame.")
            return
        }
        clearComputeIssue(\.zeroViewport)
        updateComputePlaneSizeIfNeeded(viewportSize: viewportSize)

        logCentralRayDiagnostics(camera: cameraParams,
                                  viewportSize: viewportSize,
                                  renderer: renderer)

        var uniforms = mat.snapshotUniforms()
        if uniforms.renderingQuality <= 0 {
            uniforms.renderingQuality = 256
        }

        isRenderingCompute = true
        defer {
            isRenderingCompute = false
        }

        let outputSize = viewportSize
        if AppConfig.IS_DEBUG_MODE {
            logCentralRayDiagnostics(camera: cameraParams, outputSize: outputSize)
        }
        guard let texture = computeRenderer.render(volume: volumeTexture,
                                                   uniforms: uniforms,
                                                   camera: cameraParams,
                                                   outputSize: outputSize) else {
            logComputeIssue(\.renderFailure, message: "Compute renderer returned nil texture for output \(Int(outputSize.width))x\(Int(outputSize.height)).")
            return
        }
        clearComputeIssue(\.renderFailure)

        lastComputeTexture = texture

        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            Logger.log("Publishing compute texture \(texture.label ?? "<unnamed>") size=\(texture.width)x\(texture.height)", level: .debug, category: "ComputeDiagnostics")
            self.computeDiffuseProperty?.contents = texture
            self.computeEmissionProperty?.contents = texture
            self.computePlaneNode.map { node in
                Logger.log("Compute plane hidden? \(node.isHidden)", level: .debug, category: "ComputeDiagnostics")
            }
        }

        if AppConfig.IS_DEBUG_MODE,
           computeDebugSampleCount < 3,
           let data = computeRenderer.readback(texture: texture) {
            let pixelIndex = Int(texture.width / 2) + Int(texture.height / 2) * Int(texture.width)
            let offset = pixelIndex * 4
            if offset + 3 < data.count {
                let b = Float(data[offset]) / 255.0
                let g = Float(data[offset + 1]) / 255.0
                let r = Float(data[offset + 2]) / 255.0
                let a = Float(data[offset + 3]) / 255.0
                Logger.log(String(format: "Compute sample BGRA (%.3f, %.3f, %.3f, %.3f)", b, g, r, a),
                           level: .debug,
                           category: "ComputeDiagnostics")
            } else {
                Logger.log("Failed to decode compute texture sample bytes.", level: .debug, category: "ComputeDiagnostics")
            }
            computeDebugSampleCount += 1
        }

        computeNeedsUpdate = false
    }

    private func logCentralRayDiagnostics(camera: VolumeCameraParameters,
                                          viewportSize: CGSize,
                                          renderer: SCNSceneRenderer) {
        guard AppConfig.IS_DEBUG_MODE else { return }
        guard computeDebugRayLogCount < 8 else { return }
        let width = max(1, Int(ceil(viewportSize.width)))
        let height = max(1, Int(ceil(viewportSize.height)))
        guard width > 0, height > 0 else { return }

        let sampleX = width / 2
        let sampleY = height / 2
        let flippedY = max(height - 1 - sampleY, 0)

        let ndcX = (Float(sampleX) + 0.5) / Float(width) * 2.0 - 1.0
        let ndcY = (Float(flippedY) + 0.5) / Float(height) * 2.0 - 1.0
        let clipNear = SIMD4<Float>(ndcX, ndcY, 0.0, 1.0)
        let clipFar = SIMD4<Float>(ndcX, ndcY, 1.0, 1.0)

        var worldNear = camera.inverseViewProjectionMatrix * clipNear
        var worldFar = camera.inverseViewProjectionMatrix * clipFar
        if abs(worldNear.w) > 1e-6 {
            worldNear /= worldNear.w
        }
        if abs(worldFar.w) > 1e-6 {
            worldFar /= worldFar.w
        }

        let localFar4 = camera.inverseModelMatrix * worldFar
        let localFarW = abs(localFar4.w) > 1e-6 ? localFar4.w : 1.0
        let pixelLocal01 = localFar4.xyz / localFarW + SIMD3<Float>(repeating: 0.5)
        let cameraLocal01 = camera.cameraPositionLocal + SIMD3<Float>(repeating: 0.5)

        let rayDir = VolumeComputeRenderer.computeRayDirection(cameraLocal01: cameraLocal01,
                                                                pixelLocal01: pixelLocal01)
        let hasDirection = simd_length(rayDir) > 1e-5

        var tEnter: Float = .nan
        var tExit: Float = .nan
        var hitsVolume = false

        if hasDirection {
            let intersection = VolumeComputeRenderer.rayBoxIntersection(rayOrigin: cameraLocal01,
                                                                        rayDirection: rayDir,
                                                                        boxMin: SIMD3<Float>(repeating: 0),
                                                                        boxMax: SIMD3<Float>(repeating: 1))
            tEnter = max(intersection.x, 0.0)
            tExit = intersection.y
            hitsVolume = tExit > tEnter
        }

        let cameraLocal = camera.cameraPositionLocal
        let insideVolume = (0.0...1.0).contains(cameraLocal01.x) &&
                           (0.0...1.0).contains(cameraLocal01.y) &&
                           (0.0...1.0).contains(cameraLocal01.z)

        let message = String(format: "Central ray gid=(%d,%d) cameraLocal=(%.4f, %.4f, %.4f) cameraLocal01=(%.4f, %.4f, %.4f) pixelLocal01=(%.4f, %.4f, %.4f) rayDir=(%.4f, %.4f, %.4f) tEnter=%.4f tExit=%.4f hitsVolume=%@ inside=%@",
                             sampleX, sampleY,
                             cameraLocal.x, cameraLocal.y, cameraLocal.z,
                             cameraLocal01.x, cameraLocal01.y, cameraLocal01.z,
                             pixelLocal01.x, pixelLocal01.y, pixelLocal01.z,
                             rayDir.x, rayDir.y, rayDir.z,
                             tEnter, tExit,
                             hitsVolume ? "true" : "false",
                             insideVolume ? "true" : "false")

        Logger.log(message,
                   level: hitsVolume ? .debug : .warn,
                   category: "ComputeDiagnostics")

        if !hitsVolume {
            if let cameraNode = renderer.pointOfView ?? cameraController.pointOfView ?? defaultCameraNode ?? sceneView?.pointOfView {
                let worldPosition = cameraNode.simdWorldPosition
                let worldOrientation = cameraNode.simdWorldOrientation
                let volumeOrientation = volume?.simdWorldOrientation ?? simd_quatf()
                let relativeOrientation = simd_mul(simd_inverse(volumeOrientation), worldOrientation)
                let defaultCameraActive = (cameraNode === defaultCameraNode)
                let orientationMessage = String(format: "Camera miss debug: worldPos=(%.4f, %.4f, %.4f) localPosition=(%.4f, %.4f, %.4f) relativeQuat=(%.4f, %.4f, %.4f, %.4f) usesDefaultCamera=%@",
                                                worldPosition.x, worldPosition.y, worldPosition.z,
                                                cameraLocal.x, cameraLocal.y, cameraLocal.z,
                                                relativeOrientation.vector.x, relativeOrientation.vector.y, relativeOrientation.vector.z, relativeOrientation.vector.w,
                                                defaultCameraActive ? "true" : "false")
                Logger.log(orientationMessage,
                           level: .warn,
                           category: "ComputeDiagnostics")
            }

            computeRenderer?.refreshCameraUniforms(camera)
            propagateClipStateToRenderers()
        }

        computeDebugRayLogCount += 1
    }

    func makeCameraParameters(from renderer: SCNSceneRenderer) -> VolumeCameraParameters? {
        guard let volumeNode = volume else {
            Logger.log("Camera parameters unavailable: missing volume node.", level: .debug, category: "ComputeDiagnostics")
            return nil
        }

        let cameraNode = renderer.pointOfView ?? cameraController.pointOfView ?? defaultCameraNode ?? sceneView?.pointOfView
        guard let cameraNode, let camera = cameraNode.camera else {
            Logger.log("Camera parameters unavailable: pointOfView is nil.", level: .debug, category: "ComputeDiagnostics")
            return nil
        }

        let modelMatrix = volumeNode.simdWorldTransform
        let inverseModelMatrix = simd_inverse(modelMatrix)

        let projectionMatrix = simd_float4x4(camera.projectionTransform)
        let viewMatrix = simd_inverse(cameraNode.simdWorldTransform)
        let viewProjection = projectionMatrix * viewMatrix
        let inverseViewProjection = simd_inverse(viewProjection)

        let cameraWorldPosition = cameraNode.simdWorldPosition
        let cameraLocal4 = inverseModelMatrix * SIMD4<Float>(cameraWorldPosition, 1.0)
        let cameraLocal = cameraLocal4.xyz / max(cameraLocal4.w, 1e-5)

        return VolumeCameraParameters(modelMatrix: modelMatrix,
                                      inverseModelMatrix: inverseModelMatrix,
                                      inverseViewProjectionMatrix: inverseViewProjection,
                                      cameraPositionLocal: cameraLocal)
    }

    private func logCentralRayDiagnostics(camera: VolumeCameraParameters, outputSize: CGSize) {
        guard AppConfig.IS_DEBUG_MODE else { return }

        let width = Int(outputSize.width)
        let height = Int(outputSize.height)
        guard width > 0, height > 0 else { return }

        let pixelX = max(width / 2, 0)
        let pixelY = max(height / 2, 0)

        let ndcX = ((Float(pixelX) + 0.5) / Float(width)) * 2.0 - 1.0
        let ndcY = ((Float(height - 1 - pixelY) + 0.5) / Float(height)) * 2.0 - 1.0

        let clipFar = SIMD4<Float>(ndcX, ndcY, 1.0, 1.0)

        var worldFar = camera.inverseViewProjectionMatrix * clipFar
        worldFar /= max(worldFar.w, 1.0e-6)

        let localFar4 = camera.inverseModelMatrix * worldFar
        let localFar = SIMD3<Float>(localFar4.x, localFar4.y, localFar4.z) / max(localFar4.w, 1.0e-6) + SIMD3<Float>(repeating: 0.5)

        let cameraLocal = camera.cameraPositionLocal
        let cameraLocal01 = cameraLocal + SIMD3<Float>(repeating: 0.5)
        let rayVector = localFar - cameraLocal01
        let rayDir = simd_length(rayVector) > 0.0 ? simd_normalize(rayVector) : SIMD3<Float>(repeating: 0.0)

        let intersection = computeAabbIntersection(origin: cameraLocal01,
                                                   direction: rayDir,
                                                   boxMin: SIMD3<Float>(repeating: 0.0),
                                                   boxMax: SIMD3<Float>(repeating: 1.0))
        let hitsVolume = intersection.exit > intersection.enter

        let message = String(format: "Central ray cameraLocal=%@ cameraLocal01=%@ dir=%@ tEnter=%.5f tExit=%.5f intersects=%@",
                             formatVector(cameraLocal),
                             formatVector(cameraLocal01),
                             formatVector(rayDir),
                             Double(intersection.enter),
                             Double(intersection.exit),
                             hitsVolume ? "yes" : "no")

        Logger.log(message, level: .debug, category: "ComputeDiagnostics")
        if !hitsVolume {
            Logger.log("Central ray miss detected; verify camera placement and normalization.",
                       level: .warn,
                       category: "ComputeDiagnostics")
        }
    }

    private func computeAabbIntersection(origin: SIMD3<Float>,
                                         direction: SIMD3<Float>,
                                         boxMin: SIMD3<Float>,
                                         boxMax: SIMD3<Float>) -> (enter: Float, exit: Float) {
        let epsilon: Float = 1.0e-6
        let safeDirection = SIMD3<Float>(
            abs(direction.x) < epsilon ? (direction.x >= 0 ? epsilon : -epsilon) : direction.x,
            abs(direction.y) < epsilon ? (direction.y >= 0 ? epsilon : -epsilon) : direction.y,
            abs(direction.z) < epsilon ? (direction.z >= 0 ? epsilon : -epsilon) : direction.z
        )

        let tMin = (boxMin - origin) / safeDirection
        let tMax = (boxMax - origin) / safeDirection

        let nearComponents = simd_min(tMin, tMax)
        let farComponents = simd_max(tMin, tMax)

        let sanitizedNear = SIMD3<Float>(
            nearComponents.x.isFinite ? nearComponents.x : -Float.greatestFiniteMagnitude,
            nearComponents.y.isFinite ? nearComponents.y : -Float.greatestFiniteMagnitude,
            nearComponents.z.isFinite ? nearComponents.z : -Float.greatestFiniteMagnitude
        )
        let sanitizedFar = SIMD3<Float>(
            farComponents.x.isFinite ? farComponents.x : Float.greatestFiniteMagnitude,
            farComponents.y.isFinite ? farComponents.y : Float.greatestFiniteMagnitude,
            farComponents.z.isFinite ? farComponents.z : Float.greatestFiniteMagnitude
        )

        let tEnter = max(max(sanitizedNear.x, sanitizedNear.y), sanitizedNear.z)
        let tExit = min(min(sanitizedFar.x, sanitizedFar.y), sanitizedFar.z)

        return (tEnter, tExit)
    }

    private func formatVector(_ vector: SIMD3<Float>) -> String {
        String(format: "(%.5f, %.5f, %.5f)", Double(vector.x), Double(vector.y), Double(vector.z))
    }

    func rebindVolumeNodeForVR() {
        guard let root = root, let volume = volume, let mat = mat else { return }

        if volume.parent == nil {
            root.addChildNode(volume)
        }

        if volume.geometry == nil {
            volume.geometry = SCNBox(width: 1, height: 1, length: 1, chamferRadius: 0)
        }

        if volume.geometry?.materials.count != 1 || volume.geometry?.materials.first !== mat {
            volume.geometry?.materials = [mat]
        }

        volume.simdTransform = mat.transform
        volume.isHidden = false
    }

    func applyImportedDataset(_ dataset: VolumeDataset) {
        Logger.log("Aplicando dataset importado dims=\(dataset.dimensions) spacing=\(dataset.spacing) range=\(dataset.intensityRange)",
                   level: .debug,
                   category: "SceneViewController")
        computeDebugSampleCount = 0
        mat.setDataset(device: device, dataset: dataset) { [weak self] in
            guard let self else { return }
            self.refreshHuWindowFromMaterial()
            self.updateMetadata(from: dataset)
            self.postVolumeUpdate(usingMprPart: .dicom, dataset: dataset)
            if let defaultPreset = WindowLevelPresetLibrary.ct.first(where: { $0.name == "Brain" }) ?? WindowLevelPresetLibrary.ct.first {
                let minHU = Int32(defaultPreset.minValue.rounded())
                let maxHU = Int32(defaultPreset.maxValue.rounded())
                self.setHuWindow(minHU: minHU, maxHU: maxHU)
            }
        }
    }

    func postVolumeUpdate(usingMprPart part: VolumeCubeMaterial.BodyPart, dataset: VolumeDataset? = nil) {
        volume.simdTransform = mat.transform
        updateCameraPlacement()
        syncMPRTransformWithVolume()
        resetClipBounds()
        propagateClipStateToRenderers()
        mat.setShift(device: device, shift: 0)

        let finalize: () -> Void = { [weak self] in
            guard let self else { return }
            self.syncMPRTransferFunction()

            if let dim = self.mprMat?.dimension {
                let mid = Int(dim.z / 2)
                self.mprMat?.setAxial(slice: mid)
                self.mprMat?.setSlab(thicknessInVoxels: 0, steps: 1)
            }

            self.setRenderMode(self.activeRenderMode)
            self.renderMPRAndPresent()
            self.notifyMPRDatasetObservers()
            self.refreshToneBuffersForAllChannels()
            self.enqueueHistogramUpdate()
            self.markComputeDirty()
        }

        guard part != .none else {
            finalize()
            return
        }

        if let dataset, let mprMat {
            mprMat.setDataset(device: device, dataset: dataset) {
                finalize()
            }
        } else if let mprMat {
            mprMat.setPart(device: device, part: part) {
                finalize()
            }
        } else {
            finalize()
        }
    }

    private func updateMetadata(from dataset: VolumeDataset) {
        volumeMetadata = VolumeMetadata(dimensions: dataset.dimensions,
                                        spacing: dataset.spacing,
                                        origin: dataset.origin,
                                        orientation: dataset.orientation)
        notifyMetadataObservers()
    }
}

// MARK: - Dataset Snapshot Helpers
extension SceneViewController {
    func currentVolumeTexture() -> MTLTexture? {
        mat.currentVolumeTexture()
    }

    func currentTFTexture() -> MTLTexture? {
        mat.currentTransferFunctionTexture()
    }

    func currentComputeOutputTexture() -> MTLTexture? {
        lastComputeTexture
    }

    func currentDatasetMeta() -> DatasetMeta? {
        let meta = mat.datasetMeta
        return DatasetMeta(dimension: meta.dimension, resolution: meta.resolution)
    }

    func currentVolumeMetadata() -> VolumeMetadata? {
        volumeMetadata
    }

    func currentHuWindow() -> (min: Int32, max: Int32) {
        currentHuWindowRange
    }

    func currentRenderingQuality() -> Float {
        lastStep
    }

    func currentHuWindowSnapshot() -> (min: Int32, max: Int32) {
        currentHuWindowRange
    }

    func datasetHuBounds() -> (min: Int32, max: Int32)? {
        guard let mat else { return nil }
        let uniforms = mat.snapshotUniforms()
        return (uniforms.datasetMinValue, uniforms.datasetMaxValue)
    }
}

// MARK: - Notifications
private extension SceneViewController {
    func notifyMPRDatasetObservers() {
        DispatchQueue.main.async {
            NotificationCenter.default.post(name: .sceneControllerDatasetDidChange, object: self)
        }
    }

    func notifyMPRTransferFunctionObservers() {
        DispatchQueue.main.async {
            NotificationCenter.default.post(name: .sceneControllerTransferFunctionDidChange, object: self)
        }
    }

    func notifyToneCurveObservers(channel: Int) {
        DispatchQueue.main.async {
            NotificationCenter.default.post(name: .sceneControllerToneCurveDidChange,
                                            object: self,
                                            userInfo: ["channel": channel])
        }
    }

    func notifyHistogramObservers() {
        DispatchQueue.main.async {
            NotificationCenter.default.post(name: .sceneControllerHistogramDidChange, object: self)
        }
    }

    func notifyClipBoundsObservers() {
        DispatchQueue.main.async {
            NotificationCenter.default.post(name: .sceneControllerClipBoundsDidChange, object: self)
        }
    }

    func notifyMetadataObservers() {
        DispatchQueue.main.async {
            NotificationCenter.default.post(name: .sceneControllerMetadataDidChange, object: self)
        }
    }
}

extension Notification.Name {
    static let sceneControllerDatasetDidChange = Notification.Name("SceneViewControllerDatasetDidChange")
    static let sceneControllerTransferFunctionDidChange = Notification.Name("SceneViewControllerTransferFunctionDidChange")
    static let sceneControllerToneCurveDidChange = Notification.Name("SceneViewControllerToneCurveDidChange")
    static let sceneControllerHistogramDidChange = Notification.Name("SceneViewControllerHistogramDidChange")
    static let sceneControllerClipBoundsDidChange = Notification.Name("SceneViewControllerClipBoundsDidChange")
    static let sceneControllerClipPlaneDidChange = Notification.Name("SceneViewControllerClipPlaneDidChange")
    static let sceneControllerMetadataDidChange = Notification.Name("SceneViewControllerMetadataDidChange")
    static let sceneControllerHuWindowDidChange = Notification.Name("SceneViewControllerHuWindowDidChange")
}

private extension simd_float4x4 {
    init(_ m: SCNMatrix4) {
        self.init(columns: (
            SIMD4<Float>(m.m11, m.m12, m.m13, m.m14),
            SIMD4<Float>(m.m21, m.m22, m.m23, m.m24),
            SIMD4<Float>(m.m31, m.m32, m.m33, m.m34),
            SIMD4<Float>(m.m41, m.m42, m.m43, m.m44)
        ))
    }

    func isApproximatelyEqual(to other: simd_float4x4, epsilon: Float) -> Bool {
        let difference = self - other
        let max0 = simd_reduce_max(abs(difference.columns.0))
        let max1 = simd_reduce_max(abs(difference.columns.1))
        let max2 = simd_reduce_max(abs(difference.columns.2))
        let max3 = simd_reduce_max(abs(difference.columns.3))
        let maxComponent = Swift.max(Swift.max(max0, max1), Swift.max(max2, max3))
        return maxComponent < epsilon
    }
}
