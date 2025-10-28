//  TriPlanarMPR.swift
//  Isis DICOM Viewer
//
//  Tri‑Planar MPR com linhas ortogonais arrastáveis e rotação oblíqua
//  - 3 viewports 2D (axial, coronal, sagittal) com SCNView ortográfico
//  - Reutiliza a textura 3D e a TF já carregadas pelo VR
//  - Linhas centrais arrastáveis movem os planos; rotação 2‑toques gira a triade
//
//  Thales Matheus Mendonça Santos - September 2025

import SwiftUI
import SceneKit
import Metal
import simd
import Combine
import UIKit

// MARK: - Enum de planos
enum TriPlane: CaseIterable {
    case axial, coronal, sagittal
    var title: String {
        switch self {
        case .axial: return "Axial"
        case .coronal: return "Coronal"
        case .sagittal: return "Sagittal"
        }
    }
}

// MARK: - Estado compartilhado (triade ortonormal + ponto de interseção)
final class TriMPRState: ObservableObject {
    // Base {x',y',z'} (colunas) em coords normalizadas do volume [0,1]^3
    @Published var R: simd_float3x3 = matrix_identity_float3x3
    // Ponto de interseção (crosshair) em [0,1]^3
    @Published var cross: SIMD3<Float> = SIMD3(0.5, 0.5, 0.5)
    // Dimensão do volume em voxels (convertido para Float para cálculos de velocidade)
    @Published private(set) var volumeDimensions: SIMD3<Float> = SIMD3(repeating: 1)

    // clamps
    private let eps: Float = 1e-5

    // Accessor for matrix column by index (since `columns` is a tuple, not subscriptable)
    private func baseColumn(_ i: Int) -> SIMD3<Float> {
        switch i {
        case 0: return R.columns.0
        case 1: return R.columns.1
        default: return R.columns.2
        }
    }

    // Rotação incremental ao redor de um eixo da própria base (0=x', 1=y', 2=z').
    func rotate(aroundBaseAxis i: Int, deltaRadians: Float) {
        let axis = normalize(baseColumn(i))
        let c = cos(deltaRadians), s = sin(deltaRadians)
        let C = 1 - c
        let x = axis.x, y = axis.y, z = axis.z
        let rot = simd_float3x3(
            SIMD3<Float>( c + x*x*C,     x*y*C - z*s,  x*z*C + y*s),
            SIMD3<Float>( y*x*C + z*s,   c + y*y*C,    y*z*C - x*s),
            SIMD3<Float>( z*x*C - y*s,   z*y*C + x*s,  c + z*z*C)
        )
        R = rot * R
    }

    // Tradução do cross ao longo de um eixo da base (0=x',1=y',2=z')
    func translate(alongBaseAxis i: Int, deltaNormalized: Float) {
        cross = simd_clamp(cross + deltaNormalized * baseColumn(i),
                           SIMD3<Float>(repeating: 0 + eps),
                           SIMD3<Float>(repeating: 1 - eps))
    }

    func updateVolumeDimensions(_ dims: int3) {
        volumeDimensions = SIMD3<Float>(Float(dims.x), Float(dims.y), Float(dims.z))
    }

    /// Converte a variação do gesto (já normalizada em relação ao tamanho da view) em delta seguro
    /// para `translate`, respeitando os limites remanescentes antes de alcançar [0,1]^3.
    func normalizedGestureDelta(for rawDelta: Float, axisIndex: Int) -> Float {
        guard rawDelta != 0 else { return 0 }

        let axis = baseColumn(axisIndex)
        var delta = rawDelta

        let minBounds = SIMD3<Float>(repeating: eps)
        let maxBounds = SIMD3<Float>(repeating: 1 - eps)
        var positiveLimit = Float.greatestFiniteMagnitude
        var negativeLimit = Float.greatestFiniteMagnitude

        for idx in 0..<3 {
            let component = axis[idx]
            if abs(component) <= Float.ulpOfOne { continue }

            if component > 0 {
                positiveLimit = min(positiveLimit, (maxBounds[idx] - cross[idx]) / component)
                negativeLimit = min(negativeLimit, (cross[idx] - minBounds[idx]) / component)
            } else {
                let absComponent = -component
                positiveLimit = min(positiveLimit, (cross[idx] - minBounds[idx]) / absComponent)
                negativeLimit = min(negativeLimit, (maxBounds[idx] - cross[idx]) / absComponent)
            }
        }

        if delta > 0 {
            delta = min(delta, positiveLimit)
        } else if delta < 0 {
            delta = -min(-delta, negativeLimit)
        }

        return delta
    }
}

private struct PlanePlacement {
    let origin: SIMD3<Float>
    let axisU: SIMD3<Float>
    let axisV: SIMD3<Float>
    let flipVertical: Bool
}

private func orthonormalAxes(normal: SIMD3<Float>,
                             preferred: SIMD3<Float>) -> (SIMD3<Float>, SIMD3<Float>) {
    let n = simd_normalize(normal)
    var axisU = preferred - simd_dot(preferred, n) * n
    if simd_length_squared(axisU) < 1e-6 {
        let alternative = abs(n.x) < 0.9 ? SIMD3<Float>(1, 0, 0) : SIMD3<Float>(0, 1, 0)
        axisU = alternative - simd_dot(alternative, n) * n
    }
    axisU = simd_normalize(axisU)
    var axisV = simd_cross(n, axisU)
    let lengthV = simd_length(axisV)
    if lengthV < 1e-6 {
        axisV = simd_cross(axisU, n)
    }
    axisV = simd_normalize(axisV)
    return (axisU, axisV)
}

private func computePlanePlacement(for plane: TriPlane,
                                   basis: simd_float3x3,
                                   cross: SIMD3<Float>) -> PlanePlacement {
    let center = SIMD3<Float>(repeating: 0.5)

    let normal: SIMD3<Float>
    let reference: SIMD3<Float>
    let flipVertical: Bool

    switch plane {
    case .axial:
        normal = basis.columns.2
        reference = SIMD3<Float>(1, 0, 0)
        flipVertical = false
    case .coronal:
        normal = basis.columns.1
        reference = SIMD3<Float>(1, 0, 0)
        flipVertical = false
    case .sagittal:
        normal = basis.columns.0
        reference = SIMD3<Float>(0, 1, 0)
        flipVertical = true
    }

    var (axisU, axisV) = orthonormalAxes(normal: normal, preferred: reference)
    if flipVertical {
        axisV *= -1
    }

    let adjustedOrigin = cross - 0.5 * axisU - 0.5 * axisV

    return PlanePlacement(origin: adjustedOrigin,
                          axisU: axisU,
                          axisV: axisV,
                          flipVertical: flipVertical)
}

// MARK: - Um viewport (SCNView) para um plano
final class MPRViewportController: NSObject, SCNSceneRendererDelegate, UIGestureRecognizerDelegate {
    let planeKind: TriPlane
    private let scnView = SCNView()
    private let scene = SCNScene()
    private let cameraNode = SCNNode()
    private let node = SCNNode(geometry: SCNPlane(width: 1, height: 1))
    private let matMPR: MPRPlaneMaterial
    private var device: MTLDevice
    private var zoomFactor: Float = 1.0
    private let minZoom: Float = 0.25
    private let maxZoom: Float = 4.0
    var pointerScrollHandler: ((Float) -> Void)?

    struct DatasetCache {
        let volumeTexture: MTLTexture
        let transferTexture: MTLTexture?
        let dimension: int3
        let resolution: float3
    }

    struct OrientationCache {
        let basis: simd_float3x3
        let cross: SIMD3<Float>
    }

    // Shared caches allow freshly created viewports to restore the last dataset and basis.
    private static var sharedDatasetCache: DatasetCache?
    private static var sharedOrientationCache: OrientationCache?

    init(plane: TriPlane, device: MTLDevice) {
        self.planeKind = plane
        self.device = device
        let features = FeatureFlags.evaluate(for: device)
        self.matMPR = MPRPlaneMaterial(device: device, featureFlags: features)
        super.init()
        setupView()
        reapplyCachedDatasetIfAvailable()
    }

    var view: SCNView { scnView }

    private func setupView() {
        scnView.scene = scene
        scnView.backgroundColor = .black
        scnView.allowsCameraControl = false
        scnView.autoenablesDefaultLighting = true
        scnView.isPlaying = true
        scnView.rendersContinuously = true
        scnView.pointOfView = cameraNode
        scnView.delegate = self
        scene.isPaused = false
        scnView.addGestureRecognizer(pointerPanRecognizer)

        // Câmera ortográfica 2D
        cameraNode.camera = SCNCamera()
        cameraNode.camera?.usesOrthographicProjection = true
        cameraNode.camera?.orthographicScale = 1.0      // plano 1×1 em tela
        cameraNode.position = SCNVector3(0, 0, 2)
        cameraNode.camera?.zNear = 0.001
        cameraNode.camera?.zFar  = 10
        scene.rootNode.addChildNode(cameraNode)

        node.geometry?.firstMaterial = matMPR
        node.eulerAngles = SCNVector3Zero
        node.position = SCNVector3Zero
        scene.rootNode.addChildNode(node)

        updateCameraZoom()
    }

    // Injeta dataset (textura 3D, TF, meta)
    func setDataset(volumeTex: MTLTexture, tfTex: MTLTexture?, dimension: int3, resolution: float3) {
        MPRViewportController.sharedDatasetCache = DatasetCache(volumeTexture: volumeTex,
                                                                transferTexture: tfTex,
                                                                dimension: dimension,
                                                                resolution: resolution)
        matMPR.setValue(SCNMaterialProperty(contents: volumeTex), forKey: "volume")
        if let tf = tfTex { matMPR.setTransferFunction(tf) }
        matMPR.setDataset(dimension: dimension, resolution: resolution)
        // MPR fino por padrão
        matMPR.setSlab(thicknessInVoxels: 0, steps: 1)
        renderAndPresent()
    }

    func setTransferFunction(_ tfTex: MTLTexture) {
        if let cached = MPRViewportController.sharedDatasetCache {
            MPRViewportController.sharedDatasetCache = DatasetCache(volumeTexture: cached.volumeTexture,
                                                                    transferTexture: tfTex,
                                                                    dimension: cached.dimension,
                                                                    resolution: cached.resolution)
        }
        matMPR.setTransferFunction(tfTex)
        renderAndPresent()
    }

    func setBlend(_ mode: MPRPlaneMaterial.BlendMode) {
        matMPR.setBlend(mode)
        renderAndPresent()
    }

    func setHU(min: Int32, max: Int32) {
        matMPR.setHU(min: min, max: max)
        renderAndPresent()
    }

    func setUseTF(_ enabled: Bool) {
        matMPR.setUseTF(enabled)
        renderAndPresent()
    }

    func setOverlayMask(_ texture: MTLTexture?) {
        matMPR.setOverlayMask(texture)
        renderAndPresent()
    }

    func setOverlayOpacity(_ value: Float) {
        matMPR.setOverlayOpacity(value)
        renderAndPresent()
    }

    func setOverlayColor(_ color: SIMD3<Float>) {
        matMPR.setOverlayColor(color)
        renderAndPresent()
    }

    func setOverlayChannel(_ channel: Int32) {
        matMPR.setOverlayChannel(channel)
        renderAndPresent()
    }

    func setZoom(_ zoom: Float) {
        let clamped = max(minZoom, min(maxZoom, zoom))
        guard abs(clamped - zoomFactor) > Float.ulpOfOne else { return }
        zoomFactor = clamped
        updateCameraZoom()
        renderAndPresent()
    }

    func setSlab(thicknessInVoxels: Int, steps: Int) {
        matMPR.setSlab(thicknessInVoxels: thicknessInVoxels, steps: steps)
        renderAndPresent()
    }

    // Aplica obliquidade e posição do plano a partir de base R e cross
    func apply(R: simd_float3x3, cross: SIMD3<Float>) {
        MPRViewportController.sharedOrientationCache = OrientationCache(basis: R, cross: cross)
        let placement = computePlanePlacement(for: planeKind, basis: R, cross: cross)
        matMPR.setOblique(origin: float3(placement.origin),
                          axisU: float3(placement.axisU),
                          axisV: float3(placement.axisV))
        // A geometria é sempre 1×1; não precisa girar o node: o shader resolve a amostra 3D
        renderAndPresent()
    }

    func applyCachedDataset(dataset: DatasetCache?, orientation: OrientationCache?) {
        if let dataset {
            MPRViewportController.sharedDatasetCache = dataset
            setDataset(volumeTex: dataset.volumeTexture,
                       tfTex: dataset.transferTexture,
                       dimension: dataset.dimension,
                       resolution: dataset.resolution)
        } else if let cached = MPRViewportController.sharedDatasetCache {
            setDataset(volumeTex: cached.volumeTexture,
                       tfTex: cached.transferTexture,
                       dimension: cached.dimension,
                       resolution: cached.resolution)
        }

        let orientationToApply = orientation ?? MPRViewportController.sharedOrientationCache
        if let orientationToApply {
            MPRViewportController.sharedOrientationCache = orientationToApply
            apply(R: orientationToApply.basis, cross: orientationToApply.cross)
        }
    }

    private func renderAndPresent() {
        matMPR.renderIfNeeded()
        DispatchQueue.main.async { [weak scnView] in
            scnView?.setNeedsDisplay()
        }
    }

    // MARK: - SCNSceneRendererDelegate

    func renderer(_ renderer: SCNSceneRenderer, updateAtTime time: TimeInterval) {
        matMPR.renderIfNeeded()
    }

    private func reapplyCachedDatasetIfAvailable() {
        guard MPRViewportController.sharedDatasetCache != nil else { return }
        applyCachedDataset(dataset: nil, orientation: nil)
    }

    private func updateCameraZoom() {
        let baseScale = 1.0
        cameraNode.camera?.orthographicScale = Double(baseScale) / Double(zoomFactor)
    }

    // MARK: - Pointer scroll (trackpad)

    private lazy var pointerPanRecognizer: UIPanGestureRecognizer = {
        let recognizer = UIPanGestureRecognizer(target: self, action: #selector(handlePointerPan(_:)))
        recognizer.delegate = self
        recognizer.minimumNumberOfTouches = 0
        recognizer.maximumNumberOfTouches = 0
        if #available(iOS 13.4, *) {
            recognizer.allowedScrollTypesMask = .continuous
        }
        if #available(iOS 13.4, *) {
            recognizer.allowedTouchTypes = [
                NSNumber(value: UITouch.TouchType.indirect.rawValue),
                NSNumber(value: UITouch.TouchType.indirectPointer.rawValue)
            ]
        }
        recognizer.cancelsTouchesInView = false
        recognizer.delaysTouchesBegan = false
        recognizer.delaysTouchesEnded = false
        return recognizer
    }()

    @objc private func handlePointerPan(_ recognizer: UIPanGestureRecognizer) {
        guard recognizer.state == .changed || recognizer.state == .ended else { return }
        let translation = recognizer.translation(in: scnView)
        if translation == .zero && recognizer.state != .ended { return }
        recognizer.setTranslation(.zero, in: scnView)
        let height = max(scnView.bounds.height, 1)
        let normalized = -Float(translation.y / height)
        pointerScrollHandler?(normalized)
    }

    func gestureRecognizer(_ gestureRecognizer: UIGestureRecognizer,
                           shouldRecognizeSimultaneouslyWith otherGestureRecognizer: UIGestureRecognizer) -> Bool {
        true
    }
}

// MARK: - Representable para SwiftUI
struct MPRViewportView: UIViewRepresentable {
    typealias UIViewType = SCNView

    let controller: MPRViewportController

    func makeUIView(context: Context) -> SCNView { controller.view }
    func updateUIView(_ uiView: SCNView, context: Context) {}
}

// MARK: - Overlay com linhas/gestos
/// Desenha o crosshair oblíquo alinhado com a base atual e repassa gestos
/// (translações nos eixos da base + normal, rotação e gesto dedicado de fatia)
/// para o controlador.
struct CrosshairOverlay: View {
    let plane: TriPlane
    let cross: SIMD3<Float>
    let basis: simd_float3x3
    let activeTool: InteractionTool
    let onTranslateAxis: (Int, Float) -> Void
    let onTranslateNormal: (Int, Float) -> Void
    let onTiltAxis: (Int, Float) -> Void
    let onRotate: (CGFloat) -> Void
    let onAxisTwoGesture: (Float) -> Void
    let onWindowDrag: (CGSize, Bool) -> Void
    let onZoomDrag: (CGSize, Bool) -> Void

    @State private var lastAngle: CGFloat = 0
    @State private var accumulatedAxisDrag: [Int: CGFloat] = [:]
    @State private var accumulatedAxisTilt: [Int: CGFloat] = [:]
    @State private var accumulatedNormalDrag: CGFloat = 0
    @State private var lastMagnification: CGFloat = 1

    private func column(_ index: Int) -> SIMD3<Float> {
        switch index {
        case 0: return basis.columns.0
        case 1: return basis.columns.1
        default: return basis.columns.2
        }
    }

    private func referenceAxes(for plane: TriPlane) -> (SIMD3<Float>, SIMD3<Float>) {
        switch plane {
        case .axial:    return (SIMD3<Float>(1, 0, 0), SIMD3<Float>(0, 1, 0))
        case .coronal:  return (SIMD3<Float>(1, 0, 0), SIMD3<Float>(0, 0, 1))
        case .sagittal: return (SIMD3<Float>(0, 1, 0), SIMD3<Float>(0, 0, 1))
        }
    }

    private func axisConfiguration(for plane: TriPlane) -> ([Int], Int) {
        switch plane {
        case .axial:    return ([0, 1], 2)
        case .coronal:  return ([0, 2], 1)
        case .sagittal: return ([1, 2], 0)
        }
    }

    private func crossUV(for plane: TriPlane,
                         cross: SIMD3<Float>,
                         basis: simd_float3x3) -> (CGPoint, PlanePlacement) {
        let placement = computePlanePlacement(for: plane, basis: basis, cross: cross)
        let relative = cross - placement.origin
        let uLength = simd_length_squared(placement.axisU)
        let vLength = simd_length_squared(placement.axisV)
        let uParam = uLength > Float.ulpOfOne ? simd_dot(relative, placement.axisU) / uLength : 0
        let vParam = vLength > Float.ulpOfOne ? simd_dot(relative, placement.axisV) / vLength : 0
        let point = CGPoint(
            x: CGFloat(min(max(uParam, 0), 1)),
            y: CGFloat(min(max(vParam, 0), 1))
        )
        return (point, placement)
    }

    private func direction2D(axis: SIMD3<Float>, refs: (SIMD3<Float>, SIMD3<Float>)) -> CGVector {
        let dx = CGFloat(simd_dot(axis, refs.0))
        let dy = CGFloat(simd_dot(axis, refs.1))
        return CGVector(dx: dx, dy: dy)
    }

    private func pixelDirection(from dir: CGVector, size: CGSize) -> CGVector {
        CGVector(dx: dir.dx * size.width, dy: -dir.dy * size.height)
    }

    private func endpoints(center: CGPoint, direction: CGVector, size: CGSize) -> (CGPoint, CGPoint) {
        guard hypot(direction.dx, direction.dy) > .ulpOfOne else { return (center, center) }

        func endpoint(forward: Bool) -> CGPoint {
            let dir = forward ? direction : CGVector(dx: -direction.dx, dy: -direction.dy)
            var t = CGFloat.infinity

            if dir.dx != 0 {
                let bound = dir.dx > 0 ? size.width - center.x : -center.x
                t = min(t, bound / dir.dx)
            }
            if dir.dy != 0 {
                let bound = dir.dy > 0 ? size.height - center.y : -center.y
                t = min(t, bound / dir.dy)
            }

            if t == .infinity { t = 0 }
            return CGPoint(x: center.x + dir.dx * t, y: center.y + dir.dy * t)
        }

        return (endpoint(forward: false), endpoint(forward: true))
    }

    private func normalizedDelta(for translation: CGSize, axisDir: CGVector, size: CGSize) -> CGFloat {
        let pixelDir = pixelDirection(from: axisDir, size: size)
        let len = hypot(pixelDir.dx, pixelDir.dy)
        guard len > .ulpOfOne else { return 0 }

        let axisUnit = CGVector(dx: pixelDir.dx / len, dy: pixelDir.dy / len)
        let deltaPoints = translation.width * axisUnit.dx + translation.height * axisUnit.dy
        return deltaPoints / len
    }

    var body: some View {
        GeometryReader { geo in
            let size = geo.size
            let refs = referenceAxes(for: plane)
            let (axisIndices, normalIndex) = axisConfiguration(for: plane)
            let (uv, placement) = crossUV(for: plane, cross: cross, basis: basis)
            let center = CGPoint(
                x: size.width * min(max(uv.x, 0), 1),
                y: size.height * (1 - min(max(uv.y, 0), 1))
            )

            let axisDirs = axisIndices.map { index -> CGVector in
                var axisVector = column(index)
                if placement.flipVertical && index == 2 {
                    axisVector = -axisVector
                }
                return direction2D(axis: axisVector, refs: refs)
            }
            let normalAxisVector: SIMD3<Float> = placement.flipVertical && normalIndex == 2
                ? -column(normalIndex)
                : column(normalIndex)
            let normalDir = direction2D(axis: normalAxisVector, refs: refs)
            let pixelDirs = axisDirs.map { pixelDirection(from: $0, size: size) }
            let lineEndpoints = pixelDirs.map { endpoints(center: center, direction: $0, size: size) }

            let base = ZStack {
                ForEach(Array(axisIndices.enumerated()), id: \.offset) { entry in
                    let idx = entry.element
                    let dir = axisDirs[entry.offset]
                    let pixels = pixelDirs[entry.offset]
                    let ends = lineEndpoints[entry.offset]
                    let color: Color = entry.offset == 0 ? .yellow : .purple

                    Path { path in
                        path.move(to: ends.0)
                        path.addLine(to: ends.1)
                    }
                    .stroke(color, lineWidth: 2)
                    .overlay(
                        Capsule()
                            .fill(Color.clear)
                            .frame(width: 44, height: max(size.width, size.height) * 2)
                            .position(center)
                            .rotationEffect(
                                Angle(radians: pixels.dx == 0 && pixels.dy == 0 ? 0 : atan2(pixels.dy, pixels.dx))
                            )
                            .gesture(
                                DragGesture(minimumDistance: 0)
                                    .onChanged { value in
                                        guard activeTool == .navigate else { return }
                                        let primaryTotal = normalizedDelta(for: value.translation, axisDir: dir, size: size)
                                        let previousPrimary = accumulatedAxisDrag[idx] ?? 0
                                        let primaryDelta = primaryTotal - previousPrimary
                                        accumulatedAxisDrag[idx] = primaryTotal
                                        if abs(primaryDelta) > .ulpOfOne {
                                            onTranslateAxis(idx, Float(primaryDelta))
                                        }

                                        if axisIndices.count >= 2 {
                                            let otherOffset = (entry.offset + 1) % axisIndices.count
                                            let tiltTotal = normalizedDelta(for: value.translation,
                                                                            axisDir: axisDirs[otherOffset],
                                                                            size: size)
                                            let previousTilt = accumulatedAxisTilt[idx] ?? 0
                                            let tiltDelta = tiltTotal - previousTilt
                                            accumulatedAxisTilt[idx] = tiltTotal
                                            if abs(tiltDelta) > .ulpOfOne {
                                                onTiltAxis(idx, Float(tiltDelta))
                                            }
                                        }
                                    }
                                    .onEnded { _ in
                                        accumulatedAxisDrag[idx] = 0
                                        accumulatedAxisTilt[idx] = 0
                                    }
                            )
                    )
                }

                Circle()
                    .stroke(.white.opacity(0.9), lineWidth: 2)
                    .frame(width: 22, height: 22)
                    .position(center)
                    .contentShape(Circle().inset(by: -12))
                    .highPriorityGesture(
                        DragGesture(minimumDistance: 0)
                            .onChanged { value in
                                guard activeTool == .navigate else { return }
                                let total: CGFloat
                                if normalDir.dx == 0 && normalDir.dy == 0 {
                                    total = -value.translation.height / max(size.height, 1)
                                } else {
                                    total = normalizedDelta(for: value.translation, axisDir: normalDir, size: size)
                                }
                                let delta = total - accumulatedNormalDrag
                                accumulatedNormalDrag = total
                                onTranslateNormal(normalIndex, Float(delta))
                            }
                            .onEnded { _ in
                                guard activeTool == .navigate else { return }
                                accumulatedNormalDrag = 0
                            }
                    )
            }
            .contentShape(Rectangle())
            .simultaneousGesture(
                RotationGesture()
                    .onChanged { value in
                        guard activeTool == .navigate else { return }
                        let delta = value.radians - lastAngle
                        lastAngle = value.radians
                        onRotate(delta)
                    }
                    .onEnded { _ in lastAngle = 0 }
            )
            .simultaneousGesture(
                MagnificationGesture()
                    .onChanged { value in
                        guard activeTool == .zoom else { return }
                        let delta = value - lastMagnification
                        lastMagnification = value
                        guard abs(delta) > CGFloat.ulpOfOne else { return }
                        onAxisTwoGesture(Float(delta))
                    }
                    .onEnded { _ in lastMagnification = 1 }
            )

            let windowGesture = DragGesture(minimumDistance: 0)
                .onChanged { value in
                    onWindowDrag(value.translation, false)
                }
                .onEnded { value in
                    onWindowDrag(value.translation, true)
                }

            let zoomGesture = DragGesture(minimumDistance: 0)
                .onChanged { value in
                    onZoomDrag(value.translation, false)
                }
                .onEnded { value in
                    onZoomDrag(value.translation, true)
                }

            if activeTool == .window {
                base.highPriorityGesture(windowGesture, including: .all)
            } else if activeTool == .zoom {
                base.highPriorityGesture(zoomGesture, including: .all)
            } else {
                base
            }
        }
    }
}

// MARK: - Painel Tri‑Planar (3 viewports)
struct TriPlanarMPRView: View {
    @EnvironmentObject private var model: DrawOptionModel
    @Binding private var activeTool: InteractionTool
    @StateObject private var state = TriMPRState()

    typealias CachedDataset = MPRViewportController.DatasetCache

    @State private var pendingDevice: MTLDevice?
    @State private var pendingDatasetMeta: SceneViewController.DatasetMeta?
    @State private var axialController: MPRViewportController?
    @State private var coronalController: MPRViewportController?
    @State private var sagittalController: MPRViewportController?
    @State private var cachedDataset: CachedDataset?
    @State private var zoomFactor: Float = 1.0
    private let minZoom: Float = 0.25
    private let maxZoom: Float = 4.0
    private let pinchZoomSensitivity: Float = 1.0
    private let obliqueRotationGain: Float = .pi / 2
    private let pointerScrollGain: Float = 1.2
    @State private var windowDragInitialRange: (min: Float, max: Float)?
    @State private var zoomDragInitialFactor: Float?

    private var controllers: [MPRViewportController] {
        [axialController, coronalController, sagittalController].compactMap { $0 }
    }

    private var hasControllers: Bool {
        axialController != nil && coronalController != nil && sagittalController != nil
    }

    init(activeTool: Binding<InteractionTool>) {
        self._activeTool = activeTool
        let controller = SceneViewController.Instance

        if let device = controller.device {
            _pendingDevice = State(initialValue: device)
        } else {
            _pendingDevice = State(initialValue: nil)
        }

        if controller.mat != nil,
           let meta = controller.currentDatasetMeta(),
           let volumeTexture = controller.currentVolumeTexture() {
            let transferTexture = controller.currentTFTexture()
            let initialCache = CachedDataset(volumeTexture: volumeTexture,
                                             transferTexture: transferTexture,
                                             dimension: meta.dimension,
                                             resolution: meta.resolution)
            _pendingDatasetMeta = State(initialValue: meta)
            _cachedDataset = State(initialValue: initialCache)
        } else {
            _pendingDatasetMeta = State(initialValue: nil)
            _cachedDataset = State(initialValue: nil)
        }
    }

    // Aplica R/cross em todos
    private func syncAll() {
        guard hasControllers else { return }
        controllers.forEach { $0.apply(R: state.R, cross: state.cross) }
    }

    private func handleAxisTwoGesture(rawDelta: Float) {
        guard activeTool == .zoom else { return }
        let multiplier = powf(2.0, rawDelta * pinchZoomSensitivity)
        guard multiplier.isFinite, multiplier > 0 else { return }
        let newZoom = max(minZoom, min(maxZoom, zoomFactor * multiplier))
        guard abs(newZoom - zoomFactor) > 1e-3 else { return }
        zoomFactor = newZoom
        guard hasControllers else { return }
        controllers.forEach { $0.setZoom(newZoom) }
    }

    private func handleAxisTilt(axis: Int, deltaNormalized: Float) {
        guard activeTool == .navigate else { return }
        guard abs(deltaNormalized) > Float.ulpOfOne else { return }
        state.rotate(aroundBaseAxis: axis, deltaRadians: deltaNormalized * obliqueRotationGain)
        syncAll()
    }

    private func handleWindowDrag(translation: CGSize, ended: Bool) {
        guard activeTool == .window else {
            if ended { windowDragInitialRange = nil }
            return
        }
        if windowDragInitialRange == nil {
            let snapshot = SceneViewController.Instance.currentHuWindowSnapshot()
            windowDragInitialRange = (Float(snapshot.min), Float(snapshot.max))
        }
        guard let start = windowDragInitialRange else { return }
        var datasetMin: Float = -4096
        var datasetMax: Float = 4096
        if let bounds = SceneViewController.Instance.datasetHuBounds() {
            datasetMin = Float(bounds.min)
            datasetMax = Float(bounds.max)
        }
        let startDescriptor = WindowLevelMath.widthLevel(forMin: start.min, max: start.max)
        let startWidth = max(1.0, startDescriptor.width)
        let startLevel = startDescriptor.level
        let span = max(1.0, datasetMax - datasetMin)
        let levelSensitivity: Float = span / 400.0
        let widthSensitivity: Float = span / 400.0
        let deltaLevel = Float(translation.width) * levelSensitivity
        let deltaWidth = Float(-translation.height) * widthSensitivity
        var width = max(1.0, startWidth + deltaWidth)
        let maxWidth = max(1.0, (datasetMax - datasetMin) + 1)
        width = min(width, maxWidth)
        var level = startLevel + deltaLevel
        let bounds = WindowLevelMath.bounds(forWidth: width, level: level)
        let (clampedMin, clampedMax) = clampWindow(bounds: bounds, datasetMin: datasetMin, datasetMax: datasetMax)
        let minHU = Int32(round(clampedMin))
        let maxHU = Int32(round(clampedMax))
        SceneViewController.Instance.setHuWindow(minHU: minHU, maxHU: maxHU)
        if ended {
            windowDragInitialRange = nil
        }
    }

    private func handleZoomDrag(translation: CGSize, ended: Bool) {
        guard activeTool == .zoom else {
            if ended { zoomDragInitialFactor = nil }
            return
        }
        if zoomDragInitialFactor == nil {
            zoomDragInitialFactor = zoomFactor
        }
        guard let start = zoomDragInitialFactor else { return }
        let sensitivity: Float = 0.003
        let delta = Float(-translation.height) * sensitivity
        let newZoom = max(minZoom, min(maxZoom, start * powf(2.0, delta)))
        guard abs(newZoom - zoomFactor) > 1e-3 else {
            if ended { zoomDragInitialFactor = nil }
            return
        }
        zoomFactor = newZoom
        if hasControllers {
            controllers.forEach { $0.setZoom(newZoom) }
        }
        if ended {
            zoomDragInitialFactor = nil
        }
    }

    private func refreshDatasetFromSceneController() {
        let controller = SceneViewController.Instance

        guard controller.mat != nil,
              let meta = controller.currentDatasetMeta(),
              let volumeTex = controller.currentVolumeTexture()
        else {
            pendingDatasetMeta = nil
            cachedDataset = nil
            return
        }

        pendingDatasetMeta = meta
        state.updateVolumeDimensions(meta.dimension)

        let tfTex = controller.currentTFTexture()
        let dataset = CachedDataset(volumeTexture: volumeTex,
                                    transferTexture: tfTex,
                                    dimension: meta.dimension,
                                    resolution: meta.resolution)
        cachedDataset = dataset

        guard hasControllers else { return }

        let orientation = MPRViewportController.OrientationCache(basis: state.R, cross: state.cross)
        controllers.forEach { controller in
            controller.applyCachedDataset(dataset: dataset, orientation: orientation)
        }
        controllers.forEach { $0.setZoom(zoomFactor) }
    }

    private func refreshTransferFunctionFromSceneController() {
        let controller = SceneViewController.Instance
        guard controller.mat != nil,
              let tfTex = controller.currentTFTexture()
        else { return }

        if let dataset = cachedDataset {
            cachedDataset = CachedDataset(volumeTexture: dataset.volumeTexture,
                                          transferTexture: tfTex,
                                          dimension: dataset.dimension,
                                          resolution: dataset.resolution)
        }

        guard hasControllers else { return }

        controllers.forEach { $0.setTransferFunction(tfTex) }
        applyTFUsageFromModel()
    }

    private func applyBlendFromModel() {
        applyBlend(model.mprBlend)
    }

    private func applyHuWindowFromModel() {
        let minValue = Int32(model.huMinHU.rounded())
        let maxValue = Int32(model.huMaxHU.rounded())
        applyHuWindow(min: minValue, max: maxValue)
    }

    private func applyTFUsageFromModel() {
        applyUseTF(model.useTFMpr)
    }

    private func applyModelParameters() {
        applyBlendFromModel()
        applyHuWindowFromModel()
        applyTFUsageFromModel()
    }

    private func applyBlend(_ mode: MPRPlaneMaterial.BlendMode) {
        guard hasControllers else { return }
        controllers.forEach { $0.setBlend(mode) }
    }

    private func applyHuWindow(min: Int32, max: Int32) {
        guard hasControllers else { return }
        controllers.forEach { $0.setHU(min: min, max: max) }
    }

    private func applyUseTF(_ enabled: Bool) {
        SceneViewController.Instance.setMPRUseTF(enabled)
        guard hasControllers else { return }
        controllers.forEach { $0.setUseTF(enabled) }
    }

    private func handlePointerScroll(deltaNormalized: Float, normalAxis: Int) {
        guard activeTool == .navigate else { return }
        let scaled = deltaNormalized * pointerScrollGain
        applyTranslation(axis: normalAxis, delta: scaled)
    }

    private func configurePointerScrollHandlers() {
        axialController?.pointerScrollHandler = { delta in
            handlePointerScroll(deltaNormalized: delta, normalAxis: 2)
        }
        coronalController?.pointerScrollHandler = { delta in
            handlePointerScroll(deltaNormalized: delta, normalAxis: 1)
        }
        sagittalController?.pointerScrollHandler = { delta in
            handlePointerScroll(deltaNormalized: delta, normalAxis: 0)
        }
    }

    private func clampWindow(bounds: (min: Float, max: Float),
                             datasetMin: Float,
                             datasetMax: Float) -> (Float, Float) {
        var minValue = max(datasetMin, min(datasetMax, bounds.min))
        var maxValue = max(datasetMin, min(datasetMax, bounds.max))
        if maxValue < minValue {
            let mid = max(datasetMin, min(datasetMax, (minValue + maxValue) * 0.5))
            minValue = mid
            maxValue = mid
        }
        return (minValue, maxValue)
    }

    private func applyTranslation(axis: Int, delta: Float) {
        let clamped = state.normalizedGestureDelta(for: delta, axisIndex: axis)
        guard abs(clamped) > Float.ulpOfOne else { return }
        state.translate(alongBaseAxis: axis, deltaNormalized: clamped)
        syncAll()
    }

    private func createControllersIfNeeded(using device: MTLDevice) {
        guard !hasControllers else { return }

        let axial = MPRViewportController(plane: .axial, device: device)
        let coronal = MPRViewportController(plane: .coronal, device: device)
        let sagittal = MPRViewportController(plane: .sagittal, device: device)

        axialController = axial
        coronalController = coronal
        sagittalController = sagittal

        if let dataset = cachedDataset {
            let orientation = MPRViewportController.OrientationCache(basis: state.R, cross: state.cross)
            controllers.forEach { controller in
                controller.applyCachedDataset(dataset: dataset, orientation: orientation)
            }
        } else {
            refreshDatasetFromSceneController()
        }

        refreshTransferFunctionFromSceneController()
        controllers.forEach { $0.setZoom(zoomFactor) }
        syncAll()
        applyModelParameters()
        configurePointerScrollHandlers()
    }

    private func placeholderView() -> some View {
        GeometryReader { geo in
            VStack(spacing: 6) {
                Color.black.opacity(0.85)
                    .overlay(
                        ProgressView()
                            .progressViewStyle(.circular)
                            .tint(.white)
                    )
                    .frame(height: geo.size.height * 0.5)

                HStack(spacing: 6) {
                    Color.black.opacity(0.85)
                    Color.black.opacity(0.85)
                }
                .frame(height: geo.size.height * 0.5)
            }
        }
    }

    var body: some View {
        Group {
            if let axial = axialController,
               let coronal = coronalController,
               let sagittal = sagittalController {
                GeometryReader { geo in
                    VStack(spacing: 6) {
                        // Topo: Axial (ocupando largura toda)
                        ZStack {
                            MPRViewportView(controller: axial).background(Color.black)
                            CrosshairOverlay(
                                plane: .axial,
                                cross: state.cross,
                                basis: state.R,
                                activeTool: activeTool,
                                onTranslateAxis: { axis, delta in
                                    applyTranslation(axis: axis, delta: delta)
                                },
                                onTranslateNormal: { axis, delta in
                                    applyTranslation(axis: axis, delta: delta)
                                },
                                onTiltAxis: { axis, delta in
                                    handleAxisTilt(axis: axis, deltaNormalized: delta)
                                },
                                onRotate: { deltaRad in
                                    state.rotate(aroundBaseAxis: 2, deltaRadians: Float(deltaRad))
                                    syncAll()
                                },
                                onAxisTwoGesture: { raw in
                                    handleAxisTwoGesture(rawDelta: raw)
                                },
                                onWindowDrag: { translation, ended in
                                    handleWindowDrag(translation: translation, ended: ended)
                                },
                                onZoomDrag: { translation, ended in
                                    handleZoomDrag(translation: translation, ended: ended)
                                }
                            )
                            .allowsHitTesting(true)
                        }
                        .frame(height: geo.size.height * 0.5)

                        // Base: dois viewports lado a lado (Coronal e Sagital)
                        HStack(spacing: 6) {
                            ZStack {
                                MPRViewportView(controller: coronal).background(Color.black)
                                CrosshairOverlay(
                                    plane: .coronal,
                                    cross: state.cross,
                                    basis: state.R,
                                    activeTool: activeTool,
                                    onTranslateAxis: { axis, delta in
                                        applyTranslation(axis: axis, delta: delta)
                                    },
                                    onTranslateNormal: { axis, delta in
                                        applyTranslation(axis: axis, delta: delta)
                                },
                                onTiltAxis: { axis, delta in
                                    handleAxisTilt(axis: axis, deltaNormalized: delta)
                                },
                                onRotate: { deltaRad in
                                    state.rotate(aroundBaseAxis: 1, deltaRadians: -Float(deltaRad))
                                    syncAll()
                                },
                                    onAxisTwoGesture: { raw in
                                        handleAxisTwoGesture(rawDelta: raw)
                                    },
                                    onWindowDrag: { translation, ended in
                                        handleWindowDrag(translation: translation, ended: ended)
                                    },
                                    onZoomDrag: { translation, ended in
                                        handleZoomDrag(translation: translation, ended: ended)
                                    }
                                )
                            }
                            ZStack {
                                MPRViewportView(controller: sagittal).background(Color.black)
                                CrosshairOverlay(
                                    plane: .sagittal,
                                    cross: state.cross,
                                    basis: state.R,
                                    activeTool: activeTool,
                                    onTranslateAxis: { axis, delta in
                                        applyTranslation(axis: axis, delta: delta)
                                    },
                                    onTranslateNormal: { axis, delta in
                                        applyTranslation(axis: axis, delta: delta)
                                    },
                                    onTiltAxis: { axis, delta in
                                        handleAxisTilt(axis: axis, deltaNormalized: delta)
                                    },
                                    onRotate: { deltaRad in
                                        state.rotate(aroundBaseAxis: 0, deltaRadians: Float(deltaRad))
                                        syncAll()
                                    },
                                    onAxisTwoGesture: { raw in
                                        handleAxisTwoGesture(rawDelta: raw)
                                    },
                                    onWindowDrag: { translation, ended in
                                        handleWindowDrag(translation: translation, ended: ended)
                                    },
                                    onZoomDrag: { translation, ended in
                                        handleZoomDrag(translation: translation, ended: ended)
                                    }
                                )
                            }
                        }
                        .frame(height: geo.size.height * 0.5)
                    }
                }
            } else {
                placeholderView()
            }
        }
        .onAppear {
            let controller = SceneViewController.Instance

            if let device = controller.device {
                pendingDevice = device
                createControllersIfNeeded(using: device)
            } else {
                pendingDevice = nil
            }

            if controller.mat != nil,
               let meta = controller.currentDatasetMeta() {
                pendingDatasetMeta = meta
                state.updateVolumeDimensions(meta.dimension)
            } else {
                pendingDatasetMeta = nil
            }
        }
        .onReceive(NotificationCenter.default.publisher(for: .sceneControllerDatasetDidChange)) { _ in
            refreshDatasetFromSceneController()
            applyModelParameters()
        }
        .onReceive(NotificationCenter.default.publisher(for: .sceneControllerTransferFunctionDidChange)) { _ in
            refreshTransferFunctionFromSceneController()
        }
        .onChange(of: model.mprBlend) { _, _ in
            applyBlendFromModel()
        }
        .onChange(of: model.huMinHU) { _, _ in
            applyHuWindowFromModel()
        }
        .onChange(of: model.huMaxHU) { _, _ in
            applyHuWindowFromModel()
        }
        .onChange(of: model.useTFMpr) { _, _ in
            applyTFUsageFromModel()
        }
        .task(id: pendingDevice == nil) {
            guard pendingDevice == nil else { return }
            while pendingDevice == nil {
                if Task.isCancelled { break }
                if let device = SceneViewController.Instance.device {
                    await MainActor.run {
                        pendingDevice = device
                        createControllersIfNeeded(using: device)
                    }
                    break
                }
                try? await Task.sleep(nanoseconds: 100_000_000)
            }
        }
        .onChange(of: activeTool) { _, _ in
            windowDragInitialRange = nil
            zoomDragInitialFactor = nil
        }
    }
}
