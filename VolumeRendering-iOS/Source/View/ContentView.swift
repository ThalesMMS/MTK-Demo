import MetalKit
import SceneKit
import SwiftUI
import UniformTypeIdentifiers
import simd

struct ContentView: View {
    var view = VolumeSceneView()
    
    @State var showOption = true
    @StateObject var model = DrawOptionModel()
    @State private var isInitialized = false
    @State private var activeTool: InteractionTool = .navigate
    @State private var showWindowControls = false
    
    var body: some View {
        ZStack(alignment: .topLeading) {
            // === VR 3D e Tri‑Planar MPR montados simultaneamente ===
            ZStack {
                SceneView(scnView: view)
                    .opacity(model.method == .mpr ? 0 : 1)
                    .allowsHitTesting(model.method != .mpr)

                TriPlanarMPRView(activeTool: $activeTool)
                    .environmentObject(model)
                    .opacity(model.method == .mpr ? 1 : 0)
                    .allowsHitTesting(model.method == .mpr)
            }
            .background(.gray)
            .onAppear {
                if !isInitialized {
                    // Inicializa device/texturas no controller principal (mesmo que o Tri‑Planar esteja ativo)
                    SceneViewController.Instance.onAppear(view)
                    isInitialized = true
                }
                model.syncChannelsWithController()
                model.syncClipBoundsWithController()
                model.syncClipPlaneWithController()
                model.syncAdaptiveWithController()
                model.syncJitterWithController()
                model.syncHuWindowWithController()
                model.syncStepWithController()
            }
            .onChange(of: model.method) { _, newMethod in
                if newMethod != .mpr {
                    SceneViewController.Instance.restoreVolumeIfNeeded(in: view)
                }
            }
               
            // === Painel de opções (o seu, intacto) ===
            HStack(alignment: .top) {
                Button(showOption ? "hide" : "show") {
                    showOption.toggle()
                }
                
                if showOption {
                    ScrollView(.vertical, showsIndicators: true) {
                        DrawOptionView(model: model)
                            .padding(.trailing, 8)
                    }
                    .frame(maxWidth: 420, maxHeight: 360, alignment: .topLeading)
                    .background(.clear)
                }
            }
            .padding(.vertical, 25)
        }
        .overlay(alignment: .bottomTrailing) {
            ToolOverlay(activeTool: $activeTool,
                        showWindowControls: $showWindowControls,
                        model: model)
                .padding(16)
        }
        .ignoresSafeArea()
    }
}

// === MODELO (o seu, sem mudanças) ===
struct ChannelControlState: Identifiable, Equatable {
    let id: Int
    var presetKey: String
    var gain: Float
}

class DrawOptionModel: ObservableObject {
    @Published var method = SceneViewController.RenderMode.dvr
    @Published var preset = VolumeCubeMaterial.Preset.ct_arteries
    @Published var lightingOn: Bool = true
    @Published var step: Float = 512
    @Published var shift: Float = 0

    @Published var clipXMin: Float = 0.0
    @Published var clipXMax: Float = 1.0
    @Published var clipYMin: Float = 0.0
    @Published var clipYMax: Float = 1.0
    @Published var clipZMin: Float = 0.0
    @Published var clipZMax: Float = 1.0
    @Published var clipPlanePreset: SceneViewController.ClipPlanePreset = .off
    @Published var clipPlaneOffset: Float = 0.0
    @Published var adaptiveThreshold: Float = 0.1
    @Published var jitterAmount: Float = 0.0

    // NOVO:
    @Published var gateFloor: Float = 0.02
    @Published var gateCeil:  Float = 1.0
    @Published var useTFProj: Bool  = false
    @Published var adaptiveOn: Bool  = true

    // HU gate (projeções) e TF no MPR
    @Published var huGateOn: Bool = false
    @Published var huMinHU: Float = -900
    @Published var huMaxHU: Float = -500
    @Published var datasetHuMinBound: Float = -1024
    @Published var datasetHuMaxBound: Float = 3071
    @Published var useTFMpr: Bool = false
    let ctWindowPresets: [WindowLevelPreset] = WindowLevelPresetLibrary.ct
    @Published var selectedWindowPresetID: String?

    // MPR básico (os dois abaixo não são mais necessários para o tri‑planar,
    // mas mantive para compatibilidade com seu SceneViewController)
    @Published var mprOn: Bool = false
    @Published var mprBlend: MPRPlaneMaterial.BlendMode = .single
    @Published var mprAxialSlice: Float = 0
    enum MPRPlane: String, CaseIterable { case axial, coronal, sagittal }
    @Published var mprPlane: MPRPlane = .axial

    @Published var importedVolumeName: String?
    @Published var channels: [ChannelControlState] = (0..<4).map { index in
        ChannelControlState(id: index,
                            presetKey: index == 0 ? VolumeCubeMaterial.Preset.ct_arteries.rawValue : "none",
                            gain: index == 0 ? 1.0 : 0.0)
    }
    @Published var toneCurves: [ToneCurveChannelState] = []
    @Published var selectedToneChannel: Int = 0
    let toneCurvePresets = SceneViewController.Instance.toneCurvePresets()
    private var notificationTokens: [NSObjectProtocol] = []
    @Published var datasetSizeSummary: String?
    @Published var datasetSpacingSummary: String?
    @Published var datasetOriginSummary: String?
    @Published var datasetOrientationRows: [String] = []

    var selectedWindowPreset: WindowLevelPreset? {
        guard let id = selectedWindowPresetID else { return nil }
        return ctWindowPresets.first(where: { $0.id == id })
    }

    var selectedWindowPresetDisplayName: String {
        selectedWindowPreset?.fullDisplayName ?? "Personalizado"
    }

    var windowWidth: Float {
        let derived = WindowLevelMath.widthLevel(forMin: huMinHU, max: huMaxHU)
        return derived.width
    }

    var windowLevel: Float {
        let derived = WindowLevelMath.widthLevel(forMin: huMinHU, max: huMaxHU)
        return derived.level
    }

    var windowWidthUpperBound: Float {
        max(1, (datasetHuMaxBound - datasetHuMinBound) + 1)
    }

    var windowLevelRange: ClosedRange<Float> {
        let minBound = datasetHuMinBound
        let maxBound = datasetHuMaxBound
        return minBound <= maxBound ? minBound...maxBound : -2000...3000
    }

    init() {
        registerForToneCurveNotifications()
        syncToneCurvesWithController()
        syncClipBoundsWithController()
        syncClipPlaneWithController()
        syncAdaptiveWithController()
        syncMetadataWithController()
        syncJitterWithController()
    }

    deinit {
        for token in notificationTokens {
            NotificationCenter.default.removeObserver(token)
        }
    }

    private func registerForToneCurveNotifications() {
        let center = NotificationCenter.default
        let toneToken = center.addObserver(forName: .sceneControllerToneCurveDidChange,
                                           object: nil,
                                           queue: .main) { [weak self] _ in
            self?.syncToneCurvesWithController()
        }
        let histogramToken = center.addObserver(forName: .sceneControllerHistogramDidChange,
                                                object: nil,
                                                queue: .main) { [weak self] _ in
            self?.syncToneCurvesWithController()
        }
        let clipBoundsToken = center.addObserver(forName: .sceneControllerClipBoundsDidChange,
                                                 object: nil,
                                                 queue: .main) { [weak self] _ in
            self?.syncClipBoundsWithController()
        }
        let clipPlaneToken = center.addObserver(forName: .sceneControllerClipPlaneDidChange,
                                                object: nil,
                                                queue: .main) { [weak self] _ in
            self?.syncClipPlaneWithController()
        }
        let metadataToken = center.addObserver(forName: .sceneControllerMetadataDidChange,
                                                object: nil,
                                                queue: .main) { [weak self] _ in
            self?.syncMetadataWithController()
        }
        let huWindowToken = center.addObserver(forName: .sceneControllerHuWindowDidChange,
                                               object: nil,
                                               queue: .main) { [weak self] _ in
            self?.syncHuWindowWithController()
        }
        notificationTokens = [toneToken, histogramToken, clipBoundsToken, clipPlaneToken, metadataToken, huWindowToken]
    }

    func syncChannelsWithController() {
        let snapshot = SceneViewController.Instance.channelControlSnapshot()
        guard snapshot.count == channels.count else { return }
        for (idx, entry) in snapshot.enumerated() where idx < channels.count {
            channels[idx].presetKey = entry.presetKey
            channels[idx].gain = entry.gain
        }
        syncToneCurvesWithController()
    }

    func syncToneCurvesWithController() {
        let snapshots = SceneViewController.Instance.toneCurveSnapshot().sorted { $0.index < $1.index }
        toneCurves = snapshots.map {
            ToneCurveChannelState(id: $0.index,
                                  controlPoints: $0.controlPoints,
                                  histogram: $0.histogram,
                                  presetKey: $0.presetKey,
                                  gain: $0.gain)
        }
        if !toneCurves.contains(where: { $0.id == selectedToneChannel }) {
            selectedToneChannel = toneCurves.first?.id ?? 0
        }
    }

    func syncClipBoundsWithController() {
        let snapshot = SceneViewController.Instance.clipBoundsSnapshot().sanitized()
        clipXMin = snapshot.xMin
        clipXMax = snapshot.xMax
        clipYMin = snapshot.yMin
        clipYMax = snapshot.yMax
        clipZMin = snapshot.zMin
        clipZMax = snapshot.zMax
    }

    func syncClipPlaneWithController() {
        let snapshot = SceneViewController.Instance.clipPlaneSnapshot()
        clipPlanePreset = snapshot.preset
        clipPlaneOffset = snapshot.offset
    }

    func syncAdaptiveWithController() {
        adaptiveOn = SceneViewController.Instance.isAdaptiveEnabled()
        adaptiveThreshold = SceneViewController.Instance.currentAdaptiveThreshold()
    }

    func syncJitterWithController() {
        jitterAmount = SceneViewController.Instance.currentJitterAmount()
    }

    func applyWindowPreset(_ preset: WindowLevelPreset) {
        setHuWindow(min: preset.minValue, max: preset.maxValue)
        selectedWindowPresetID = preset.id
    }

    func updateHuWindow(min: Float, max: Float) {
        setHuWindow(min: min, max: max)
    }

    func updateWindowWidth(_ newWidth: Float) {
        let clampedWidth = max(1, min(newWidth, windowWidthUpperBound))
        let bounds = WindowLevelMath.bounds(forWidth: clampedWidth, level: windowLevel)
        setHuWindow(min: bounds.min, max: bounds.max)
    }

    func updateWindowLevel(_ newLevel: Float) {
        let bounds = WindowLevelMath.bounds(forWidth: windowWidth, level: newLevel)
        let (clampedMin, clampedMax) = clampWindowBounds(min: bounds.min, max: bounds.max)
        setHuWindow(min: clampedMin, max: clampedMax)
    }

    private func updateSelectedWindowPreset(min: Float, max: Float) {
        if let preset = matchingPreset(min: min, max: max) {
            if selectedWindowPresetID != preset.id {
                selectedWindowPresetID = preset.id
            }
        } else if selectedWindowPresetID != nil {
            selectedWindowPresetID = nil
        }
    }

    private func matchingPreset(min: Float, max: Float) -> WindowLevelPreset? {
        ctWindowPresets.first(where: { $0.matches(min: min, max: max) })
    }

    private func clampWindowBounds(min: Float, max: Float) -> (Float, Float) {
        var clampedMin = min
        var clampedMax = max
        if datasetHuMaxBound > datasetHuMinBound {
            clampedMin = Swift.max(datasetHuMinBound, Swift.min(datasetHuMaxBound, clampedMin))
            clampedMax = Swift.max(datasetHuMinBound, Swift.min(datasetHuMaxBound, clampedMax))
        }
        if clampedMax < clampedMin {
            let mid = Swift.max(datasetHuMinBound, Swift.min(datasetHuMaxBound, (clampedMin + clampedMax) * 0.5))
            clampedMin = mid
            clampedMax = mid
        }
        return (clampedMin, clampedMax)
    }

    private func setHuWindow(min: Float, max: Float) {
        let (clampedMin, clampedMax) = clampWindowBounds(min: min, max: max)
        huMinHU = clampedMin
        huMaxHU = clampedMax
        SceneViewController.Instance.setHuWindow(minHU: Int32(clampedMin.rounded()),
                                                 maxHU: Int32(clampedMax.rounded()))
        updateSelectedWindowPreset(min: clampedMin, max: clampedMax)
    }

    func syncHuWindowWithController() {
        let window = SceneViewController.Instance.currentHuWindow()
        let minHU = Float(window.min)
        let maxHU = Float(window.max)
        let bounds = SceneViewController.Instance.datasetHuBounds()
        if let bounds {
            datasetHuMinBound = Float(bounds.min)
            datasetHuMaxBound = Float(bounds.max)
        }
        if abs(huMinHU - minHU) > Float.ulpOfOne {
            huMinHU = minHU
        }
        if abs(huMaxHU - maxHU) > Float.ulpOfOne {
            huMaxHU = maxHU
        }
        updateSelectedWindowPreset(min: minHU, max: maxHU)
    }

    func syncStepWithController() {
        let quality = SceneViewController.Instance.currentRenderingQuality()
        if abs(step - quality) > 0.5 {
            step = quality
        }
    }

    func syncMetadataWithController() {
        guard let metadata = SceneViewController.Instance.currentVolumeMetadata() else {
            datasetSizeSummary = nil
            datasetSpacingSummary = nil
            datasetOriginSummary = nil
            datasetOrientationRows = []
            if let bounds = SceneViewController.Instance.datasetHuBounds() {
                datasetHuMinBound = Float(bounds.min)
                datasetHuMaxBound = Float(bounds.max)
            }
            return
        }

        datasetSizeSummary = String(format: "%d × %d × %d voxels",
                                     metadata.dimensions.x,
                                     metadata.dimensions.y,
                                     metadata.dimensions.z)

        datasetSpacingSummary = formatSpacing(metadata.spacing)
        datasetOriginSummary = formatOrigin(metadata.origin)
        datasetOrientationRows = formatOrientation(metadata.orientation)
        syncHuWindowWithController()
        syncStepWithController()
    }

    enum ClipAxis {
        case x, y, z
    }

    private func propagateClipBoundsToController() {
        SceneViewController.Instance.updateClipBounds(xMin: clipXMin,
                                                      xMax: clipXMax,
                                                      yMin: clipYMin,
                                                      yMax: clipYMax,
                                                      zMin: clipZMin,
                                                      zMax: clipZMax)
    }

    func updateClipBound(axis: ClipAxis, isMin: Bool, value: Float) {
        let clamped = min(max(value, 0.0), 1.0)
        switch axis {
        case .x:
            if isMin {
                clipXMin = clamped
                if clipXMin > clipXMax { clipXMax = clamped }
            } else {
                clipXMax = clamped
                if clipXMax < clipXMin { clipXMin = clamped }
            }
        case .y:
            if isMin {
                clipYMin = clamped
                if clipYMin > clipYMax { clipYMax = clamped }
            } else {
                clipYMax = clamped
                if clipYMax < clipYMin { clipYMin = clamped }
            }
        case .z:
            if isMin {
                clipZMin = clamped
                if clipZMin > clipZMax { clipZMax = clamped }
            } else {
                clipZMax = clamped
                if clipZMax < clipZMin { clipZMin = clamped }
            }
        }
        propagateClipBoundsToController()
    }

    func resetClipBoundsToDefault() {
        SceneViewController.Instance.resetClipBounds()
        syncClipBoundsWithController()
        syncClipPlaneWithController()
    }

    func alignClipBoundsToView() {
        SceneViewController.Instance.alignClipBoxToView()
        syncClipBoundsWithController()
    }

    func updateClipPlanePreset(_ preset: SceneViewController.ClipPlanePreset) {
        clipPlanePreset = preset
        SceneViewController.Instance.setClipPlanePreset(preset)
    }

    func updateClipPlaneOffset(_ value: Float) {
        let clamped = max(-0.5, min(value, 0.5))
        clipPlaneOffset = clamped
        SceneViewController.Instance.setClipPlaneOffset(clamped)
    }

    func alignClipPlaneToView() {
        SceneViewController.Instance.alignClipPlaneToView()
        syncClipPlaneWithController()
    }

    func updateAdaptiveThreshold(_ value: Float) {
        adaptiveThreshold = value
        SceneViewController.Instance.setAdaptiveThreshold(value)
    }

    func updateJitterAmount(_ value: Float) {
        jitterAmount = value
        SceneViewController.Instance.setJitterAmount(value)
    }

    private func formatSpacing(_ spacing: float3) -> String {
        let mm = spacing * 1000.0
        return String(format: "%.3f × %.3f × %.3f mm", mm.x, mm.y, mm.z)
    }

    private func formatOrigin(_ origin: float3) -> String {
        let mm = origin * 1000.0
        return String(format: "Origin: [%.2f, %.2f, %.2f] mm", mm.x, mm.y, mm.z)
    }

    private func formatOrientation(_ orientation: simd_float3x3) -> [String] {
        let row = orientation.columns.0
        let column = orientation.columns.1
        let normal = orientation.columns.2
        return [
            formatVector(name: "Row", vector: row),
            formatVector(name: "Column", vector: column),
            formatVector(name: "Normal", vector: normal)
        ]
    }

    private func formatVector(name: String, vector: SIMD3<Float>) -> String {
        String(format: "%@: [%.2f, %.2f, %.2f]", name, vector.x, vector.y, vector.z)
    }
}

// === VIEW DO PAINEL (o seu, com 1 ajuste no trecho final do MPR) ===
struct DrawOptionView: View {
    @ObservedObject var model: DrawOptionModel
    @State private var showingImporter = false
    @State private var isLoadingDicom = false
    @State private var isShowingImportError = false
    @State private var importErrorMessage = ""
    @State private var dicomProgress: Double = 0
    @State private var dicomStatus: String = ""

    private let dicomContentTypes: [UTType] = {
        var types: [UTType] = [.folder, .data]
        if let zip = UTType(filenameExtension: "zip") {
            types.append(zip)
        }
        if let dicom = UTType(filenameExtension: "dcm") {
            types.insert(dicom, at: 0)
        }
        return types
    }()

    private var availablePresets: [VolumeCubeMaterial.Preset] {
        VolumeCubeMaterial.Preset.allCases
    }
    
    var body: some View {
        VStack (spacing: 10) {
            HStack {
                Picker("Choose a mode", selection: $model.method) {
                    ForEach(SceneViewController.RenderMode.allCases, id: \.self) { mode in
                        Text(mode.rawValue)
                    }
                }
                .pickerStyle(SegmentedPickerStyle())
                .padding()
                .onChange(of: model.method) { _, newValue in
                    SceneViewController.Instance.setRenderMode(newValue)
                }
                .foregroundColor(.orange)
                .onAppear() {
                    UISegmentedControl.appearance().selectedSegmentTintColor = .blue
                    UISegmentedControl.appearance().setTitleTextAttributes([.foregroundColor: UIColor.white], for: .selected)
                    UISegmentedControl.appearance().setTitleTextAttributes([.foregroundColor: UIColor.blue], for: .normal)
                }
            }.frame(height: 30)
            
            HStack(spacing: 12) {
                Button {
                    showingImporter = true
                } label: {
                    Label("Import DICOM", systemImage: "tray.and.arrow.down")
                }
                .buttonStyle(.borderedProminent)
                .disabled(isLoadingDicom)

                if isLoadingDicom {
                    VStack(alignment: .leading, spacing: 4) {
                        ProgressView(value: dicomProgress)
                            .progressViewStyle(LinearProgressViewStyle())
                            .frame(width: 180)
                        Text(dicomStatus)
                            .font(.footnote)
                            .foregroundColor(.secondary)
                            .frame(maxWidth: 180, alignment: .leading)
                    }
                }
            }
            .padding(.horizontal)

            if let seriesName = model.importedVolumeName {
                Text("Loaded series: \(seriesName)")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            if let sizeSummary = model.datasetSizeSummary {
                Text("Size: \(sizeSummary)")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            if let spacingSummary = model.datasetSpacingSummary {
                Text("Spacing: \(spacingSummary)")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            if let originSummary = model.datasetOriginSummary {
                Text(originSummary)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            if !model.datasetOrientationRows.isEmpty {
                VStack(alignment: .leading, spacing: 2) {
                    ForEach(model.datasetOrientationRows, id: \.self) { row in
                        Text(row)
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
            }

            Divider()

            HStack {
                Group {
                    if availablePresets.count <= 4 {
                        Picker("Choose a Preset", selection: $model.preset) {
                            ForEach(availablePresets, id: \.self) { preset in
                                Text(preset.displayName).tag(preset)
                            }
                        }
                        .pickerStyle(SegmentedPickerStyle())
                        .padding()
                        .foregroundColor(.orange)
                        .onAppear {
                            UISegmentedControl.appearance().selectedSegmentTintColor = .blue
                            UISegmentedControl.appearance().setTitleTextAttributes([.foregroundColor: UIColor.white], for: .selected)
                            UISegmentedControl.appearance().setTitleTextAttributes([.foregroundColor: UIColor.blue], for: .normal)
                        }
                    } else {
                        Picker("Choose a Preset", selection: $model.preset) {
                            ForEach(availablePresets, id: \.self) { preset in
                                Text(preset.displayName).tag(preset)
                            }
                        }
                        .pickerStyle(MenuPickerStyle())
                        .padding(.horizontal)
                    }
                }
                .onChange(of: model.preset) { _, newPreset in
                    SceneViewController.Instance.setPreset(preset: newPreset)
                    model.shift = 0
                    model.syncChannelsWithController()
                }
            }
            .frame(height: 30)
            
            HStack {
                Toggle("Lighting On",
                       isOn: $model.lightingOn)
                .foregroundColor(.white)
                .onChange(of: model.lightingOn) { _, isLit in
                    SceneViewController.Instance.setLighting(isOn: isLit)
                }
            }.frame(height: 30)
            
            HStack {
                Text("Step")
                    .foregroundColor(.white)
                
                Slider(value: $model.step, in: 128...1024, step: 1)
                    .padding()
                    .onChange(of: model.step) { _, newStep in
                        SceneViewController.Instance.setStep(step: newStep)
                    }
            }.frame(height: 30)
            
            HStack {
                Text("Shift")
                    .foregroundColor(.white)
                Slider(value: $model.shift, in: -100...100, step: 1)
                    .padding()
                    .onChange(of: model.shift) { _, newShift in
                        SceneViewController.Instance.setShift(shift: newShift)
                    }
            }.frame(height: 30)

            VStack(alignment: .leading, spacing: 12) {
                Text("Channels")
                    .foregroundColor(.white)
                    .font(.headline)

                ForEach($model.channels) { $channel in
                    ChannelControlRow(channel: $channel)
                }
            }
            .padding(.horizontal)

            ToneCurveEditorPanel(model: model)

            Spacer()
            HStack {
                Picker("MPR Blend", selection: $model.mprBlend) {
                    ForEach(MPRPlaneMaterial.BlendMode.allCases, id: \.self) { mode in
                        let title: String = {
                            switch mode {
                            case .single: return "single"
                            case .mip:    return "mip"
                            case .minip:  return "minip"
                            case .mean:   return "avg"
                            }
                        }()
                        Text(title).tag(mode)
                    }
                }
                .pickerStyle(SegmentedPickerStyle())
                // O Tri‑Planar observa esta seleção e replica nos 3 painéis
                .onChange(of: model.mprBlend) { _, blend in
                    SceneViewController.Instance.setMPRBlend(blend)
                }
            }.frame(height: 30)

            HStack {
                Toggle("Gate por HU (projeções)", isOn: $model.huGateOn)
                    .onChange(of: model.huGateOn) { _, isEnabled in
                        SceneViewController.Instance.setHuGate(enabled: isEnabled)
                    }
            }.frame(height: 30)

            VStack {
                HStack {
                    Text("HU Min").foregroundColor(.white)
                    Slider(
                        value: Binding(get: { Double(model.huMinHU) }, set: { model.huMinHU = Float($0) }),
                        in: -1200...3000, step: 1
                    )
                        .padding()
                        .onChange(of: model.huMinHU) { _, newMin in
                            SceneViewController.Instance.setHuWindow(minHU: Int32(newMin), maxHU: Int32(model.huMaxHU))
                        }
                }.frame(height: 30)

                HStack {
                    Text("HU Max").foregroundColor(.white)
                    Slider(
                        value: Binding(get: { Double(model.huMaxHU) }, set: { model.huMaxHU = Float($0) }),
                        in: -1200...3000, step: 1
                    )
                        .padding()
                        .onChange(of: model.huMaxHU) { _, newMax in
                            SceneViewController.Instance.setHuWindow(minHU: Int32(model.huMinHU), maxHU: Int32(newMax))
                        }
                }.frame(height: 30)
            }

            VStack {
                HStack {
                    Text("Gate Floor").foregroundColor(.white)
                    Slider(
                        value: Binding(get: { Double(model.gateFloor) }, set: { model.gateFloor = Float($0) }),
                        in: 0.0...1.0, step: 0.01
                    )
                        .padding()
                        .onChange(of: model.gateFloor) { _, newFloor in
                            SceneViewController.Instance.setDensityGate(floor: newFloor, ceil: model.gateCeil)
                        }
                }.frame(height: 30)

                HStack {
                    Text("Gate Ceil").foregroundColor(.white)
                    Slider(
                        value: Binding(get: { Double(model.gateCeil) }, set: { model.gateCeil = Float($0) }),
                        in: 0.0...1.0, step: 0.01
                    )
                        .padding()
                        .onChange(of: model.gateCeil) { _, newCeil in
                            SceneViewController.Instance.setDensityGate(floor: model.gateFloor, ceil: newCeil)
                        }
                }.frame(height: 30)
            }

            VStack(alignment: .leading, spacing: 6) {
                Text("Clipping Box")
                    .foregroundColor(.white)
                    .font(.headline)

                HStack {
                    Text("X min").foregroundColor(.white).frame(width: 56, alignment: .leading)
                    Slider(
                        value: Binding(get: { Double(model.clipXMin) },
                                       set: { model.updateClipBound(axis: .x, isMin: true, value: Float($0)) }),
                        in: 0.0...1.0
                    )
                        .padding()
                }.frame(height: 30)

                HStack {
                    Text("X max").foregroundColor(.white).frame(width: 56, alignment: .leading)
                    Slider(
                        value: Binding(get: { Double(model.clipXMax) },
                                       set: { model.updateClipBound(axis: .x, isMin: false, value: Float($0)) }),
                        in: 0.0...1.0
                    )
                        .padding()
                }.frame(height: 30)

                HStack {
                    Text("Y min").foregroundColor(.white).frame(width: 56, alignment: .leading)
                    Slider(
                        value: Binding(get: { Double(model.clipYMin) },
                                       set: { model.updateClipBound(axis: .y, isMin: true, value: Float($0)) }),
                        in: 0.0...1.0
                    )
                        .padding()
                }.frame(height: 30)

                HStack {
                    Text("Y max").foregroundColor(.white).frame(width: 56, alignment: .leading)
                    Slider(
                        value: Binding(get: { Double(model.clipYMax) },
                                       set: { model.updateClipBound(axis: .y, isMin: false, value: Float($0)) }),
                        in: 0.0...1.0
                    )
                        .padding()
                }.frame(height: 30)

                HStack {
                    Text("Z min").foregroundColor(.white).frame(width: 56, alignment: .leading)
                    Slider(
                        value: Binding(get: { Double(model.clipZMin) },
                                       set: { model.updateClipBound(axis: .z, isMin: true, value: Float($0)) }),
                        in: 0.0...1.0
                    )
                        .padding()
                }.frame(height: 30)

                HStack {
                    Text("Z max").foregroundColor(.white).frame(width: 56, alignment: .leading)
                    Slider(
                        value: Binding(get: { Double(model.clipZMax) },
                                       set: { model.updateClipBound(axis: .z, isMin: false, value: Float($0)) }),
                        in: 0.0...1.0
                    )
                        .padding()
                }.frame(height: 30)

                HStack(spacing: 12) {
                    Button("Reset") {
                        model.resetClipBoundsToDefault()
                    }
                    .buttonStyle(.bordered)

                    Button("Align to view") {
                        model.alignClipBoundsToView()
                    }
                    .buttonStyle(.bordered)
                }
                .padding(.top, 4)
            }
            .padding(.horizontal)

            VStack(alignment: .leading, spacing: 6) {
                Text("Clipping Plane")
                    .foregroundColor(.white)
                    .font(.headline)

                Picker("Plane preset", selection: $model.clipPlanePreset) {
                    ForEach(SceneViewController.ClipPlanePreset.allCases) { preset in
                        Text(preset.displayName)
                            .tag(preset)
                    }
                }
                .pickerStyle(SegmentedPickerStyle())
                .onChange(of: model.clipPlanePreset) { _, newPreset in
                    model.updateClipPlanePreset(newPreset)
                }

                if model.clipPlanePreset != .off {
                    HStack {
                        Text("Offset").foregroundColor(.white).frame(width: 56, alignment: .leading)
                        Slider(
                            value: Binding(get: { Double(model.clipPlaneOffset) },
                                           set: { model.updateClipPlaneOffset(Float($0)) }),
                            in: -0.5...0.5
                        )
                            .padding()
                        Text(String(format: "%.2f", model.clipPlaneOffset))
                            .foregroundColor(.white)
                            .font(.caption.monospacedDigit())
                            .frame(width: 48, alignment: .trailing)
                    }.frame(height: 30)

                    if model.clipPlanePreset == .custom {
                        Button("Align custom plane to view") {
                            model.alignClipPlaneToView()
                        }
                        .buttonStyle(.bordered)
                    }
                }
            }
            .padding(.horizontal)

            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Toggle("Adaptive steps (durante interação)", isOn: $model.adaptiveOn)
                        .onChange(of: model.adaptiveOn) { _, isEnabled in
                            SceneViewController.Instance.setAdaptive(isEnabled)
                        }
                }.frame(height: 30)

                HStack {
                    Text("Gradient threshold")
                        .foregroundColor(.white)
                        .frame(width: 140, alignment: .leading)
                    Slider(
                        value: Binding(get: { Double(model.adaptiveThreshold) },
                                       set: { model.updateAdaptiveThreshold(Float($0)) }),
                        in: 0.01...0.5,
                        step: 0.01
                    )
                        .padding()
                        .disabled(!model.adaptiveOn)
                    Text(String(format: "%.2f", model.adaptiveThreshold))
                        .foregroundColor(.white)
                        .font(.caption.monospacedDigit())
                        .frame(width: 48, alignment: .trailing)
                }.frame(height: 30)

                HStack {
                    Text("Jitter amount")
                        .foregroundColor(.white)
                        .frame(width: 140, alignment: .leading)
                    Slider(
                        value: Binding(get: { Double(model.jitterAmount) },
                                       set: { model.updateJitterAmount(Float($0)) }),
                        in: 0.0...1.0,
                        step: 0.01
                    )
                        .padding()
                    Text(String(format: "%.2f", model.jitterAmount))
                        .foregroundColor(.white)
                        .font(.caption.monospacedDigit())
                        .frame(width: 48, alignment: .trailing)
                }.frame(height: 30)
            }
            .padding(.horizontal)

            // === Em MPR, o controle de navegação é pelas linhas; removi o slider de fatia ===
            if model.method == .mpr {
                Text("Tri‑Planar active: drag the lines to navigate, pinch to zoom, and use 2-finger rotation for oblique MPR.")
                    .foregroundColor(.white)
                    .font(.footnote)
            }
        }
        .fileImporter(isPresented: $showingImporter,
                       allowedContentTypes: dicomContentTypes,
                       allowsMultipleSelection: false) { result in
            switch result {
            case .success(let urls):
                guard let url = urls.first else { return }
                importDicom(from: url)
            case .failure(let error):
                importErrorMessage = error.localizedDescription
                isShowingImportError = true
            }
        }
        .alert("Import error", isPresented: $isShowingImportError, actions: {
            Button("OK", role: .cancel) {}
        }, message: {
            Text(importErrorMessage)
        })
    }
}

struct ChannelControlRow: View {
    @Binding var channel: ChannelControlState
    private let presetOptions = ["none"] + VolumeCubeMaterial.Preset.allCases.map { $0.rawValue }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Channel \(channel.id + 1)")
                    .foregroundColor(.white)
                    .font(.subheadline)

                Spacer()

                Picker("Transfer Function", selection: $channel.presetKey) {
                    ForEach(presetOptions, id: \.self) { option in
                        Text(displayName(for: option)).tag(option)
                    }
                }
                .pickerStyle(MenuPickerStyle())
            }

            HStack {
                Text("Gain")
                    .foregroundColor(.white)
                    .font(.caption)

                Slider(value: $channel.gain, in: 0...2, step: 0.05)
                    .accentColor(.orange)

                Text(String(format: "%.2f", channel.gain))
                    .foregroundColor(.white)
                    .font(.caption.monospacedDigit())
                    .frame(width: 48, alignment: .trailing)
            }
        }
        .padding(8)
        .background(Color.black.opacity(0.15))
        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
        .onChange(of: channel.presetKey) { _, newValue in
            SceneViewController.Instance.setChannelPreset(channel: channel.id, presetKey: newValue)
            if newValue == "none" && channel.gain > 0.0 {
                channel.gain = 0.0
            }
            let snapshot = SceneViewController.Instance.channelControlSnapshot()
            if channel.id < snapshot.count {
                let snapshotGain = snapshot[channel.id].gain
                if abs(snapshotGain - channel.gain) > 0.001 {
                    channel.gain = snapshotGain
                }
            }
        }
        .onChange(of: channel.gain) { _, newValue in
            SceneViewController.Instance.setChannelGain(channel: channel.id, gain: newValue)
        }
    }

    private func displayName(for option: String) -> String {
        if option == "none" { return "none" }
        if let preset = VolumeCubeMaterial.Preset(rawValue: option) {
            return preset.displayName
        }
        return option.replacingOccurrences(of: "_", with: " ").capitalized
    }
}

private extension DrawOptionView {
    func importDicom(from url: URL) {
        let hasScope = url.startAccessingSecurityScopedResource()
        isLoadingDicom = true
        dicomProgress = 0
        dicomStatus = "Preparando arquivos…"

        SceneViewController.Instance.loadDicomSeries(from: url, progress: { update in
            switch update {
            case .started(let totalSlices, let previewTarget):
                dicomProgress = 0
                dicomStatus = "Lendo série (\(totalSlices) fatias)…"
                if previewTarget > 0 {
                    dicomStatus += " Prévia em \(previewTarget) fatias."
                }
            case .reading(let fraction):
                dicomProgress = max(0, min(1, fraction))
            case .previewAvailable(let fraction):
                dicomProgress = max(dicomProgress, min(1, fraction))
                let percent = Int((dicomProgress * 100).rounded())
                if fraction >= 1 {
                    dicomStatus = "Finalizando volume (\(percent)% concluído)…"
                } else {
                    dicomStatus = "Pré-visualização disponível (\(percent)% concluído)."
                }
            }
        }, completion: { result in
            if hasScope {
                url.stopAccessingSecurityScopedResource()
            }
            isLoadingDicom = false

            switch result {
            case .success(let importResult):
                model.importedVolumeName = importResult.seriesDescription
                model.shift = 0
                dicomProgress = 1
                dicomStatus = "Volume importado com sucesso."
            case .failure(let error):
                importErrorMessage = friendlyMessage(for: error)
                isShowingImportError = true
                dicomProgress = 0
                dicomStatus = ""
            }
        })
    }

    func friendlyMessage(for error: Error) -> String {
        if let loaderError = error as? DicomVolumeLoaderError {
            switch loaderError {
            case .unsupportedBitDepth:
                return "Apenas séries DICOM escalares de 16 bits são suportadas no momento."
            case .securityScopeUnavailable:
                return "Não foi possível acessar os arquivos selecionados."
            case .missingResult:
                return "A conversão da série DICOM não retornou dados."
            case .bridgeError(let nsError):
                return nsError.localizedDescription
            }
        }

        let nsError = error as NSError
        if nsError.domain == DICOMSeriesLoaderErrorDomain {
            return nsError.localizedDescription
        }

        return error.localizedDescription
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
            .previewInterfaceOrientation(.landscapeRight)
            .previewDevice("iPad Pro (11-inch) (3rd generation)")
    }
}
struct ToolOverlay: View {
    @Binding var activeTool: InteractionTool
    @Binding var showWindowControls: Bool
    @ObservedObject var model: DrawOptionModel

    private let buttonSpacing: CGFloat = 12

    var body: some View {
        VStack(alignment: .trailing, spacing: 12) {
            HStack(spacing: buttonSpacing) {
                ForEach(InteractionTool.allCases, id: \.self) { tool in
                    ToolButton(icon: icon(for: tool),
                               label: label(for: tool),
                               isActive: activeTool == tool) {
                        activeTool = tool
                        SceneViewController.Instance.setActiveTool(tool)
                    }
                }
            }

            Button(action: { withAnimation { showWindowControls.toggle() } }) {
                Label("Windowing", systemImage: "slider.horizontal.3")
                    .padding(.horizontal, 12)
                    .padding(.vertical, 10)
                    .background(Color.black.opacity(0.35))
                    .clipShape(Capsule())
                    .foregroundColor(.white)
            }

            if showWindowControls {
                WindowControls(model: model)
                    .frame(width: 320)
                    .background(.ultraThinMaterial)
                    .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
                    .shadow(radius: 8)
            }
        }
    }

    private func icon(for tool: InteractionTool) -> String {
        switch tool {
        case .navigate: return "rotate.3d"
        case .window: return "slider.horizontal.3"
        case .zoom: return "magnifyingglass"
        }
    }

    private func label(for tool: InteractionTool) -> String {
        switch tool {
        case .navigate: return "Scroll/Rotate"
        case .window: return "Windowing"
        case .zoom: return "Zoom"
        }
    }
}

private struct ToolButton: View {
    let icon: String
    let label: String
    let isActive: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(spacing: 4) {
                Image(systemName: icon)
                    .font(.system(size: 18, weight: .medium))
                Text(label)
                    .font(.caption2)
            }
            .padding(10)
            .background(isActive ? Color.accentColor.opacity(0.85) : Color.black.opacity(0.35))
            .foregroundColor(.white)
            .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        }
    }
}

struct WindowControls: View {
    @ObservedObject var model: DrawOptionModel

    private var controllerWindow: (min: Int32, max: Int32) {
        SceneViewController.Instance.currentHuWindow()
    }

    private var displayWindowWidth: Float {
        let descriptor = WindowLevelMath.widthLevel(forMin: Float(controllerWindow.min),
                                                    max: Float(controllerWindow.max))
        return descriptor.width
    }

    private var displayWindowLevel: Float {
        let descriptor = WindowLevelMath.widthLevel(forMin: Float(controllerWindow.min),
                                                    max: Float(controllerWindow.max))
        return descriptor.level
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Window / Level")
                .font(.headline)

            if !model.ctWindowPresets.isEmpty {
                Menu {
                    ForEach(WindowLevelPreset.Source.allCases, id: \.self) { source in
                        let presets = model.ctWindowPresets.filter { $0.source == source }
                        if !presets.isEmpty {
                            Section(source.displayName) {
                                ForEach(presets) { preset in
                                    Button {
                                        model.applyWindowPreset(preset)
                                    } label: {
                                        HStack {
                                            Text(preset.name)
                                            Spacer()
                                            Text(preset.windowLevelSummary)
                                                .foregroundColor(.secondary)
                                            if model.selectedWindowPresetID == preset.id {
                                                Image(systemName: "checkmark")
                                                    .font(.system(size: 12, weight: .bold))
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                } label: {
                    HStack {
                        Text(model.selectedWindowPresetDisplayName)
                            .lineLimit(1)
                        Spacer()
                        Image(systemName: "chevron.up.chevron.down")
                            .font(.footnote)
                    }
                    .padding(.horizontal, 10)
                    .padding(.vertical, 8)
                    .background(Color.black.opacity(0.1))
                    .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                }

                Text("Preset atual: \(model.selectedWindowPresetDisplayName)")
                    .font(.caption2)
                    .foregroundColor(.secondary)

                Text("W \(formatted(displayWindowWidth)) / L \(formatted(displayWindowLevel))")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }

            VStack(alignment: .leading, spacing: 4) {
                Text("Window Width: \(Int(displayWindowWidth.rounded()))")
                    .font(.caption)
                    .foregroundColor(.secondary)
                Slider(value: Binding(get: { Double(model.windowWidth) }, set: { value in
                    model.updateWindowWidth(Float(value))
                }), in: 1...Double(max(1, model.windowWidthUpperBound)), step: 1)
            }

            VStack(alignment: .leading, spacing: 4) {
                Text("Window Level: \(Int(displayWindowLevel.rounded()))")
                    .font(.caption)
                    .foregroundColor(.secondary)
                let range = model.windowLevelRange
                Slider(value: Binding(get: { Double(model.windowLevel) }, set: { value in
                    model.updateWindowLevel(Float(value))
                }), in: Double(range.lowerBound)...Double(range.upperBound), step: 1)
            }

            Divider()

            Toggle("TF nas projeções", isOn: Binding(get: { model.useTFProj }, set: { newValue in
                model.useTFProj = newValue
                SceneViewController.Instance.setUseTFOnProjections(newValue)
            }))

            Toggle("TF no MPR", isOn: Binding(get: { model.useTFMpr }, set: { newValue in
                model.useTFMpr = newValue
            }))
        }
        .padding(16)
    }

    private func formatted(_ value: Float) -> String {
        let rounded = value.rounded()
        if abs(rounded - value) < 0.05 {
            return String(format: "%.0f", rounded)
        }
        return String(format: "%.1f", value)
    }
}
