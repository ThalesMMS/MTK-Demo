import SwiftUI

struct ToneCurveChannelState: Identifiable, Equatable {
    let id: Int
    var controlPoints: [ToneCurvePoint]
    var histogram: [UInt32]
    var presetKey: String
    var gain: Float
}

struct ToneCurveEditorPanel: View {
    @ObservedObject var model: DrawOptionModel

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Tone Curve")
                .font(.headline)
                .foregroundColor(.white)

            if model.toneCurves.isEmpty {
                Text("Carregue um volume para visualizar histogramas e editar a curva.")
                    .font(.caption)
                    .foregroundColor(.secondary)
            } else {
                Picker("Canal", selection: $model.selectedToneChannel) {
                    ForEach(model.toneCurves) { channel in
                        Text("Ch\(channel.id + 1)").tag(channel.id)
                    }
                }
                .pickerStyle(.segmented)
                .padding(.trailing, 8)

                if let binding = bindingForSelectedChannel() {
                    let currentChannel = binding.wrappedValue
                    let onPointsChanged: ([ToneCurvePoint]) -> Void = { points in
                        SceneViewController.Instance.updateToneCurve(channel: model.selectedToneChannel,
                                                                     controlPoints: points)
                    }
                    let onAddPoint: (ToneCurvePoint) -> Void = { point in
                        SceneViewController.Instance.insertToneCurvePoint(channel: model.selectedToneChannel,
                                                                           point: point)
                    }
                    let onRemovePoint: (Int) -> Void = { index in
                        SceneViewController.Instance.removeToneCurvePoint(channel: model.selectedToneChannel,
                                                                          pointIndex: index)
                    }

                    ToneCurveGraph(channel: binding,
                                   channelIndex: model.selectedToneChannel,
                                   onPointsChanged: onPointsChanged,
                                   onAddPoint: onAddPoint,
                                   onRemovePoint: onRemovePoint)
                        .frame(height: 220)
                        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                        .overlay(
                            RoundedRectangle(cornerRadius: 12, style: .continuous)
                                .stroke(Color.white.opacity(0.25), lineWidth: 1)
                        )
                        .padding(.vertical, 4)

                    controlButtons(for: currentChannel, onAddPoint: onAddPoint)

                    Text("Arraste os pontos para ajustar. Toque duas vezes em um ponto intermediário para removê-lo.")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
            }
        }
        .padding(.vertical, 12)
    }

    private func bindingForSelectedChannel() -> Binding<ToneCurveChannelState>? {
        guard let index = model.toneCurves.firstIndex(where: { $0.id == model.selectedToneChannel }) else {
            return nil
        }
        return $model.toneCurves[index]
    }

    @ViewBuilder
    private func controlButtons(for channel: ToneCurveChannelState,
                                onAddPoint: @escaping (ToneCurvePoint) -> Void) -> some View {
        HStack(spacing: 12) {
            Menu("Presets") {
                ForEach(model.toneCurvePresets, id: \.id) { preset in
                    Button(preset.title) {
                        SceneViewController.Instance.applyAutoWindowPreset(preset,
                                                                           channel: model.selectedToneChannel)
                    }
                }
            }
            .menuStyle(.borderlessButton)
            .foregroundColor(.white)

            Button {
                onAddPoint(defaultInsertionPoint(for: channel))
            } label: {
                Label("Adicionar ponto", systemImage: "plus.circle")
            }
            .buttonStyle(.borderedProminent)
            .tint(.blue.opacity(0.6))

            Button {
                SceneViewController.Instance.resetToneCurve(channel: model.selectedToneChannel)
            } label: {
                Label("Resetar", systemImage: "arrow.counterclockwise")
            }
            .buttonStyle(.bordered)
        }
    }

    private func defaultInsertionPoint(for channel: ToneCurveChannelState) -> ToneCurvePoint {
        let points = channel.controlPoints
        guard points.count >= 2 else {
            return ToneCurvePoint(x: 128, y: 0.5)
        }

        var bestIndex = 0
        var bestGap: Float = 0
        for index in 0..<(points.count - 1) {
            let gap = points[index + 1].x - points[index].x
            if gap > bestGap {
                bestGap = gap
                bestIndex = index
            }
        }

        let start = points[bestIndex]
        let end = points[bestIndex + 1]
        let newX = start.x + bestGap / 2
        let newY = (start.y + end.y) / 2
        return ToneCurvePoint(x: newX, y: newY)
    }
}

private struct ToneCurveGraph: View {
    @Binding var channel: ToneCurveChannelState
    let channelIndex: Int
    let onPointsChanged: ([ToneCurvePoint]) -> Void
    let onAddPoint: (ToneCurvePoint) -> Void
    let onRemovePoint: (Int) -> Void

    var body: some View {
        GeometryReader { geometry in
            let size = geometry.size
            ZStack {
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .fill(Color.black.opacity(0.35))

                Canvas { context, canvasSize in
                    drawGrid(in: canvasSize, context: &context)
                    drawHistogram(in: canvasSize, context: &context)
                    drawCurve(in: canvasSize, context: &context)
                }

                handles(in: size)
            }
        }
    }

    private func drawGrid(in size: CGSize, context: inout GraphicsContext) {
        var gridPath = Path()
        let thirds: [CGFloat] = [0, 0.5, 1.0]
        for value in thirds {
            let y = size.height * (1 - value)
            gridPath.move(to: CGPoint(x: 0, y: y))
            gridPath.addLine(to: CGPoint(x: size.width, y: y))
        }
        for value in thirds {
            let x = size.width * value
            gridPath.move(to: CGPoint(x: x, y: 0))
            gridPath.addLine(to: CGPoint(x: x, y: size.height))
        }
        context.stroke(gridPath, with: .color(Color.white.opacity(0.08)), lineWidth: 1)
    }

    private func drawHistogram(in size: CGSize, context: inout GraphicsContext) {
        let normalized = normalizedHistogram(channel.histogram)
        guard normalized.count > 1 else { return }

        var path = Path()
        path.move(to: CGPoint(x: 0, y: size.height))
        for (index, value) in normalized.enumerated() {
            let x = CGFloat(index) / CGFloat(normalized.count - 1) * size.width
            let y = size.height - CGFloat(value) * size.height
            path.addLine(to: CGPoint(x: x, y: y))
        }
        path.addLine(to: CGPoint(x: size.width, y: size.height))
        path.closeSubpath()
        context.fill(path, with: .color(Color.blue.opacity(0.25)))
    }

    private func drawCurve(in size: CGSize, context: inout GraphicsContext) {
        let samples = sampledCurveValues()
        guard samples.count > 1 else { return }

        var path = Path()
        for (index, value) in samples.enumerated() {
            let x = CGFloat(index) / CGFloat(samples.count - 1) * size.width
            let y = size.height - CGFloat(value) * size.height
            if index == 0 {
                path.move(to: CGPoint(x: x, y: y))
            } else {
                path.addLine(to: CGPoint(x: x, y: y))
            }
        }
        context.stroke(path, with: .color(Color.white), lineWidth: 2)
    }

    @ViewBuilder
    private func handles(in size: CGSize) -> some View {
        ForEach(Array(channel.controlPoints.enumerated()), id: \.offset) { index, point in
            let position = position(for: point, in: size)
            let isEndpoint = (index == 0 || index == channel.controlPoints.count - 1)

            Circle()
                .fill(isEndpoint ? Color.orange : Color.white)
                .frame(width: 16, height: 16)
                .position(position)
                .gesture(dragGesture(for: index, canvasSize: size))
                .onTapGesture(count: 2) {
                    guard !isEndpoint else { return }
                    channel.controlPoints.remove(at: index)
                    onRemovePoint(index)
                }
        }
    }

    private func dragGesture(for index: Int, canvasSize: CGSize) -> some Gesture {
        DragGesture(minimumDistance: 0)
            .onChanged { value in
                updatePoint(at: index, with: value.location, canvasSize: canvasSize)
            }
            .onEnded { value in
                updatePoint(at: index, with: value.location, canvasSize: canvasSize)
            }
    }

    private func updatePoint(at index: Int, with location: CGPoint, canvasSize: CGSize) {
        guard channel.controlPoints.indices.contains(index) else { return }
        var points = channel.controlPoints
        let clamped = clampedPoint(for: location, index: index, points: points, canvasSize: canvasSize)
        points[index] = clamped
        channel.controlPoints = points
        onPointsChanged(points)
    }

    private func normalizedHistogram(_ histogram: [UInt32]) -> [Double] {
        guard !histogram.isEmpty else { return [] }
        let values = histogram.map { Double($0) }
        let sorted = values.sorted()
        let count = values.count
        let lowerIndex = Int(Double(count) * 0.02)
        let upperIndex = Int(Double(count) * 0.98)
        let lowerBound = sorted[max(0, min(lowerIndex, count - 1))]
        let upperBound = sorted[max(0, min(upperIndex, count - 1))]
        let clamped = values.map { min(max($0, lowerBound), upperBound) }
        let maxValue = clamped.max() ?? 1
        if maxValue <= 0 { return Array(repeating: 0, count: count) }
        return clamped.map { $0 / maxValue }
    }

    private func sampledCurveValues() -> [Float] {
        guard channel.controlPoints.count >= 2 else { return [] }
        let model = ToneCurveModel(points: channel.controlPoints)
        return model.sampledValues(scale: ToneCurveModel.sampleScale)
    }

    private func position(for point: ToneCurvePoint, in size: CGSize) -> CGPoint {
        let rangeX = ToneCurveModel.xRange.upperBound - ToneCurveModel.xRange.lowerBound
        let normalizedX = CGFloat((point.x - ToneCurveModel.xRange.lowerBound) / max(rangeX, 1))
        let normalizedY = CGFloat(point.y)
        let x = normalizedX * size.width
        let y = size.height - normalizedY * size.height
        return CGPoint(x: x, y: y)
    }

    private func clampedPoint(for location: CGPoint,
                              index: Int,
                              points: [ToneCurvePoint],
                              canvasSize: CGSize) -> ToneCurvePoint {
        let clampedX = min(max(location.x / canvasSize.width, 0), 1)
        let clampedY = min(max(1 - (location.y / canvasSize.height), 0), 1)
        let rangeX = ToneCurveModel.xRange.upperBound - ToneCurveModel.xRange.lowerBound

        var normalizedX = Float(clampedX) * rangeX + ToneCurveModel.xRange.lowerBound
        var normalizedY = Float(clampedY)

        if index == 0 {
            normalizedX = ToneCurveModel.xRange.lowerBound
        } else if index == points.count - 1 {
            normalizedX = ToneCurveModel.xRange.upperBound
        } else {
            let lower = points[index - 1].x + ToneCurveModel.minimumDeltaX
            let upper = points[index + 1].x - ToneCurveModel.minimumDeltaX
            normalizedX = max(lower, min(normalizedX, upper))
        }

        normalizedY = max(ToneCurveModel.yRange.lowerBound,
                          min(normalizedY, ToneCurveModel.yRange.upperBound))

        return ToneCurvePoint(x: normalizedX, y: normalizedY)
    }
}
