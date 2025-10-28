import Foundation

struct WindowLevelPreset: Identifiable, Equatable {
    enum Modality: String {
        case ct
        case pt

        var displayName: String {
            rawValue.uppercased()
        }
    }

    enum Source: String, CaseIterable {
        case ohif
        case weasis

        var displayName: String {
            switch self {
            case .ohif: return "OHIF"
            case .weasis: return "Weasis"
            }
        }
    }

    let id: String
    let name: String
    let modality: Modality
    let window: Double
    let level: Double
    let source: Source

    var minValue: Float {
        WindowLevelMath.bounds(forWidth: Float(window), level: Float(level)).min
    }

    var maxValue: Float {
        WindowLevelMath.bounds(forWidth: Float(window), level: Float(level)).max
    }

    var fullDisplayName: String {
        "\(name) (\(source.displayName))"
    }

    var windowLevelSummary: String {
        "W \(format(window)) / L \(format(level))"
    }

    func matches(min: Float, max: Float, tolerance: Float = 1.0) -> Bool {
        abs(minValue - min) <= tolerance && abs(maxValue - max) <= tolerance
    }

    private func format(_ value: Double) -> String {
        let rounded = value.rounded()
        if abs(rounded - value) < 0.05 {
            return String(format: "%.0f", rounded)
        }
        return String(format: "%.1f", value)
    }
}

enum WindowLevelPresetLibrary {
    static let ct: [WindowLevelPreset] = [
        // OHIF presets
        WindowLevelPreset(id: "ohif.ct-soft-tissue", name: "Soft Tissue", modality: .ct, window: 400, level: 40, source: .ohif),
        WindowLevelPreset(id: "ohif.ct-lung", name: "Lung", modality: .ct, window: 1500, level: -600, source: .ohif),
        WindowLevelPreset(id: "ohif.ct-liver", name: "Liver", modality: .ct, window: 150, level: 90, source: .ohif),
        WindowLevelPreset(id: "ohif.ct-bone", name: "Bone", modality: .ct, window: 2500, level: 480, source: .ohif),
        WindowLevelPreset(id: "ohif.ct-brain", name: "Brain", modality: .ct, window: 80, level: 40, source: .ohif),

        // Weasis presets
        WindowLevelPreset(id: "weasis.ct-brain", name: "Brain", modality: .ct, window: 110, level: 35, source: .weasis),
        WindowLevelPreset(id: "weasis.ct-abdomen", name: "Abdomen", modality: .ct, window: 320, level: 50, source: .weasis),
        WindowLevelPreset(id: "weasis.ct-mediastinum", name: "Mediastinum", modality: .ct, window: 400, level: 80, source: .weasis),
        WindowLevelPreset(id: "weasis.ct-bone", name: "Bone", modality: .ct, window: 2000, level: 350, source: .weasis),
        WindowLevelPreset(id: "weasis.ct-lung", name: "Lung", modality: .ct, window: 1500, level: -500, source: .weasis),
        WindowLevelPreset(id: "weasis.ct-mip", name: "MIP", modality: .ct, window: 380, level: 120, source: .weasis)
    ]

    static let pt: [WindowLevelPreset] = [
        WindowLevelPreset(id: "ohif.pt-default", name: "Default", modality: .pt, window: 5, level: 2.5, source: .ohif),
        WindowLevelPreset(id: "ohif.pt-suv-3", name: "SUV 3", modality: .pt, window: 0, level: 3, source: .ohif),
        WindowLevelPreset(id: "ohif.pt-suv-5", name: "SUV 5", modality: .pt, window: 0, level: 5, source: .ohif),
        WindowLevelPreset(id: "ohif.pt-suv-7", name: "SUV 7", modality: .pt, window: 0, level: 7, source: .ohif),
        WindowLevelPreset(id: "ohif.pt-suv-8", name: "SUV 8", modality: .pt, window: 0, level: 8, source: .ohif),
        WindowLevelPreset(id: "ohif.pt-suv-10", name: "SUV 10", modality: .pt, window: 0, level: 10, source: .ohif),
        WindowLevelPreset(id: "ohif.pt-suv-15", name: "SUV 15", modality: .pt, window: 0, level: 15, source: .ohif)
    ]
}

enum WindowLevelMath {
    static func bounds(forWidth width: Float, level: Float) -> (min: Float, max: Float) {
        let clampedWidth = max(width, 1)
        let halfSpan = (clampedWidth - 1) * 0.5
        let minValue = level - 0.5 - halfSpan
        let maxValue = level - 0.5 + halfSpan
        return (minValue, maxValue)
    }

    static func widthLevel(forMin min: Float, max maxValue: Float) -> (width: Float, level: Float) {
        let span = maxValue - min
        let width = Swift.max(span + 1, 1)
        let level = min + width * 0.5
        return (width, level)
    }
}
