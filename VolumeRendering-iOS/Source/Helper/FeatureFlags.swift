import Foundation
import Metal

struct FeatureFlags: OptionSet, CustomStringConvertible {
    let rawValue: Int

    static let argumentBuffers = FeatureFlags(rawValue: 1 << 0)
    static let nonUniformThreadgroups = FeatureFlags(rawValue: 1 << 1)
    static let heapAllocations = FeatureFlags(rawValue: 1 << 2)

    static func evaluate(for device: MTLDevice) -> FeatureFlags {
        var flags: FeatureFlags = []

        if device.supportsAdvancedArgumentBuffers {
            flags.insert(.argumentBuffers)
        }

        if device.supportsNonUniformThreadgroups {
            flags.insert(.nonUniformThreadgroups)
        }

        if device.supportsPrivateHeaps {
            flags.insert(.heapAllocations)
        }

        return flags
    }

    var description: String {

        var components: [String] = []
        if contains(.argumentBuffers) { components.append("argumentBuffers") }
        if contains(.nonUniformThreadgroups) { components.append("nonUniformThreadgroups") }
        if contains(.heapAllocations) { components.append("heapAllocations") }
        if components.isEmpty { components.append("none") }
        return components.joined(separator: ", ")
    }
}

extension MTLDevice {
    func evaluatedFeatureFlags() -> FeatureFlags {
        FeatureFlags.evaluate(for: self)
    }
}

private extension MTLDevice {
    var supportsAdvancedArgumentBuffers: Bool {
        if #available(iOS 13.0, *) {
            if supportsAnyFamily([.apple4, .apple5, .apple6, .apple7, .apple8, .apple9, .apple10, .common3]) {
                return true
            }
        } else {
            return argumentBuffersSupport == .tier2
        }

        return argumentBuffersSupport == .tier2
    }

    var supportsNonUniformThreadgroups: Bool {
        if #available(iOS 13.0, *) {
            if supportsAnyFamily([.apple4, .apple5, .apple6, .apple7, .apple8, .apple9, .apple10, .common3]) {
                return true
            }
        }

        return false
    }

    var supportsPrivateHeaps: Bool {
        if #available(iOS 13.0, *) {
            if supportsAnyFamily([.apple4, .apple5, .apple6, .apple7, .apple8, .apple9, .apple10, .common3]) {
                return true
            }
        }

        return false
    }

    @available(iOS 13.0, *)
    func supportsAnyFamily(_ families: [MTLGPUFamily]) -> Bool {
        families.contains { supportsFamily($0) }
    }
}
