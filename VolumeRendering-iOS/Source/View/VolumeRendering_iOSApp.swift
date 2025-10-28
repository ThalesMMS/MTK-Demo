import SwiftUI
import Metal

@main
struct VolumeRendering_iOSApp: App {

    init() {
        Logger.log("VolumeRendering_iOSApp launched", level: .info, category: "App")
        logDefaultMetalDevice()
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }

    private func logDefaultMetalDevice() {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Logger.log("MTLDevice: nil", level: .warn, category: "Metal")
            Logger.log("Argument Buffers: desconhecido", level: .warn, category: "Metal")
            return
        }

        Logger.log("MTLDevice: \(device.name)", level: .info, category: "Metal")

        let tierDescription: String
        switch device.argumentBuffersSupport {
        case .tier2:
            tierDescription = "Argument Buffers: Tier 2 ✅"
        case .tier1:
            tierDescription = "Argument Buffers: Tier 1 (fallback) ⚠️"
        default:
            tierDescription = "Argument Buffers: desconhecido"
        }

        Logger.log(tierDescription, level: .info, category: "Metal")
    }
}
