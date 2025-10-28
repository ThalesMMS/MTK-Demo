import MetalKit
import SceneKit
import SwiftUI
import UIKit

final class VolumeSceneView: SCNView {
    private static func makeCommand(input: String, title: String) -> UIKeyCommand {
        let command = UIKeyCommand(title: title,
                                   action: #selector(handleEarlyTerminationKey(_:)),
                                   input: input,
                                   modifierFlags: [])
        command.discoverabilityTitle = title
        return command
    }

    private static var earlyTerminationCommands: [UIKeyCommand] {
        [
            makeCommand(input: "1", title: "Early termination threshold 0.90"),
            makeCommand(input: "2", title: "Early termination threshold 0.95"),
            makeCommand(input: "3", title: "Early termination threshold 0.99")
        ]
    }

    override var canBecomeFirstResponder: Bool { true }

    override func didMoveToWindow() {
        super.didMoveToWindow()
        if let window {
            // Delay claiming first responder until the view is fully attached to avoid
            // UIKit calling into responder maps before they are initialized (triggers the NSMapGet warning).
            DispatchQueue.main.async { [weak self, weak window] in
                guard
                    let self,
                    window != nil,
                    self.window != nil,
                    self.canBecomeFirstResponder
                else { return }
                _ = self.becomeFirstResponder()
            }
        } else if isFirstResponder {
            _ = resignFirstResponder()
        }
    }

    override var keyCommands: [UIKeyCommand]? {
        let inherited = super.keyCommands ?? []
        return inherited + Self.earlyTerminationCommands
    }

    @objc private func handleEarlyTerminationKey(_ sender: UIKeyCommand) {
        guard let input = sender.input else { return }
        SceneViewController.Instance.handleEarlyTerminationShortcut(for: input)
    }
}

struct SceneView: UIViewRepresentable {
    typealias UIViewType = VolumeSceneView
    var scnView: VolumeSceneView
    
    func makeUIView(context: Context) -> VolumeSceneView {
        let scene = SCNScene()
        scnView.allowsCameraControl = true
        scnView.autoenablesDefaultLighting = true
        scnView.showsStatistics = true
        scnView.backgroundColor = .clear
        scnView.clearsContextBeforeDrawing = true
        scnView.autoresizingMask = [.flexibleWidth, .flexibleHeight]

        scnView.scene = scene

        // NOVO: gestures para adaptive steps
        let controller = SceneViewController.Instance

        let pan = UIPanGestureRecognizer(target: controller, action: #selector(SceneViewController.handlePan(_:)))
        let pinch = UIPinchGestureRecognizer(target: controller, action: #selector(SceneViewController.handlePinch(_:)))
        let rot = UIRotationGestureRecognizer(target: controller, action: #selector(SceneViewController.handleRotate(_:)))
        pan.cancelsTouchesInView = false
        pinch.cancelsTouchesInView = false
        rot.cancelsTouchesInView = false
        pan.delegate = controller
        pinch.delegate = controller
        rot.delegate = controller
        scnView.addGestureRecognizer(pan)
        scnView.addGestureRecognizer(pinch)
        scnView.addGestureRecognizer(rot)

        let rotateTool = UIPanGestureRecognizer(target: controller, action: #selector(SceneViewController.handleRotateToolPan(_:)))
        rotateTool.maximumNumberOfTouches = 1
        rotateTool.cancelsTouchesInView = false
        rotateTool.delegate = controller
        scnView.addGestureRecognizer(rotateTool)

        let zoomTool = UIPinchGestureRecognizer(target: controller, action: #selector(SceneViewController.handleZoomToolPinch(_:)))
        zoomTool.cancelsTouchesInView = false
        zoomTool.delegate = controller
        scnView.addGestureRecognizer(zoomTool)

        controller.registerAdaptiveGestures(pan: pan, pinch: pinch, rotate: rot)
        controller.registerToolGestures(rotate: rotateTool, zoom: zoomTool)
        controller.setActiveTool(.navigate)

        return scnView
    }

    
    func updateUIView(_ uiView: VolumeSceneView, context: Context) {}
}
