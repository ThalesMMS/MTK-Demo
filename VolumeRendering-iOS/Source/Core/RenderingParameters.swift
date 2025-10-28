import Foundation
import simd

struct PackedColor: sizeable {
    var ch1: SIMD4<Float> = SIMD4<Float>(repeating: 0)
    var ch2: SIMD4<Float> = SIMD4<Float>(repeating: 0)
    var ch3: SIMD4<Float> = SIMD4<Float>(repeating: 0)
    var ch4: SIMD4<Float> = SIMD4<Float>(repeating: 0)
}

struct RenderingParameters: sizeable {
    var material: VolumeCubeMaterial.Uniforms = .init()

    var scale: Float = 1.0
    var zScale: Float = 1.0
    var sliceNo: UInt16 = 0
    var sliceMax: UInt16 = 0
    var trimXMin: Float = 0.0
    var trimXMax: Float = 1.0
    var trimYMin: Float = 0.0
    var trimYMax: Float = 1.0
    var trimZMin: Float = 0.0
    var trimZMax: Float = 1.0
    var color: PackedColor = PackedColor()
    var cropLockQuaternions: SIMD4<Float> = SIMD4<Float>(0, 0, 0, 1)
    var clipBoxQuaternion: SIMD4<Float> = SIMD4<Float>(0, 0, 0, 1)
    var clipPlane0: SIMD4<Float> = .zero
    var clipPlane1: SIMD4<Float> = .zero
    var clipPlane2: SIMD4<Float> = .zero
    var cropSliceNo: UInt16 = 0
    var eulerX: Float = 0.0
    var eulerY: Float = 0.0
    var eulerZ: Float = 0.0
    var translationX: Float = 0.0
    var translationY: Float = 0.0
    var viewSize: UInt16 = 0
    var pointX: Float = 0.0
    var pointY: Float = 0.0
    var alphaPower: UInt8 = 1
    var renderingStep: Float = 1.0
    var earlyTerminationThreshold: Float = 0.99
    var adaptiveGradientThreshold: Float = 0.0
    var jitterAmount: Float = 0.0
    var intensityRatio: SIMD4<Float> = SIMD4<Float>(repeating: 1.0)
    var light: Float = 1.0
    var shade: Float = 1.0
    var dicomOrientationRow: SIMD4<Float> = SIMD4<Float>(1, 0, 0, 0)
    var dicomOrientationColumn: SIMD4<Float> = SIMD4<Float>(0, 1, 0, 0)
    var dicomOrientationNormal: SIMD4<Float> = SIMD4<Float>(0, 0, 1, 0)
    var dicomOrientationActive: UInt32 = 0
    var dicomOrientationPadding: SIMD3<UInt32> = SIMD3<UInt32>(repeating: 0)
    var renderingMethod: UInt8 = UInt8(truncatingIfNeeded: VolumeCubeMaterial.Method.dvr.idInt32)
    var backgroundColor: SIMD3<Float> = SIMD3<Float>(repeating: 0.0)

    var padding0: UInt8 = 0
    var padding1: UInt16 = 0

    init() {}
}

extension RenderingParameters {
    mutating func applyVolumeDefaults(dataset: VolumeDataset) {
        zScale = dataset.spacing.z / max(dataset.spacing.x, 1e-6)
        sliceMax = UInt16(clamping: dataset.dimensions.z)
    }
}
