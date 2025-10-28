import Foundation
import Metal
import simd

enum VolumePixelFormat {
    case int16Signed
    case int16Unsigned

    var metalPixelFormat: MTLPixelFormat {
        switch self {
        case .int16Signed:
            return .r16Sint
        case .int16Unsigned:
            return .r16Uint
        }
    }

    var bytesPerVoxel: Int {
        switch self {
        case .int16Signed, .int16Unsigned:
            return MemoryLayout<UInt16>.size
        }
    }

    var defaultIntensityRange: ClosedRange<Int32> {
        switch self {
        case .int16Signed:
            let min = Int32(Int16.min)
            let max = Int32(Int16.max)
            return min...max
        case .int16Unsigned:
            let min = Int32(UInt16.min)
            let max = Int32(UInt16.max)
            return min...max
        }
    }
}

struct VolumeDataset {
    var data: Data
    var dimensions: int3
    var spacing: float3
    var pixelFormat: VolumePixelFormat
    var intensityRange: ClosedRange<Int32>
    var orientation: simd_float3x3
    var origin: float3

    init(data: Data,
         dimensions: int3,
         spacing: float3,
         pixelFormat: VolumePixelFormat,
         intensityRange: ClosedRange<Int32>? = nil,
         orientation: simd_float3x3 = matrix_identity_float3x3,
         origin: float3 = .zero) {
        self.data = data
        self.dimensions = dimensions
        self.spacing = spacing
        self.pixelFormat = pixelFormat
        self.intensityRange = intensityRange ?? pixelFormat.defaultIntensityRange
        self.orientation = orientation
        self.origin = origin
    }

    var voxelCount: Int {
        Int(dimensions.x) * Int(dimensions.y) * Int(dimensions.z)
    }

    var scale: float3 {
        float3(spacing.x * Float(dimensions.x),
               spacing.y * Float(dimensions.y),
               spacing.z * Float(dimensions.z))
    }
}
