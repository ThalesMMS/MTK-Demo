//
//  DICOMGeometry.swift
//  Isis DICOM Viewer
//
//  Geometria DICOM essencial: mapeia WORLD(mm, LPS) <-> VOXEL(i,j,k) <-> TEX([0,1]^3)
//  Thales Matheus Mendonça Santos - September 2025
//

import simd

/// Geometria DICOM essencial: mapeia WORLD(mm, LPS) <-> VOXEL(i,j,k) <-> TEX([0,1]^3)
struct DICOMGeometry {
    let cols: Int32, rows: Int32, slices: Int32
    let spacingX: Float, spacingY: Float, spacingZ: Float  // mm
    let iopRow: simd_float3    // ImageOrientationPatient (r)
    let iopCol: simd_float3    // (c)
    let ipp0:  simd_float3     // ImagePositionPatient do primeiro slice

    var iopNorm: simd_float3 { simd_normalize(simd_cross(iopRow, iopCol)) }

    /// VOXEL -> WORLD (mm): IPP0 + i*Δx*r + j*Δy*c + k*Δz*n
    var voxelToWorld: simd_float4x4 {
        let Rx = iopRow * spacingX
        let Cy = iopCol * spacingY
        let Nz = iopNorm * spacingZ
        let t  = ipp0
        return simd_float4x4(columns: (
            simd_float4(Rx.x, Cy.x, Nz.x, t.x),
            simd_float4(Rx.y, Cy.y, Nz.y, t.y),
            simd_float4(Rx.z, Cy.z, Nz.z, t.z),
            simd_float4(0,     0,    0,    1)
        ))
    }

    var worldToVoxel: simd_float4x4 { simd_inverse(voxelToWorld) }

    /// VOXEL -> TEX ([0,1]^3): (voxel + 0.5) / dims
    private var voxelToTex: simd_float4x4 {
        let dx = Float(cols), dy = Float(rows), dz = Float(slices)
        let scale = simd_float4x4(diagonal: simd_float4(1/dx, 1/dy, 1/dz, 1))
        let half  = simd_float4x4(columns: (
            simd_float4(1,0,0,0.5/dx),
            simd_float4(0,1,0,0.5/dy),
            simd_float4(0,0,1,0.5/dz),
            simd_float4(0,0,0,1)
        ))
        return half * scale
    }

    /// WORLD -> TEX ([0,1]^3)
    var worldToTex: simd_float4x4 { voxelToTex * worldToVoxel }

    /// Converte um plano definido em WORLD(mm) para TEX ([0,1]^3).
    /// originW, axisUW, axisVW: mm no espaço do paciente (LPS).
    func planeWorldToTex(originW: simd_float3,
                         axisUW: simd_float3,
                         axisVW: simd_float3) -> (originT: simd_float3,
                                                  axisUT: simd_float3,
                                                  axisVT: simd_float3)
    {
        let O = (worldToTex * simd_float4(originW, 1)).xyz
        let U = (worldToTex * simd_float4(originW + axisUW, 1)).xyz - O
        let V = (worldToTex * simd_float4(originW + axisVW, 1)).xyz - O
        return (O, U, V)
    }
}
