import SceneKit
import simd

extension SCNNode {
    /// Configura a pose de um plano (SCNPlane) filho do cubo do volume.
    /// originTex, UTex, VTex em coordenadas de textura [0,1]^3 (mesmo espaço do shader).
    /// - Importante: o nó deve ser FILHO de `volume` para herdar a escala anisotrópica.
    func setTransformFromBasisTex(originTex o: simd_float3,
                                  UTex u: simd_float3,
                                  VTex v: simd_float3)
    {
        // Dimensões do retângulo (em [0,1])
        let width  = simd_length(u)
        let height = simd_length(v)

        // Base ortonormal (U,V,N)
        let Uhat = width > 0 ? simd_normalize(u) : simd_float3(1, 0, 0)
        let vOrtho = v - simd_dot(v, Uhat) * Uhat
        let Vhat = simd_length(vOrtho) > 0 ? simd_normalize(vOrtho) : simd_float3(0, 1, 0)
        let Nhat = simd_normalize(simd_cross(Uhat, Vhat))
        let R = simd_float3x3(columns: (Uhat, Vhat, Nhat))

        // Centro do plano em local do cubo [-0.5..+0.5]^3
        let centerLocal = o + 0.5 * u + 0.5 * v - simd_float3(0.5, 0.5, 0.5)

        // Aplicar rotação/posição
        self.simdOrientation = simd_quatf(R)
        self.simdPosition    = centerLocal

        // Largura/altura do SCNPlane no espaço local do cubo
        if let plane = self.geometry as? SCNPlane {
            plane.width  = CGFloat(width)
            plane.height = CGFloat(height)
        }
    }
}


