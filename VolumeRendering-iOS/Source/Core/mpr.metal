//
//  mpr.metal
//  Isis DICOM Viewer
//
//  MVP MPR Shader (Metal)
//  - Renderiza um plano de reamostragem dentro do volume (dicom 3D).
//  - Suporta MPR fino e thick slab com MIP/MinIP/Mean.
//  - Normalização HU -> [0,1] via min/max (mesma ideia do VR).
//  Thales Matheus Mendonça Santos - September 2025
//

#include <metal_stdlib>
#include "shared.metal"   // traz NodeBuffer, SCNSceneBuffer, samplers e Utils

using namespace metal;

struct MPRUniforms {
    int   voxelMinValue;
    int   voxelMaxValue;
    int   datasetMinValue;
    int   datasetMaxValue;
    int   blendMode;     // 0=single, 1=MIP, 2=MinIP, 3=Mean
    int   numSteps;      // >=1; 1 => MPR fino
    float slabHalf;      // metade da espessura em [0,1]
    float3 _pad0;

    float3 planeOrigin;  // origem do plano em [0,1]^3
    float  _pad1;
    float3 planeX;       // eixo U do plano (tamanho = largura em [0,1])
    float  _pad2;
    float3 planeY;       // eixo V do plano (tamanho = altura em [0,1])
    float  overlayOpacity;
    int   useTFMpr;      // 0=grayscale, 1=usar TF 1D
    int   overlayEnabled;
    int   overlayChannel;
    float3 overlayColor;
    float  _pad7;
};

struct VertexIn {
    float3 position  [[attribute(SCNVertexSemanticPosition)]];
    float3 normal    [[attribute(SCNVertexSemanticNormal)]];
    float4 color     [[attribute(SCNVertexSemanticColor)]];
    float2 uv        [[attribute(SCNVertexSemanticTexcoord0)]];
};

struct VSOut {
    float4 position [[position]];
    float2 uv;
};

vertex VSOut mpr_vertex(VertexIn in                   [[ stage_in ]],
                        constant NodeBuffer& scn_node [[ buffer(1) ]]) {
    VSOut out;
    out.position = Unity::ObjectToClipPos(float4(in.position, 1.0f), scn_node);
    out.uv = in.uv;
    return out;
}

inline float sampleDensity01(texture3d<short, access::sample> volume, float3 p,
                             short minV, short maxV,
                             thread short &huOut) {
    short hu = volume.sample(sampler3d, p).r;
    huOut = hu;
    return Util::normalize(hu, minV, maxV); // HU -> [0,1]
}

fragment float4 mpr_fragment(VSOut in                                       [[stage_in]],
                             constant SCNSceneBuffer& scn_frame              [[buffer(0)]],
                             constant NodeBuffer& scn_node                   [[buffer(1)]],
                             constant MPRUniforms& U                         [[buffer(4)]],
                             texture3d<short, access::sample> volume         [[texture(0)]],
                             texture2d<float, access::sample> transferColor  [[texture(3)]]) {

    // Coord do plano no volume (normalizada)
    float3 Pw = U.planeOrigin + in.uv.x * U.planeX + in.uv.y * U.planeY;

    // Fora do volume? (com pequena margem)
    if (any(Pw < -1e-6) || any(Pw > 1.0 + 1e-6)) {
        return float4(0,0,0,1);
    }

    const short windowMin = short(U.voxelMinValue);
    const short windowMax = short(U.voxelMaxValue);
    const short dataMin = short(U.datasetMinValue);
    const short dataMax = short(U.datasetMaxValue);

    if (U.numSteps <= 1 || U.slabHalf <= 0.0f || U.blendMode == 0) {
        short huSample = 0;
        float densityWindow = clamp(sampleDensity01(volume, Pw, windowMin, windowMax, huSample),
                                    0.0f, 1.0f);
        float windowIntensity = densityWindow;
        if (U.useTFMpr != 0 && transferColor.get_width() > 0) {
            float densityDataset = clamp(Util::normalize(huSample, dataMin, dataMax), 0.0f, 1.0f);
            float4 tfColor = VR::getTfColour(transferColor, densityDataset);
            if (length(tfColor.rgb) > 0.001f) {
                tfColor.rgb *= windowIntensity;
                return float4(clamp(tfColor.rgb, float3(0.0f), float3(1.0f)), 1.0f);
            }
        }
        return float4(densityWindow, densityWindow, densityWindow, 1.0f);
    }

    // Thick slab: percorre ao longo da normal do plano
    float3 N = normalize(cross(U.planeX, U.planeY));
    int steps = max(2, U.numSteps);
    int halfSteps = (steps - 1) / 2;
    float stepN = (2.0f * U.slabHalf) / float(steps - 1);

    float vmaxWindow = 0.0f;
    float vmaxDataset = 0.0f;
    float vminWindow = 1.0f;
    float vminDataset = 1.0f;
    float vaccWindow = 0.0f;
    float vaccDataset = 0.0f;
    int   cnt  = 0;

    for (int i = -halfSteps; i <= halfSteps; ++i) {
        float3 Pi = Pw + float(i) * stepN * N;
        if (any(Pi < 0.0f) || any(Pi > 1.0f)) continue;

        short huSample = 0;
        float sampleWindow = clamp(sampleDensity01(volume, Pi, windowMin, windowMax, huSample),
                                   0.0f, 1.0f);
        float sampleDataset = clamp(Util::normalize(huSample, dataMin, dataMax), 0.0f, 1.0f);

        if (sampleWindow > vmaxWindow) {
            vmaxWindow = sampleWindow;
            vmaxDataset = sampleDataset;
        }
        if (sampleWindow < vminWindow) {
            vminWindow = sampleWindow;
            vminDataset = sampleDataset;
        }
        vaccWindow += sampleWindow;
        vaccDataset += sampleDataset;
        cnt++;
    }

    bool hit = (cnt > 0);

    float valueWindow = 0.0f;
    float valueDataset = 0.0f;
    switch (U.blendMode) {
        case 1: // MIP
            valueWindow = hit ? vmaxWindow : 0.0f;
            valueDataset = hit ? vmaxDataset : 0.0f;
            break;
        case 2: // MinIP
            valueWindow = hit ? vminWindow : 0.0f;
            valueDataset = hit ? vminDataset : 0.0f;
            break;
        case 3: // Mean
            valueWindow = (cnt > 0) ? (vaccWindow / float(cnt)) : 0.0f;
            valueDataset = (cnt > 0) ? (vaccDataset / float(cnt)) : 0.0f;
            break;
        default: {
            short huSample = 0;
            valueWindow = clamp(sampleDensity01(volume, Pw, windowMin, windowMax, huSample),
                                0.0f, 1.0f);
            valueDataset = clamp(Util::normalize(huSample, dataMin, dataMax), 0.0f, 1.0f);
            break;
        }
    }

    float windowIntensity = valueWindow;
    float4 color = float4(windowIntensity, windowIntensity, windowIntensity, 1.0f);
    if (U.useTFMpr != 0 && transferColor.get_width() > 0) {
        float4 tfColor = VR::getTfColour(transferColor, valueDataset);
        if (length(tfColor.rgb) > 0.001f) {
            tfColor.rgb *= windowIntensity;
            color = float4(clamp(tfColor.rgb, float3(0.0f), float3(1.0f)), 1.0f);
        }
    }

    return color;
}

// MARK: - Compute kernel (Fase 6 M6.1)

struct MPRArguments {
    texture3d<short, access::sample> volume        [[id(0)]];
    constant MPRUniforms &params                   [[id(1)]];
    texture2d<float, access::write> outputTexture  [[id(2)]];
    texture2d<float, access::sample> transferColor [[id(3)]];
    sampler linearSampler                          [[id(4)]];
    texture3d<float, access::sample> maskTexture   [[id(5)]];
};

inline float sampleDensity(texture3d<short, access::sample> volume,
                            sampler linearSampler,
                            float3 coord,
                            short minV,
                            short maxV,
                            thread short &huOut) {
    short hu = volume.sample(linearSampler, coord).r;
    huOut = hu;
    return Util::normalize(hu, minV, maxV);
}

inline void mpr_renderPixel(texture3d<short, access::sample> volume,
                            sampler linearSampler,
                            texture2d<float, access::write> outputTexture,
                            texture2d<float, access::sample> transferColor,
                            texture3d<float, access::sample> maskTexture,
                            constant MPRUniforms &params,
                            uint2 gid) {
    uint width = outputTexture.get_width();
    uint height = outputTexture.get_height();

    if (gid.x >= width || gid.y >= height) {
        return;
    }

    const float2 uv = (float2(gid) + 0.5f) / float2(width, height);
    const float3 Pw = params.planeOrigin + uv.x * params.planeX + uv.y * params.planeY;

    if (any(Pw < -1e-6f) || any(Pw > 1.0f + 1e-6f)) {
        outputTexture.write(float4(0.0f, 0.0f, 0.0f, 1.0f), gid);
        return;
    }

    const short windowMin = short(params.voxelMinValue);
    const short windowMax = short(params.voxelMaxValue);
    const short dataMin = short(params.datasetMinValue);
    const short dataMax = short(params.datasetMaxValue);

    float valueWindow = 0.0f;
    float valueDataset = 0.0f;
    bool useSlab = (params.numSteps > 1) && (params.slabHalf > 0.0f) && (params.blendMode != 0);

    if (!useSlab) {
        short huSample = 0;
        valueWindow = clamp(sampleDensity(volume,
                                          linearSampler,
                                          Pw,
                                          windowMin,
                                          windowMax,
                                          huSample),
                            0.0f, 1.0f);
        valueDataset = clamp(Util::normalize(huSample, dataMin, dataMax), 0.0f, 1.0f);
    } else {
        const int steps = max(2, int(params.numSteps));
        const int halfSteps = (steps - 1) / 2;
        const float3 normal = normalize(cross(params.planeX, params.planeY));
        const float stepN = (steps > 1) ? ((2.0f * params.slabHalf) / float(steps - 1)) : 0.0f;

        float vmaxWindow = 0.0f;
        float vmaxDataset = 0.0f;
        float vminWindow = 1.0f;
        float vminDataset = 1.0f;
        float vaccWindow = 0.0f;
        float vaccDataset = 0.0f;
        int count = 0;

        for (int i = -halfSteps; i <= halfSteps; ++i) {
            float3 Pi = Pw + float(i) * stepN * normal;
            if (any(Pi < 0.0f) || any(Pi > 1.0f)) {
                continue;
            }

            short huSample = 0;
            float sampleWindow = clamp(sampleDensity(volume,
                                                     linearSampler,
                                                     Pi,
                                                     windowMin,
                                                     windowMax,
                                                     huSample),
                                       0.0f, 1.0f);
            float sampleDataset = clamp(Util::normalize(huSample, dataMin, dataMax), 0.0f, 1.0f);

            if (sampleWindow > vmaxWindow) {
                vmaxWindow = sampleWindow;
                vmaxDataset = sampleDataset;
            }
            if (sampleWindow < vminWindow) {
                vminWindow = sampleWindow;
                vminDataset = sampleDataset;
            }
            vaccWindow += sampleWindow;
            vaccDataset += sampleDataset;
            count++;
        }

        switch (params.blendMode) {
        case 1:
            valueWindow = (count > 0) ? vmaxWindow : 0.0f;
            valueDataset = (count > 0) ? vmaxDataset : 0.0f;
            break;
        case 2:
            valueWindow = (count > 0) ? vminWindow : 0.0f;
            valueDataset = (count > 0) ? vminDataset : 0.0f;
            break;
        case 3:
            valueWindow = (count > 0) ? (vaccWindow / float(count)) : 0.0f;
            valueDataset = (count > 0) ? (vaccDataset / float(count)) : 0.0f;
            break;
        default: {
            short huSample = 0;
            valueWindow = clamp(sampleDensity(volume,
                                              linearSampler,
                                              Pw,
                                              windowMin,
                                              windowMax,
                                              huSample),
                                0.0f, 1.0f);
            valueDataset = clamp(Util::normalize(huSample, dataMin, dataMax), 0.0f, 1.0f);
            break;
        }
        }
    }

    float windowIntensity = valueWindow;
    float4 color = float4(windowIntensity, windowIntensity, windowIntensity, 1.0f);

    if (params.useTFMpr != 0 && transferColor.get_width() > 0) {
        float4 tfColor = VR::getTfColour(transferColor, valueDataset);
        if (length(tfColor.rgb) > 0.001f) {
            tfColor.rgb *= windowIntensity;
            color = float4(clamp(tfColor.rgb, float3(0.0f), float3(1.0f)), 1.0f);
        }
    }

    if (params.overlayEnabled != 0 && params.overlayOpacity > 0.0f && maskTexture.get_width() > 0) {
        float maskSample = maskTexture.sample(linearSampler, Pw).r;
        if (maskSample > 0.001f) {
            float opacity = clamp(maskSample * params.overlayOpacity, 0.0f, 1.0f);
            float3 overlayColor = params.overlayColor;
            color.rgb = mix(color.rgb, overlayColor, opacity);
        }
    }

    outputTexture.write(color, gid);
}

kernel void mpr_compute(constant MPRArguments &args [[buffer(0)]],
                        uint2 gid [[thread_position_in_grid]]) {
    mpr_renderPixel(args.volume,
                    args.linearSampler,
                    args.outputTexture,
                    args.transferColor,
                    args.maskTexture,
                    args.params,
                    gid);
}

kernel void mpr_compute_legacy(texture3d<short, access::sample> volume [[texture(0)]],
                               constant MPRUniforms &params [[buffer(0)]],
                               texture2d<float, access::write> outputTexture [[texture(1)]],
                               texture2d<float, access::sample> transferColor [[texture(2)]],
                               sampler linearSampler [[sampler(0)]],
                               texture3d<float, access::sample> maskTexture [[texture(3)]],
                               uint2 gid [[thread_position_in_grid]]) {
    mpr_renderPixel(volume,
                    linearSampler,
                    outputTexture,
                    transferColor,
                    maskTexture,
                    params,
                    gid);
}
