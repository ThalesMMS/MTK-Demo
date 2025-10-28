#include <metal_stdlib>
#include "shared.metal"

using namespace metal;

struct VolumeUniforms {
    int isLightingOn;
    int isBackwardOn;
    int method;
    int renderingQuality;
    int voxelMinValue;
    int voxelMaxValue;
    int datasetMinValue;
    int datasetMaxValue;
    float densityFloor;
    float densityCeil;
    int gateHuMin;
    int gateHuMax;
    int useHuGate;
    int dimX;
    int dimY;
    int dimZ;
    int useTFProj;
    int _pad0;
    int _pad1;
    int _pad2;
};

struct PackedColor {
    float4 ch1;
    float4 ch2;
    float4 ch3;
    float4 ch4;
};

struct RenderingParameters {
    VolumeUniforms material;
    float scale;
    float zScale;
    ushort sliceNo;
    ushort sliceMax;
    float trimXMin;
    float trimXMax;
    float trimYMin;
    float trimYMax;
    float trimZMin;
    float trimZMax;
    PackedColor color;
    float4 cropLockQuaternions;
    float4 clipBoxQuaternion;
    float4 clipPlane0;
    float4 clipPlane1;
    float4 clipPlane2;
    ushort cropSliceNo;
    float eulerX;
    float eulerY;
    float eulerZ;
    float translationX;
    float translationY;
    ushort viewSize;
    float pointX;
    float pointY;
    uchar alphaPower;
    float renderingStep;
    float earlyTerminationThreshold;
    float adaptiveGradientThreshold;
    float jitterAmount;
    float4 intensityRatio;
    float light;
    float shade;
    float4 dicomOrientationRow;
    float4 dicomOrientationColumn;
    float4 dicomOrientationNormal;
    uint dicomOrientationActive;
    uint3 dicomOrientationPadding;
    uchar renderingMethod;
    float3 backgroundColor;
    uchar padding0;
    ushort padding1;
};

struct CameraUniforms {
    float4x4 modelMatrix;
    float4x4 inverseModelMatrix;
    float4x4 inverseViewProjectionMatrix;
    float3   cameraPositionLocal;
    uint     frameIndex;
    uint     padding;
};

struct RenderingArguments {
    texture3d<short, access::sample> volumeTexture [[id(0)]];
    constant RenderingParameters &params           [[id(1)]];
    texture2d<float, access::write> outputTexture  [[id(2)]];
    device float *toneBufferCh1                    [[id(3)]];
    device float *toneBufferCh2                    [[id(4)]];
    device float *toneBufferCh3                    [[id(5)]];
    device float *toneBufferCh4                    [[id(6)]];
    constant ushort &optionValue                   [[id(7)]];
    constant float4 &quaternion                    [[id(8)]];
    constant ushort &targetViewSize                [[id(9)]];
    sampler volumeSampler                          [[id(10)]];
    constant ushort &pointSetCount                 [[id(11)]];
    constant ushort &pointSelectedIndex            [[id(12)]];
    constant float3 *pointSet                      [[id(13)]];
    device uint8_t *legacyOutputBuffer             [[id(14)]];
    texture2d<float, access::sample> transferTextureCh1 [[id(15)]];
    texture2d<float, access::sample> transferTextureCh2 [[id(16)]];
    texture2d<float, access::sample> transferTextureCh3 [[id(17)]];
    texture2d<float, access::sample> transferTextureCh4 [[id(18)]];
};

namespace VolumeCompute {

constant uint kToneSampleCount = 2551u;
constant float kToneLookupScale = 2550.0f;
constant ushort OPTION_ADAPTIVE = 1u << 2;
constant ushort OPTION_DEBUG_DENSITY = 1u << 3;

static inline float3 computeRayDirection(float3 cameraLocal01,
                                         float3 pixelLocal01)
{
    return normalize(pixelLocal01 - cameraLocal01);
}

static inline float4 sampleTransfer(texture2d<float, access::sample> transfer,
                                    float density,
                                    bool useTransfer)
{
    if (!useTransfer) {
        return float4(density, density, density, density);
    }
    return VR::getTfColour(transfer, density);
}

static inline float sampleTone(device float *toneBuffer,
                               float density)
{
    if (toneBuffer == nullptr) {
        return 1.0f;
    }
    float lookup = clamp(density * kToneLookupScale, 0.0f, float(kToneSampleCount - 1));
    uint index = (uint)round(lookup);
    return clamp(toneBuffer[index], 0.0f, 1.0f);
}

static inline bool gateDensity(float density,
                               constant RenderingParameters& params,
                               short hu)
{
    constant VolumeUniforms& uniforms = params.material;
    if (uniforms.method == 2 || uniforms.method == 3 || uniforms.method == 4) {
        if (density < uniforms.densityFloor || density > uniforms.densityCeil) {
            return true;
        }
        if (uniforms.useHuGate != 0 &&
            (hu < uniforms.gateHuMin || hu > uniforms.gateHuMax)) {
            return true;
        }
    }
    return false;
}

static inline float4 sampleChannel(texture2d<float, access::sample> transfer,
                                   device float *toneBuffer,
                                   float density,
                                   bool useTransfer,
                                   float weight)
{
    if (weight <= 0.0001f) {
        return float4(0.0f);
    }
    float tone = sampleTone(toneBuffer, density);
    float4 colour = sampleTransfer(transfer, density, useTransfer);
    colour.rgb *= weight;
    colour.a = clamp(colour.a * weight * tone, 0.0f, 1.0f);
    return clamp(colour, float4(0.0f), float4(1.0f));
}

static inline float4 compositeChannels(constant RenderingArguments& args,
                                       constant RenderingParameters& params,
                                       float density,
                                       bool useTransfer)
{
    const float4 weights = params.intensityRatio;

    float4 channels[4];
    channels[0] = sampleChannel(args.transferTextureCh1, args.toneBufferCh1, density, useTransfer, weights.x);
    channels[1] = sampleChannel(args.transferTextureCh2, args.toneBufferCh2, density, useTransfer, weights.y);
    channels[2] = sampleChannel(args.transferTextureCh3, args.toneBufferCh3, density, useTransfer, weights.z);
    channels[3] = sampleChannel(args.transferTextureCh4, args.toneBufferCh4, density, useTransfer, weights.w);

    float alpha = 0.0f;
    float3 premult = float3(0.0f);

    for (uint i = 0; i < 4; ++i) {
        const float channelAlpha = channels[i].a;
        if (channelAlpha <= 0.0f) {
            continue;
        }
        const float3 channelColor = clamp(channels[i].rgb, float3(0.0f), float3(1.0f));
        premult += channelColor * channelAlpha;
        alpha = clamp(alpha + channelAlpha, 0.0f, 1.0f);
        if (alpha >= 0.9999f) {
            alpha = 0.9999f;
        }
    }

    if (alpha <= 1e-5f) {
        return float4(0.0f);
    }

    float3 colour = premult / max(alpha, 1e-5f);
    colour = clamp(colour, float3(0.0f), float3(1.0f));
    return float4(colour, clamp(alpha, 0.0f, 1.0f));
}

struct ClipSampleContext {
    float3 centeredCoordinate;
    float3 orientedCoordinate;
};

static inline float4 normalizeQuaternionSafe(float4 q) {
    float lenSq = dot(q, q);
    if (lenSq <= 1.0e-8f) {
        return float4(0.0f, 0.0f, 0.0f, 1.0f);
    }
    float invLen = rsqrt(lenSq);
    return q * invLen;
}

static inline ClipSampleContext prepareClipSample(float3 texCoordinate,
                                                  constant RenderingParameters &params)
{
    ClipSampleContext context;
    context.centeredCoordinate = texCoordinate - float3(0.5f);
    float4 clipQuaternion = normalizeQuaternionSafe(params.clipBoxQuaternion);
    float4 clipQuaternionInv = quatInv(clipQuaternion);
    float3 rotated = quatMul(clipQuaternionInv, context.centeredCoordinate);
    context.orientedCoordinate = rotated + float3(0.5f);
    return context;
}

static inline bool isOutsideTrimBounds(float3 orientedCoordinate,
                                       constant RenderingParameters &params)
{
    return orientedCoordinate.x < params.trimXMin || orientedCoordinate.x > params.trimXMax ||
           orientedCoordinate.y < params.trimYMin || orientedCoordinate.y > params.trimYMax ||
           orientedCoordinate.z < params.trimZMin || orientedCoordinate.z > params.trimZMax;
}

static inline bool isClippedByPlanes(thread const ClipSampleContext &context,
                                     constant RenderingParameters &params)
{
    float4 planes[3] = { params.clipPlane0, params.clipPlane1, params.clipPlane2 };
    for (uint i = 0; i < 3; ++i) {
        float3 normal = planes[i].xyz;
        if (all(normal == float3(0.0f))) {
            continue;
        }
        float signedDistance = dot(context.centeredCoordinate, normal) + planes[i].w;
        if (signedDistance > 0.0f) {
            return true;
        }
    }
    return false;
}

} // namespace VolumeCompute

kernel void volume_compute(constant RenderingArguments& args [[buffer(0)]],
                           constant CameraUniforms& camera          [[buffer(1)]],
                           uint2 gid [[thread_position_in_grid]])
{
    const uint width  = args.outputTexture.get_width();
    const uint height = args.outputTexture.get_height();

    if (gid.x >= width || gid.y >= height) {
        return;
    }

#if DEBUG
    const bool debugDensityEnabled = ((args.optionValue & VolumeCompute::OPTION_DEBUG_DENSITY) != 0);
    const uint2 debugCenterPixel = uint2(width / 2, height / 2);
    bool debugDensitySentinelWritten = false;
    bool debugPreBlendSentinelWritten = false;
    if (debugDensityEnabled) {
        args.outputTexture.write(float4(1.0f, 0.0f, 1.0f, 1.0f), gid); // Kernel start
    }
#endif

    // Coordenadas de dispositivo normalizadas (-1 .. +1).
    const float x = (float(gid.x) + 0.5f) / float(width);
    const float y = (float(height - 1 - gid.y) + 0.5f) / float(height);
    const float2 ndc = float2(x * 2.0f - 1.0f,
                              y * 2.0f - 1.0f);

    const float4 clipNear = float4(ndc, 0.0f, 1.0f);
    const float4 clipFar  = float4(ndc, 1.0f, 1.0f);

    float4 worldNear = camera.inverseViewProjectionMatrix * clipNear;
    float4 worldFar  = camera.inverseViewProjectionMatrix * clipFar;
    worldNear /= worldNear.w;
    worldFar  /= worldFar.w;

    float4 localFar4  = camera.inverseModelMatrix * worldFar;
    float3 localFar   = (localFar4.xyz / localFar4.w) + float3(0.5f);

    float3 cameraLocal01 = camera.cameraPositionLocal + float3(0.5f);
    float3 rayDir = VolumeCompute::computeRayDirection(cameraLocal01, localFar);

    float2 intersection = VR::intersectAABB(cameraLocal01,
                                            rayDir,
                                            float3(0.0f),
                                            float3(1.0f));
    float tEnter = max(intersection.x, 0.0f);
    float tExit  = intersection.y;

#if DEBUG
    if (debugDensityEnabled) {
        args.outputTexture.write(float4(0.0f, 1.0f, 0.0f, 1.0f), gid); // After intersectBox
    }
#endif

    if (!(tExit > tEnter)) {
        args.outputTexture.write(float4(0.0f), gid);
        return;
    }

    float3 startPos = cameraLocal01 + rayDir * tEnter;
    float3 endPos   = cameraLocal01 + rayDir * tExit;

    constant VolumeUniforms& material = args.params.material;
    const float opacityThreshold = clamp(args.params.earlyTerminationThreshold, 0.0f, 0.9999f);

    const int maxSteps = max(material.renderingQuality, 1);
    const float baseStepForJitter = sqrt(3.0f) / float(maxSteps);
    const float jitterParam = clamp(args.params.jitterAmount, 0.0f, 1.0f);
    if (jitterParam > 0.0f) {
        float jitterSeed = dot(float2(gid), float2(12.9898f, 78.233f)) + float(camera.frameIndex);
        float jitterNoise = fract(sin(jitterSeed) * 43758.5453f);
        float availableDistance = max(1.0e-5f, abs(tExit - tEnter));
        float jitterDistance = min(baseStepForJitter * jitterParam * jitterNoise,
                                   availableDistance * 0.99f);

        if (material.isBackwardOn != 0) {
            tExit = max(tExit - jitterDistance, tEnter);
        } else {
            tEnter = min(tEnter + jitterDistance, tExit);
        }

        startPos = cameraLocal01 + rayDir * tEnter;
        endPos   = cameraLocal01 + rayDir * tExit;
    }

    VR::RayInfo ray;
    if (material.isBackwardOn != 0) {
        ray.startPosition = endPos;
        ray.endPosition   = startPos;
        ray.direction     = -rayDir;
        ray.aabbIntersection = float2(tExit, tEnter);
    } else {
        ray.startPosition = startPos;
        ray.endPosition   = endPos;
        ray.direction     = rayDir;
        ray.aabbIntersection = float2(tEnter, tExit);
    }

    VR::RaymarchInfo march = VR::initRayMarch(ray, maxSteps);
    if (march.numSteps <= 0) {
#if DEBUG
        if (debugDensityEnabled) {
            args.outputTexture.write(float4(1.0f, 0.0f, 0.0f, 1.0f), gid); // Ray march init failure
        }
#endif
        args.outputTexture.write(float4(0.0f), gid);
        return;
    }

    float4 accumulator = float4(0.0f);
    float debugMaxDensity = 0.0f;
    float debugMaxSampleAlpha = 0.0f;
    const float3 dimension = float3(material.dimX,
                                    material.dimY,
                                    material.dimZ);

    const float baseStep = max(march.stepSize, 1.0e-5f);
    int zeroCount = 0;
    constexpr int kZeroRun = 4;
    constexpr int kZeroSkip = 3;
    const bool adaptiveEnabled = ((args.optionValue & VolumeCompute::OPTION_ADAPTIVE) != 0) &&
                                 (args.params.adaptiveGradientThreshold > 1.0e-4f);
    const float totalDistance = max(length(ray.endPosition - ray.startPosition), baseStep);
    float distanceTravelled = 0.0f;
    int iteration = 0;
    const int maxIterations = max(march.numSteps * 4, march.numSteps + 16);

    while (distanceTravelled < totalDistance && iteration < maxIterations) {
        float stepDistance = baseStep;
        const float t = (totalDistance > 0.0f)
            ? distanceTravelled / totalDistance
            : 0.0f;
        const float3 samplePos = Util::lerp(ray.startPosition,
                                            ray.endPosition,
                                            t);

        if (samplePos.x < 0.0f || samplePos.x > 1.0f ||
            samplePos.y < 0.0f || samplePos.y > 1.0f ||
            samplePos.z < 0.0f || samplePos.z > 1.0f) {
            break;
        }

        VolumeCompute::ClipSampleContext clipContext = VolumeCompute::prepareClipSample(samplePos, args.params);
        if (VolumeCompute::isOutsideTrimBounds(clipContext.orientedCoordinate, args.params) ||
            VolumeCompute::isClippedByPlanes(clipContext, args.params)) {
            distanceTravelled += stepDistance;
            iteration++;
            continue;
        }

        const short hu = VR::getDensity(args.volumeTexture, samplePos);
        const short windowMin = (short)material.voxelMinValue;
        const short windowMax = (short)material.voxelMaxValue;
        const short dataMin = (short)material.datasetMinValue;
        const short dataMax = (short)material.datasetMaxValue;

        float densityWindow = Util::normalize(hu, windowMin, windowMax);
        float densityDataset = Util::normalize(hu, dataMin, dataMax);

        densityWindow = clamp(densityWindow, 0.0f, 1.0f);
        densityDataset = clamp(densityDataset, 0.0f, 1.0f);

#if DEBUG
        if (debugDensityEnabled && !debugDensitySentinelWritten) {
            args.outputTexture.write(float4(0.0f, 0.5f, 1.0f, 1.0f), gid); // After VR::getDensity
            debugDensitySentinelWritten = true;
        }
        if (debugDensityEnabled && all(gid == debugCenterPixel)) {
            printf("[DensityDebug] iter=%u tEnter=%.6f tExit=%.6f numSteps=%d hu=%d window=%.6f dataset=%.6f\n",
                   uint(iteration),
                   tEnter,
                   tExit,
                   march.numSteps,
                   int(hu),
                   densityWindow,
                   densityDataset);
        }
#endif

        debugMaxDensity = max(debugMaxDensity, densityWindow);

        if (VolumeCompute::gateDensity(densityWindow, args.params, hu)) {
            distanceTravelled += stepDistance;
            iteration++;
            continue;
        }

        float3 gradient = float3(0.0f);
        if (adaptiveEnabled || material.isLightingOn != 0) {
            gradient = VR::calGradient(args.volumeTexture, samplePos, dimension);
        }

        if (adaptiveEnabled) {
            float gradMagnitude = length(gradient);
            float threshold = max(args.params.adaptiveGradientThreshold, 1.0e-4f);
            float normalizedGradient = clamp(gradMagnitude / threshold, 0.0f, 1.0f);
            float adaptFactor = mix(2.0f, 0.5f, normalizedGradient);
            stepDistance = baseStep * adaptFactor;
        }

        const bool useTransfer = (material.useTFProj != 0 || material.method == 1);
        const float densityForColour = useTransfer ? densityDataset : densityWindow;
        const float windowIntensity = densityWindow;
        float4 sampleColour = VolumeCompute::compositeChannels(args,
                                                               args.params,
                                                               densityForColour,
                                                               useTransfer);

        if (useTransfer) {
            sampleColour.rgb *= windowIntensity;
        }

        if (material.isLightingOn != 0) {
            float lengthSq = dot(gradient, gradient);
            float3 normal = lengthSq > 1.0e-6f ? normalize(gradient) : float3(0.0f);
            float3 eyeDir = (material.isBackwardOn != 0) ? ray.direction : -ray.direction;
            float3 lightDir = normalize(cameraLocal01 - samplePos);
            sampleColour.rgb = Util::calculateLighting(sampleColour.rgb,
                                                       normal,
                                                       lightDir,
                                                       eyeDir,
                                                       0.3f);
        }

        const float densityFloor = clamp(args.params.material.densityFloor, 0.0f, 1.0f);
        if (densityWindow <= densityFloor) {
            sampleColour.a = 0.0f;
        }

        if (useTransfer) {
            sampleColour.a *= densityWindow;
        } else {
            sampleColour.a = densityWindow;
        }

        debugMaxSampleAlpha = max(debugMaxSampleAlpha, clamp(sampleColour.a, 0.0f, 1.0f));

        if (sampleColour.a < 0.001f) {
            zeroCount++;
            if (zeroCount >= kZeroRun) {
                distanceTravelled += baseStep * float(kZeroSkip);
                distanceTravelled = min(distanceTravelled, totalDistance);
                zeroCount = 0;
            }
            distanceTravelled += stepDistance;
            distanceTravelled = min(distanceTravelled, totalDistance);
            iteration++;
            continue;
        }
        zeroCount = 0;

        const float alpha = clamp(sampleColour.a, 0.0f, 1.0f);
        const float3 premultColor = sampleColour.rgb * alpha;

#if DEBUG
        if (debugDensityEnabled && !debugPreBlendSentinelWritten) {
            args.outputTexture.write(float4(1.0f, 0.5f, 0.0f, 1.0f), gid); // Before blend FTB
            debugPreBlendSentinelWritten = true;
        }
#endif

        if (material.isBackwardOn != 0) {
            const float oneMinusSampleAlpha = 1.0f - alpha;
            accumulator.rgb = premultColor + oneMinusSampleAlpha * accumulator.rgb;
            accumulator.a = alpha + oneMinusSampleAlpha * accumulator.a;
        } else {
            const float oneMinusAccumAlpha = 1.0f - accumulator.a;
            accumulator.rgb += premultColor * oneMinusAccumAlpha;
            accumulator.a += alpha * oneMinusAccumAlpha;
        }

        if (accumulator.a > opacityThreshold) {
            break;
        }

        distanceTravelled += stepDistance;
        distanceTravelled = min(distanceTravelled, totalDistance);
        iteration++;
    }

    accumulator = clamp(accumulator, float4(0.0f), float4(1.0f));

    if ((args.optionValue & VolumeCompute::OPTION_DEBUG_DENSITY) != 0) {
        float3 debugRGB = float3(debugMaxDensity, debugMaxSampleAlpha, accumulator.a);
        args.outputTexture.write(float4(debugRGB, 1.0f), gid);
        return;
    }
    args.outputTexture.write(accumulator, gid);
}
