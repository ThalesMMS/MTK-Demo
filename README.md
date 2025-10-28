# MTK Demo
This repository is a demonstration of the [MTK](https://github.com/ThalesMMS/MTK) library, compatible with iOS, iPadOS and macOS, combining SceneKit and Metal.

## Highlights
- Direct volume rendering, surface rendering, and projection modes (MIP, MinIP, AIP) powered by MTK helpers
- Tri-planar reconstruction with shared transfer functions and oblique controls
- GDCM-driven DICOM import for zipped series at runtime

### DICOM via GDCM
1. Build GDCM for iOS (static libs or an XCFramework). A simple path is to use CMake with the iOS toolchain or to reuse prebuilt binaries from your toolchain.
2. Copy the resulting headers and libraries into `Vendor/GDCM/include` and `Vendor/GDCM/lib` respectively. The Xcode target already adds these locations to the header and library search paths.
3. Add the required static libraries (for example `libgdcmCommon`, `libgdcmMSFF`, `libexpat`, `libz`, `libopenjp2`, `libcharls`) to **Link Binary With Libraries**. If you use an `.xcframework`, drop it in the folder and drag it into Xcode.
4. Run the app, tap **Import DICOM**, and select a `.zip`, folder, or file representing a series. When GDCM is not linked, the importer gracefully reports that the native loader is unavailable.

## Quick Start
1. Open `VolumeRendering-iOS.xcodeproj` in Xcode and target iOS 15 or later.
2. Supply DICOM data as zipped `.dcm` series.

## Credits
- Forked from [VolumeRendering-in-iOS](https://github.com/eunwonki/Volume-Rendering-In-iOS).
- Inspired by [Unity Volume Rendering](https://github.com/mlavik1/UnityVolumeRendering).
- New code is released under the Apache License 2.0 (`LICENSE`).
- Thales Matheus Mendonça Santos — implemented MinIP and average intensity projection, refined DVR pipelines, and implemented the tri-planar oblique MPR workflow with GDCM-based DICOM integration. Currently migrating to depend on the [MTK library](https://github.com/ThalesMMS/MTK).
