# GDCM Vendor Drop-In

Static GDCM archives for both iOS device and simulator are already staged here. The folder layout matches the Xcode build settings (`HEADER_SEARCH_PATHS`, `LIBRARY_SEARCH_PATHS`).

```
Vendor/GDCM/
├── include/                 # headers under gdcm-3.3/
└── lib/
    ├── iphoneos/            # arm64 static libs from Release build
    └── iphonesimulator/     # arm64 simulator static libs
```

If you need to rebuild GDCM, rerun:

```bash
cmake -S Vendor/GDCM/src -B Vendor/GDCM/build/ios-device -GXcode \\
  -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_SYSROOT=iphoneos \\
  -DCMAKE_OSX_ARCHITECTURES=arm64 -DBUILD_SHARED_LIBS=OFF \\
  -DGDCM_BUILD_APPLICATIONS=OFF -DGDCM_BUILD_EXAMPLES=OFF -DGDCM_BUILD_TESTING=OFF
cmake --build Vendor/GDCM/build/ios-device --config Release
cmake --install Vendor/GDCM/build/ios-device --config Release --prefix Vendor/GDCM/install/ios-device

cmake -S Vendor/GDCM/src -B Vendor/GDCM/build/ios-sim -GXcode \\
  -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_SYSROOT=iphonesimulator \\
  -DCMAKE_OSX_ARCHITECTURES=arm64 -DBUILD_SHARED_LIBS=OFF \\
  -DGDCM_BUILD_APPLICATIONS=OFF -DGDCM_BUILD_EXAMPLES=OFF -DGDCM_BUILD_TESTING=OFF
cmake --build Vendor/GDCM/build/ios-sim --config Release
cmake --install Vendor/GDCM/build/ios-sim --config Release --prefix Vendor/GDCM/install/ios-sim

rsync -a --delete Vendor/GDCM/install/ios-device/include/ Vendor/GDCM/include/
rsync -a --delete Vendor/GDCM/install/ios-device/lib/ Vendor/GDCM/lib/iphoneos/
rsync -a --delete Vendor/GDCM/install/ios-sim/lib/ Vendor/GDCM/lib/iphonesimulator/
```

Link the required `.a` archives (e.g. `libgdcmCommon`, `libgdcmMSFF`, `libgdcmIOD`, `libgdcmDICT`, `libgdcmDSED`, `libgdcmzlib`, `libgdcmexpat`, `libgdcmopenjp2`, `libgdcmcharls`, `libsocketxx`, `libgdcmuuid`, and the JPEG variants) in the Xcode target. All archives are built as Release static libraries for arm64.
