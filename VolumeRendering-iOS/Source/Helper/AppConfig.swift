//
//  AppConfig.swift
//  VolumeRendering-iOS
//
//  Centralizes demo-only flags until MTK configuration objects replace these helpers.
//  Thales Matheus Mendonça Santos — October 2025
//

import Foundation

/// Lightweight configuration holder mirroring the macOS target expectations.
/// Exposes the debug toggle used by infra components (e.g. ArgumentEncoderManager).
enum AppConfig {
    #if DEBUG
    static let IS_DEBUG_MODE = true
    #else
    static let IS_DEBUG_MODE = false
    #endif

    /// Default GPU histogram bin count kept for compatibility with the legacy demo shaders.
    static let HISTOGRAM_BIN_COUNT: Int = 512

    /// Diagnostic toggle: when true, compute kernel outputs normalized density instead of shaded colour.
    /// Leave disabled by default so production builds render the shaded output unless diagnostics are explicitly requested.
    static let ENABLE_DENSITY_DEBUG: Bool = false
}
