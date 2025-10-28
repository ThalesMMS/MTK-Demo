//
//  Logger.swift
//  VolumeRendering-iOS
//
//  Lightweight logger used by the demo while MTK-based logging hooks are integrated.
//  Thales Matheus Mendonça Santos — October 2025
//

import Foundation

/// Lightweight logger with level filtering and file output tailored for the demo target.
final class Logger {

    enum Level: Int, Comparable, CaseIterable {
        case debug = 0
        case info  = 1
        case warn  = 2
        case error = 3

        var tag: String {
            switch self {
            case .debug: return "DEBUG"
            case .info:  return "INFO"
            case .warn:  return "WARN"
            case .error: return "ERROR"
            }
        }

        static func < (lhs: Logger.Level, rhs: Logger.Level) -> Bool {
            lhs.rawValue < rhs.rawValue
        }
    }

    struct Configuration {
        var minimumLevel: Level
        var logToConsole: Bool

        static var `default`: Configuration {
            #if DEBUG
            return Configuration(minimumLevel: .debug, logToConsole: true)
            #else
            return Configuration(minimumLevel: .info, logToConsole: true)
            #endif
        }
    }

    static let shared = Logger(configuration: .default)

    private var minimumLevel: Level
    private var logToConsole: Bool

    private let queue = DispatchQueue(label: "com.acto3d.volumerendering.logger", qos: .utility)
    private let logDirectory: URL
    private let logFileURL: URL

    private lazy var fileHandle: FileHandle? = FileHandle(forWritingAtPath: logFileURL.path)

    private init(configuration: Configuration) {
        self.minimumLevel = configuration.minimumLevel
        self.logToConsole = configuration.logToConsole

        let fm = FileManager.default
        if let library = fm.urls(for: .libraryDirectory, in: .userDomainMask).first {
            logDirectory = library.appendingPathComponent("Logs", isDirectory: true)
        } else {
            logDirectory = fm.temporaryDirectory.appendingPathComponent("Logs", isDirectory: true)
        }

        if !fm.fileExists(atPath: logDirectory.path) {
            try? fm.createDirectory(at: logDirectory, withIntermediateDirectories: true, attributes: nil)
        }

        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd-HH-mm-ss"
        let filename = "VolumeRendering-\(formatter.string(from: Date())).log"
        logFileURL = logDirectory.appendingPathComponent(filename)

        if !fm.fileExists(atPath: logFileURL.path) {
            fm.createFile(atPath: logFileURL.path, contents: nil)
        }

        let bootstrapMessage = "Logger initialized → writing to \(logFileURL.lastPathComponent)"
        write(message: bootstrapMessage, level: .info, includeConsole: configuration.logToConsole)
    }

    deinit {
        fileHandle?.closeFile()
    }

    // MARK: - Public API

    static func configure(minimumLevel: Level? = nil, logToConsole: Bool? = nil) {
        shared.queue.async {
            if let min = minimumLevel {
                shared.minimumLevel = min
            }
            if let console = logToConsole {
                shared.logToConsole = console
            }
        }
    }

    static func log(_ message: String,
                    level: Level = .info,
                    category: String? = nil,
                    writeToFile: Bool = true,
                    file: String = #fileID,
                    line: Int = #line,
                    function: String = #function) {
        shared.enqueueLog(message,
                          level: level,
                          category: category,
                          writeToFile: writeToFile,
                          file: file,
                          line: line,
                          function: function)
    }

    static func logOnlyToFile(_ message: String,
                              level: Level = .info,
                              category: String? = nil) {
        log(message, level: level, category: category, writeToFile: true)
    }

    static func logPrintAndWrite(_ message: String,
                                 level: Level = .info,
                                 category: String? = nil) {
        log(message, level: level, category: category, writeToFile: true)
    }

    static var logFilePath: String {
        shared.logFileURL.path
    }

    static var logDirectoryPath: String {
        shared.logDirectory.path
    }

    // MARK: - Internal

    private func enqueueLog(_ message: String,
                            level: Level,
                            category: String?,
                            writeToFile: Bool,
                            file: String,
                            line: Int,
                            function: String) {
        guard level >= minimumLevel else { return }

        queue.async {
            let timestamp = Logger.timestampFormatter.string(from: Date())
            let categoryTag = category.map { "[\($0)] " } ?? ""
            let caller = "\(file)#L\(line)"
            let payload = "[\(timestamp)][\(level.tag)] \(categoryTag)\(message) (\(function) @ \(caller))"

            if self.logToConsole {
                print(payload)
            }

            if writeToFile {
                self.write(message: payload, level: level, includeConsole: false)
            }
        }
    }

    private func write(message: String, level: Level, includeConsole: Bool) {
        let line = "\(message)\n"
        if includeConsole {
            print(line, terminator: "")
        }

        guard let data = line.data(using: .utf8) else { return }

        if let handle = fileHandle {
            handle.seekToEndOfFile()
            handle.write(data)
        } else {
            do {
                try data.write(to: logFileURL)
                fileHandle = FileHandle(forWritingAtPath: logFileURL.path)
                fileHandle?.seekToEndOfFile()
            } catch {
                print("Logger failed creating file: \(error)")
            }
        }
    }

    private static let timestampFormatter: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter
    }()
}
