import XCTest
import Metal
@testable import VolumeRendering_iOS

final class TransferFunctionLoaderTests: XCTestCase {

    func testAllBundledTransferFunctionsLoadAndCreateTextures() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal indisponível para validar transfer functions.")
        }

        let tfDirectory = DicomTestSupport.appResourcesRoot()
            .appendingPathComponent("Resource", isDirectory: true)
            .appendingPathComponent("TransferFunction", isDirectory: true)

        let tfURLs = try FileManager.default.contentsOfDirectory(at: tfDirectory,
                                                                 includingPropertiesForKeys: nil,
                                                                 options: [.skipsHiddenFiles])
            .filter { $0.pathExtension.lowercased() == "tf" }

        XCTAssertFalse(tfURLs.isEmpty, "Nenhum arquivo .tf foi encontrado em Resource/TransferFunction.")

        for url in tfURLs {
            let transfer = TransferFunction.load(from: url)

            XCTAssertFalse(transfer.colourPoints.isEmpty,
                           "Transfer function \(url.lastPathComponent) sem colour points.")
            XCTAssertFalse(transfer.alphaPoints.isEmpty,
                           "Transfer function \(url.lastPathComponent) sem alpha points.")
            XCTAssertLessThan(transfer.min, transfer.max,
                              "Faixa inválida em \(url.lastPathComponent) (\(transfer.min) >= \(transfer.max)).")

            let texture = transfer.get(device: device)
            XCTAssertNotNil(texture, "Falha ao gerar textura Metal para \(url.lastPathComponent).")
            XCTAssertEqual(texture?.width, 512)
            XCTAssertEqual(texture?.height, 2)
        }
    }
}
