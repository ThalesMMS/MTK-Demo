import Foundation
import Metal
import simd

struct TransferFunction: Codable
{
    var version: Int?
    var name: String = ""
    var colourPoints: [ColorPoint] = []
    var alphaPoints: [AlphaPoint] = []
    
    var min: Float = -1024
    var max: Float = 3071
    var shift: Float = 0
    
    static func load(from: URL) -> TransferFunction {
        Logger.log("Tentando carregar Transfer Function em \(from.lastPathComponent)", level: .debug, category: "TransferFunction")
        guard let data = try? Data(contentsOf: from) else {
            Logger.log("Não foi possível ler dados de \(from.path)", level: .error, category: "TransferFunction")
            return TransferFunction()
        }
        guard let tf = try? JSONDecoder().decode(TransferFunction.self, from: data) else {
            Logger.log("Falha ao decodificar Transfer Function \(from.path)", level: .error, category: "TransferFunction")
            return TransferFunction()
        }
        Logger.log("Transfer Function carregada: \(tf.name)", level: .info, category: "TransferFunction")
        return tf
    }
    
    func get(device: MTLDevice) -> MTLTexture?
    {
        let TEXTURE_WIDTH = 512
        let TEXTURE_HEIGHT = 2
        
        var tfCols = [RGBAColor].init(repeating: RGBAColor(), count: TEXTURE_WIDTH * TEXTURE_HEIGHT)
        
        // sort
        var cols = colourPoints.sorted(by: { $0.dataValue < $1.dataValue })
        var alps = alphaPoints.sorted(by: { $0.dataValue < $1.dataValue })

        // apply shift
        cols = cols.map { point in
            var tmp = point
            tmp.dataValue = clamp(tmp.dataValue + shift, min: min, max: max)
            return tmp
        }

        alps = alps.map { point in
            var tmp = point
            tmp.dataValue = clamp(tmp.dataValue + shift, min: min, max: max)
            return tmp
        }

        cols = sanitize(points: cols,
                         defaultStart: ColorPoint(dataValue: min, colourValue: RGBAColor(r: 1, g: 1, b: 1, a: 1)),
                         defaultEnd: ColorPoint(dataValue: max, colourValue: RGBAColor(r: 1, g: 1, b: 1, a: 1)))

        alps = sanitize(points: alps,
                         defaultStart: AlphaPoint(dataValue: min, alphaValue: 0),
                         defaultEnd: AlphaPoint(dataValue: max, alphaValue: 1))

        var iCurrColor = 0
        var iCurrAlpha = 0

        for ix in 0 ..< TEXTURE_WIDTH
        {
            let t = Float(ix) / Float(TEXTURE_WIDTH - 1)
            while iCurrColor < cols.count - 2,
                  normalize(cols[iCurrColor + 1].dataValue) <= t
            {
                iCurrColor += 1
            }
            while iCurrAlpha < alps.count - 2,
                  normalize(alps[iCurrAlpha + 1].dataValue) <= t
            {
                iCurrAlpha += 1
            }

            let leftCol = cols[iCurrColor]
            let rightCol = cols[iCurrColor + 1]
            let leftAlp = alps[iCurrAlpha]
            let rightAlp = alps[iCurrAlpha + 1]

            let leftColNorm = normalize(leftCol.dataValue)
            let rightColNorm = normalize(rightCol.dataValue)
            let leftAlpNorm = normalize(leftAlp.dataValue)
            let rightAlpNorm = normalize(rightAlp.dataValue)

            let colDenom = Swift.max(abs(rightColNorm - leftColNorm), 1.0e-6)
            let alpDenom = Swift.max(abs(rightAlpNorm - leftAlpNorm), 1.0e-6)

            let clampedCol = simd_clamp(t,
                                        Swift.min(leftColNorm, rightColNorm),
                                        Swift.max(leftColNorm, rightColNorm))
            let clampedAlp = simd_clamp(t,
                                        Swift.min(leftAlpNorm, rightAlpNorm),
                                        Swift.max(leftAlpNorm, rightAlpNorm))

            let tCol = (clampedCol - leftColNorm) / colDenom
            let tAlp = (clampedAlp - leftAlpNorm) / alpDenom

            var pixCol = rightCol.colourValue * Float(tCol) + leftCol.colourValue * Float(1 - tCol)
            let blendedAlpha = rightAlp.alphaValue * tAlp + leftAlp.alphaValue * (1 - tAlp)
            pixCol.a = blendedAlpha

            for iy in 0 ..< TEXTURE_HEIGHT
            {
                tfCols[ix + iy * TEXTURE_WIDTH] = pixCol
            }
        }
        
        let textureDescriptor = MTLTextureDescriptor()
        textureDescriptor.textureType = .type2D
        textureDescriptor.pixelFormat = .rgba32Float
        textureDescriptor.width = TEXTURE_WIDTH
        textureDescriptor.height = TEXTURE_HEIGHT
        textureDescriptor.usage = .shaderRead
        
        guard let texture = device.makeTexture(descriptor: textureDescriptor) else {
            Logger.log("Falha ao criar textura Metal para Transfer Function", level: .error, category: "TransferFunction")
            return nil // Retorna nil em vez de crashar
        }
        let labelName = name.isEmpty ? "TransferFunction" : "TransferFunction_\(name)"
        texture.label = labelName
        
        texture.replace(region: MTLRegionMake2D(0, 0, TEXTURE_WIDTH, TEXTURE_HEIGHT),
                        mipmapLevel: 0,
                        slice: 0,
                        withBytes: tfCols,
                        bytesPerRow: RGBAColor.size * TEXTURE_WIDTH,
                        bytesPerImage: TEXTURE_WIDTH * TEXTURE_HEIGHT * RGBAColor.size)
        
        return texture // Retorna a textura criada
    }
    
    func normalize(_ value: Float) -> Float
    {
        return (value - min) / (max - min)
    }
}

struct RGBAColor: Codable, sizeable
{
    var r: Float = 0
    var g: Float = 0
    var b: Float = 0
    var a: Float = 0
}

struct ColorPoint: Codable
{
    var dataValue: Float = 0
    var colourValue: RGBAColor = .init()
}

struct AlphaPoint: Codable
{
    var dataValue: Float = 0
    var alphaValue: Float = 0
}

func * (color: RGBAColor, value: Float) -> RGBAColor
{
    return RGBAColor(r: color.r * value, g: color.g * value, b: color.b * value, a: color.a * value)
}

func + (a: RGBAColor, b: RGBAColor) -> RGBAColor
{
    return RGBAColor(r: a.r + b.r, g: a.g + b.g, b: a.b + b.b, a: a.a + b.a)
}

private extension TransferFunction {
    func clamp(_ value: Float, min: Float, max: Float) -> Float {
        return Swift.max(min, Swift.min(max, value))
    }

    func sanitize(points: [ColorPoint], defaultStart: ColorPoint, defaultEnd: ColorPoint) -> [ColorPoint] {
        guard !points.isEmpty else {
            return adjustEndpoints([defaultStart, defaultEnd])
        }

        var result: [ColorPoint] = []
        for point in points {
            if let last = result.last, abs(last.dataValue - point.dataValue) < 1.0e-6 {
                // Replace duplicate entry, prefer latest value
                result[result.count - 1] = point
            } else {
                result.append(point)
            }
        }
        result = adjustEndpoints(result)
        if result.count == 1 {
            result.append(defaultEnd)
        }
        return result
    }

    func sanitize(points: [AlphaPoint], defaultStart: AlphaPoint, defaultEnd: AlphaPoint) -> [AlphaPoint] {
        guard !points.isEmpty else {
            return adjustEndpoints([defaultStart, defaultEnd])
        }

        var result: [AlphaPoint] = []
        for point in points {
            if let last = result.last, abs(last.dataValue - point.dataValue) < 1.0e-6 {
                result[result.count - 1] = point
            } else {
                result.append(point)
            }
        }
        result = adjustEndpoints(result)
        if result.count == 1 {
            result.append(defaultEnd)
        }
        return result
    }

    func adjustEndpoints<T>(_ points: [T]) -> [T] where T: TransferPoint {
        var mutablePoints = points
        if mutablePoints.isEmpty {
            return points
        }

        if mutablePoints.first!.dataValue > min {
            mutablePoints.insert(mutablePoints.first!.withDataValue(min), at: 0)
        } else if mutablePoints.first!.dataValue < min {
            mutablePoints[0] = mutablePoints.first!.withDataValue(min)
        }

        if mutablePoints.last!.dataValue < max {
            mutablePoints.append(mutablePoints.last!.withDataValue(max))
        } else if mutablePoints.last!.dataValue > max {
            mutablePoints[mutablePoints.count - 1] = mutablePoints.last!.withDataValue(max)
        }

        if mutablePoints.count == 1 {
            mutablePoints.append(mutablePoints[0].withDataValue(max))
        }

        return mutablePoints
    }
}

private protocol TransferPoint {
    var dataValue: Float { get }
    func withDataValue(_ value: Float) -> Self
}

extension ColorPoint: TransferPoint {
    func withDataValue(_ value: Float) -> ColorPoint {
        var copy = self
        copy.dataValue = value
        return copy
    }
}

extension AlphaPoint: TransferPoint {
    func withDataValue(_ value: Float) -> AlphaPoint {
        var copy = self
        copy.dataValue = value
        return copy
    }
}
