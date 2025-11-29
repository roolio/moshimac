// Copyright (c) MoshiMac
// Swift wrapper for rustymimi C FFI

import Foundation
import MLX

// C function declarations (importing from librustymimi_c.dylib)
@_silgen_name("mimi_tokenizer_new")
func mimi_tokenizer_new(_ path: UnsafePointer<CChar>, _ numCodebooks: Int) -> OpaquePointer?

@_silgen_name("mimi_encode_step")
func mimi_encode_step(
    _ tokenizer: OpaquePointer?,
    _ pcmData: UnsafePointer<Float>?,
    _ samples: Int,
    _ outCodes: UnsafeMutablePointer<UnsafeMutablePointer<UInt32>?>,
    _ outCodebooks: UnsafeMutablePointer<Int>,
    _ outSteps: UnsafeMutablePointer<Int>
) -> Int32

@_silgen_name("mimi_reset")
func mimi_reset(_ tokenizer: OpaquePointer?)

@_silgen_name("mimi_tokenizer_free")
func mimi_tokenizer_free(_ tokenizer: OpaquePointer?)

@_silgen_name("mimi_free_codes")
func mimi_free_codes(_ codes: UnsafeMutablePointer<UInt32>?, _ size: Int)

/// Swift wrapper for the Rust-based Mimi tokenizer
public class RustyMimiTokenizer {
    private var tokenizer: OpaquePointer?

    public init(weightsPath: String, numCodebooks: Int = 32) throws {
        tokenizer = mimi_tokenizer_new(weightsPath, numCodebooks)
        if tokenizer == nil {
            throw NSError(
                domain: "RustyMimi",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Failed to create Mimi tokenizer from \(weightsPath)"]
            )
        }
    }

    deinit {
        if let tokenizer = tokenizer {
            mimi_tokenizer_free(tokenizer)
        }
    }

    /// Encode PCM audio data to codes (streaming mode)
    /// - Parameter pcm: PCM audio data as MLXArray with shape [1, 1, samples]
    /// - Returns: Audio codes as MLXArray with shape [1, codebooks, steps]
    public func encodeStep(_ pcm: MLXArray) throws -> MLXArray {
        // Get PCM data as contiguous float array
        let pcmData = pcm.asArray(Float.self)
        let samples = pcmData.count

        var outCodes: UnsafeMutablePointer<UInt32>? = nil
        var outCodebooks: Int = 0
        var outSteps: Int = 0

        let result = pcmData.withUnsafeBufferPointer { pcmPtr in
            mimi_encode_step(
                tokenizer,
                pcmPtr.baseAddress,
                samples,
                &outCodes,
                &outCodebooks,
                &outSteps
            )
        }

        guard result == 0, let codesPtr = outCodes else {
            throw NSError(
                domain: "RustyMimi",
                code: 2,
                userInfo: [NSLocalizedDescriptionKey: "Failed to encode audio"]
            )
        }

        // Convert flat codes back to [1, codebooks, steps]
        var codes: [[[UInt32]]] = [[[UInt32]]](
            repeating: [[UInt32]](
                repeating: [UInt32](repeating: 0, count: outSteps),
                count: outCodebooks
            ),
            count: 1
        )

        for step in 0..<outSteps {
            for cb in 0..<outCodebooks {
                let idx = step * outCodebooks + cb
                codes[0][cb][step] = codesPtr[idx]
            }
        }

        // Free the allocated buffer
        mimi_free_codes(codesPtr, outCodebooks * outSteps)

        // Convert to MLXArray - codes shape is [1, codebooks, steps]
        // Flatten to 1D array first
        var flatCodes: [Int32] = []
        flatCodes.reserveCapacity(outCodebooks * outSteps)
        for cb in 0..<outCodebooks {
            for step in 0..<outSteps {
                flatCodes.append(Int32(codes[0][cb][step]))
            }
        }

        // Create MLXArray and reshape to [1, codebooks, steps]
        let mlxArray = MLXArray(flatCodes)
        return mlxArray.reshaped([1, outCodebooks, outSteps])
    }

    /// Reset the tokenizer state
    public func reset() {
        if let tokenizer = tokenizer {
            mimi_reset(tokenizer)
        }
    }
}
