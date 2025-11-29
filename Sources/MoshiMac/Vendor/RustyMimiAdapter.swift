// Copyright (c) MoshiMac
// Adapter to make RustyMimiTokenizer compatible with ASR's Mimi interface

import Foundation
import MLX
import MLXNN

/// Adapter that wraps RustyMimiTokenizer to look like Mimi for ASR
public class RustyMimiAdapter {
    private let tokenizer: RustyMimiTokenizer

    public init(tokenizer: RustyMimiTokenizer) {
        self.tokenizer = tokenizer
    }

    /// Encode audio in streaming mode
    public func encodeStep(_ pcm: StreamArray) -> StreamArray {
        guard let pcmArray = pcm.asArray() else {
            return StreamArray(nil)
        }

        do {
            let codes = try tokenizer.encodeStep(pcmArray)
            return StreamArray(codes)
        } catch {
            print("Error encoding with RustyMimi: \(error)")
            return StreamArray(nil)
        }
    }

    /// Reset tokenizer state
    public func resetState() {
        tokenizer.reset()
    }

    // Dummy methods to satisfy potential Mimi interface requirements
    public func warmup() {
        // RustyMimi doesn't need warmup
    }
}
