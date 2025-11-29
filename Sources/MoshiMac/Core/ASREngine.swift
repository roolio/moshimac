// Copyright (c) MoshiMac
// ASR Engine adapted to use RustyMimiTokenizer

import Foundation
import MLX
import MLXNN

/// ASR Engine using RustyMimi for audio encoding
class ASREngine {
    let lm: LM
    let vocab: [Int: String]
    let mimi: RustyMimiTokenizer
    var prevTextToken: Int = 0
    let sampler: Sampler = Sampler(temp: 0.0)

    init(lm: LM, mimi: RustyMimiTokenizer, vocab: [Int: String]) {
        self.lm = lm
        self.mimi = mimi
        self.vocab = vocab
    }

    func reset() {
        mimi.reset()
        lm.resetCache()
        prevTextToken = lm.cfg.textInitToken()

        let textIds = MLXArray([prevTextToken]).reshaped([1, 1])
        let audioIds = (0..<lm.cfg.audioCodebooks).map { _ in
            MLXArray([lm.cfg.audioPaddingToken()])
        }
        let (_, textLogits) = lm.stepMain(textIds: textIds, audioIds: audioIds)
        let (textToken, _) = sampler(logits: textLogits)
        let textTokenI: Int = textToken[0].item()
        prevTextToken = textTokenI
    }

    func onPcmInput(_ pcm: MLXArray) -> [String] {
        var tokens: [String] = []
        let codebooks = lm.cfg.audioCodebooks

        // Encode with RustyMimi
        let codes: MLXArray
        do {
            codes = try mimi.encodeStep(pcm)
        } catch {
            print("Error encoding audio: \(error)")
            return []
        }

        codes.eval()

        let (_, _, steps) = codes.shape3
        for step in 0..<steps {
            let textIds = MLXArray([prevTextToken]).reshaped([1, 1])
            let audioIds = (0..<codebooks).map { codes[0..., $0, step].reshaped(1, 1) }

            let (_, textLogits) = lm.stepMain(textIds: textIds, audioIds: audioIds)
            eval(textLogits)

            let (textToken, _) = sampler(logits: textLogits)
            let textTokenI: Int = textToken[0].item()

            if textTokenI != 0 && textTokenI != 3 {
                if var v = vocab[textTokenI] {
                    v.replace("▁", with: " ")
                    tokens.append(v)
                }
            }
            prevTextToken = textTokenI
        }

        return tokens
    }
}
