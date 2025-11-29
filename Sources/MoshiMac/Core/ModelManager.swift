// Model downloading and management utilities

import Foundation
import Hub
import Tokenizers
import MLX
import MLXNN

class ModelManager {
    static func downloadModel(repo: String, filename: String) async throws -> URL {
        let hubApi = HubApi()
        let repoRef = Hub.Repo(id: repo)

        // Check if already cached
        let targetURL = hubApi.localRepoLocation(repoRef).appending(path: filename)
        if FileManager.default.fileExists(atPath: targetURL.path) {
            print("Using cached file: \(targetURL.path)")
            return targetURL
        }

        // Download with progress
        print("Downloading \(filename) from \(repo)...")
        let url = try await Hub.snapshot(from: repoRef, matching: filename) { progress in
            let pct = Int(progress.fractionCompleted * 100)
            if pct % 10 == 0 {
                print("Downloading \(filename): \(pct)%")
            }
        }

        print("Downloaded \(filename)")
        return url.appending(path: filename)
    }

    static func loadVocab(from repo: String, vocabSize: Int = 8000) async throws -> [Int: String] {
        // Determine the vocab file based on size
        let filename: String
        switch vocabSize {
        case 48000:
            filename = "tokenizer_spm_48k_multi6_2.json"
        case 32000:
            filename = "tokenizer_spm_32k_3.json"
        case 8000:
            filename = "tokenizer_spm_8k_0.json"
        case 4000:
            filename = "test_en_audio_4000.json"
        default:
            throw NSError(domain: "ModelManager", code: 1, userInfo: [NSLocalizedDescriptionKey: "Unsupported vocab size: \(vocabSize)"])
        }

        // Download from lmz/moshi-swift repo (where vocab files are stored)
        let vocabURL = try await downloadModel(repo: "lmz/moshi-swift", filename: filename)
        let data = try Data(contentsOf: vocabURL)

        // Parse JSON: it's a dictionary [Int: String]
        let decoder = JSONDecoder()
        let vocab = try decoder.decode([Int: String].self, from: data)

        return vocab
    }

    static func loadMimi(numCodebooks: Int = 32) async throws -> RustyMimiTokenizer {
        // Download Mimi weights from HuggingFace
        print("Downloading Mimi weights...")
        let mimiURL = try await downloadModel(
            repo: "kyutai/stt-1b-en_fr-mlx",
            filename: "mimi-pytorch-e351c8d8@125.safetensors"
        )

        print("Loading Mimi tokenizer from Rust...")
        let tokenizer = try RustyMimiTokenizer(weightsPath: mimiURL.path, numCodebooks: numCodebooks)

        print("Mimi tokenizer loaded successfully!")
        return tokenizer
    }
}
