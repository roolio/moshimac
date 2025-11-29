// Copyright (c) MoshiMac
// Real STT Engine using moshi-swift ASR

import Foundation
import MLX
import MLXNN
import AVFoundation

@MainActor
class STTEngine: ObservableObject {
    @Published var isReady = false
    @Published var currentTranscript = ""

    private var asr: ASREngine?
    private var audioCapture: MicrophoneCapture?
    private var isProcessing = false

    // Configuration
    private let modelRepo = "kyutai/stt-1b-en_fr-mlx"
    private let modelFilename = "model.safetensors"
    private let chunkSize = 1920  // 80ms @ 24kHz

    func initialize() async {
        print("Initializing STT Engine...")

        do {
            // 1. Download and load models
            print("Downloading model from \(modelRepo)...")
            let modelURL = try await ModelManager.downloadModel(repo: modelRepo, filename: modelFilename)
            // Vocab is loaded from lmz/moshi-swift repo (vocab size 8000 for this model)
            let vocab = try await ModelManager.loadVocab(from: modelRepo, vocabSize: 8000)

            print("Loading models into memory...")

            // 2. Load Mimi codec (from lmz/moshi-swift with key mapping)
            let mimi = try await ModelManager.loadMimi(numCodebooks: 32)

            // 3. Create LM model
            let lmConfig = LmConfig.asr1b()
            let lm = LM(lmConfig, bSize: 1)
            let lmWeights = try loadArrays(url: modelURL)
            try lm.update(parameters: ModuleParameters.unflattened(lmWeights), verify: [.all])

            // 4. Warmup
            print("Warming up LM...")
            lm.warmup()

            // 5. Create ASR with RustyMimi
            asr = ASREngine(lm: lm, mimi: mimi, vocab: vocab)
            asr?.reset()

            await MainActor.run {
                isReady = true
                print("STT Engine ready!")
            }
        } catch {
            print("Failed to initialize STT Engine: \(error)")
            await MainActor.run {
                isReady = false
            }
        }
    }

    func startListening() {
        guard isReady, !isProcessing, let asr = asr else {
            print("Cannot start listening: not ready or already processing")
            return
        }

        isProcessing = true
        currentTranscript = ""

        // Start audio capture
        audioCapture = MicrophoneCapture()
        audioCapture?.startCapturing()

        // Process audio chunks in background
        Task {
            while isProcessing, let audioCapture = audioCapture {
                if let pcmData = audioCapture.receive() {
                    let tokens = await processChunk(pcmData)

                    await MainActor.run {
                        for token in tokens {
                            currentTranscript += token
                        }
                    }
                }
            }
        }

        print("Started listening")
    }

    func stopListening() {
        guard isProcessing else { return }

        audioCapture?.stopCapturing()
        audioCapture = nil
        isProcessing = false

        print("Stopped listening. Final transcript: \(currentTranscript)")
    }

    func reset() {
        stopListening()
        currentTranscript = ""
        asr?.reset()
    }

    private func processChunk(_ pcmData: [Float]) async -> [String] {
        guard let asr = asr else { return [] }

        // Convert to MLX array with proper shape: [1, 1, chunkSize]
        let pcmArray = MLXArray(pcmData).reshaped([1, 1, pcmData.count])

        // Process through ASR
        let tokens = asr.onPcmInput(pcmArray)

        return tokens
    }
}
