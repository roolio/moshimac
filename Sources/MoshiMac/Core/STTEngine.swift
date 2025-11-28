// Copyright (c) MoshiMac
// This source code is licensed under the MIT license

import Foundation
import MLX
import MLXNN
import AVFoundation

// NOTE: This is a placeholder. We'll integrate the actual moshi-swift code next.
// For now, this provides the interface that the rest of the app expects.

@MainActor
class STTEngine: ObservableObject {
    @Published var isReady = false
    @Published var currentTranscript = ""

    private var isProcessing = false
    private var audioCapture: AudioCapture?

    // Configuration
    private let modelRepo = "kyutai/stt-1b-en_fr-mlx"
    private let chunkSize = 1920  // 80ms @ 24kHz

    func initialize() async {
        print("Initializing STT Engine...")

        // TODO: Load model from moshi-swift
        // For now, just simulate initialization
        try? await Task.sleep(nanoseconds: 2_000_000_000) // 2s

        await MainActor.run {
            isReady = true
            print("STT Engine ready")
        }
    }

    func startListening() {
        guard isReady, !isProcessing else { return }

        isProcessing = true
        currentTranscript = ""

        // TODO: Start audio capture with MicrophoneCapture from moshi-swift
        audioCapture = AudioCapture()
        audioCapture?.startCapturing { [weak self] pcmData in
            Task { @MainActor in
                await self?.processAudioChunk(pcmData)
            }
        }

        print("Started listening")
    }

    func stopListening() {
        guard isProcessing else { return }

        audioCapture?.stopCapturing()
        audioCapture = nil
        isProcessing = false

        print("Stopped listening")
    }

    func reset() {
        stopListening()
        currentTranscript = ""
    }

    private func processAudioChunk(_ pcmData: [Float]) async {
        // TODO: Process with ASR from moshi-swift
        // For now, just simulate transcription
        // This will be replaced with actual moshi-swift integration

        // Placeholder: append dummy text for testing
        // In real implementation, this calls asr.onPcmInput(pcmArray)
    }
}

// MARK: - Placeholder AudioCapture
// This will be replaced with MicrophoneCapture from moshi-swift

class AudioCapture {
    private var audioEngine: AVAudioEngine?
    private var callback: (([Float]) -> Void)?

    func startCapturing(callback: @escaping ([Float]) -> Void) {
        self.callback = callback
        audioEngine = AVAudioEngine()

        guard let engine = audioEngine else { return }
        let inputNode = engine.inputNode

        let desiredSampleRate: Double = 24000.0
        let inputFormat = inputNode.inputFormat(forBus: 0)

        guard let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: desiredSampleRate,
            channels: 1,
            interleaved: false
        ) else {
            print("Could not create audio format")
            return
        }

        let converter = AVAudioConverter(from: inputFormat, to: targetFormat)

        inputNode.installTap(onBus: 0, bufferSize: 1920, format: inputFormat) { [weak self] buffer, _ in
            let targetLen = Int(buffer.frameLength) * 24000 / Int(inputFormat.sampleRate)
            guard let convertedBuffer = AVAudioPCMBuffer(
                pcmFormat: targetFormat,
                frameCapacity: AVAudioFrameCount(targetLen)
            ) else { return }

            var error: NSError?
            converter?.convert(to: convertedBuffer, error: &error) { _, outStatus in
                outStatus.pointee = .haveData
                return buffer
            }

            if error == nil, let channelData = convertedBuffer.floatChannelData {
                let frameCount = Int(convertedBuffer.frameLength)
                let pcmData = Array(UnsafeBufferPointer(start: channelData[0], count: frameCount))
                self?.callback?(pcmData)
            }
        }

        do {
            try engine.start()
            print("Audio capture started")
        } catch {
            print("Error starting audio engine: \(error)")
        }
    }

    func stopCapturing() {
        audioEngine?.stop()
        audioEngine?.inputNode.removeTap(onBus: 0)
        audioEngine = nil
        print("Audio capture stopped")
    }
}
