// Copyright (c) MoshiMac
// This source code is licensed under the MIT license

import Foundation
import Combine

@MainActor
class RecordingSession: ObservableObject {
    @Published var state: RecordingState = .idle
    @Published var transcript = ""
    @Published var duration: TimeInterval = 0

    private let sttEngine: STTEngine
    private var timer: Timer?
    private var cancellables = Set<AnyCancellable>()

    init(sttEngine: STTEngine) {
        self.sttEngine = sttEngine

        // Subscribe to engine transcript updates
        sttEngine.$currentTranscript
            .receive(on: DispatchQueue.main)
            .sink { [weak self] newTranscript in
                self?.transcript = newTranscript
            }
            .store(in: &cancellables)
    }

    func startRecording() {
        guard state == .idle else { return }
        guard sttEngine.isReady else {
            state = .error("Engine not ready")
            return
        }

        state = .recording
        transcript = ""
        duration = 0

        sttEngine.startListening()

        // Update duration timer
        timer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            guard let self = self else { return }
            Task { @MainActor in
                self.duration += 0.1
            }
        }

        print("Recording started")
    }

    func stopRecording() async {
        guard state == .recording else { return }

        timer?.invalidate()
        timer = nil

        state = .processing

        sttEngine.stopListening()

        // Wait for final tokens to be processed
        try? await Task.sleep(nanoseconds: 500_000_000) // 0.5s

        state = .done

        // Insert text into active application
        if !transcript.isEmpty {
            await TextInserter.shared.insertText(transcript)
        }

        print("Recording stopped. Transcript: \(transcript)")

        // Reset after delay
        try? await Task.sleep(nanoseconds: 1_000_000_000) // 1s
        state = .idle
        transcript = ""
        duration = 0
    }

    func reset() {
        timer?.invalidate()
        timer = nil
        state = .idle
        transcript = ""
        duration = 0
        sttEngine.reset()
    }
}
