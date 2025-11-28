// Copyright (c) MoshiMac
// This source code is licensed under the MIT license

import Foundation
import KeyboardShortcuts

extension KeyboardShortcuts.Name {
    static let toggleRecording = Self("toggleRecording", default: .init(.v, modifiers: [.command, .shift]))
    static let pushToTalk = Self("pushToTalk", default: .init(.t, modifiers: [.command, .shift]))
}

class HotkeyManager {
    private let recordingSession: RecordingSession

    init(recordingSession: RecordingSession) {
        self.recordingSession = recordingSession
    }

    func setup() {
        // Toggle recording (press once to start, press again to stop)
        KeyboardShortcuts.onKeyDown(for: .toggleRecording) { [weak self] in
            Task { @MainActor in
                guard let self = self else { return }

                switch self.recordingSession.state {
                case .idle:
                    self.recordingSession.startRecording()
                case .recording:
                    await self.recordingSession.stopRecording()
                default:
                    break
                }
            }
        }

        // Push to talk (hold to record, release to stop)
        KeyboardShortcuts.onKeyDown(for: .pushToTalk) { [weak self] in
            Task { @MainActor in
                guard let self = self else { return }
                if self.recordingSession.state == .idle {
                    self.recordingSession.startRecording()
                }
            }
        }

        KeyboardShortcuts.onKeyUp(for: .pushToTalk) { [weak self] in
            Task { @MainActor in
                guard let self = self else { return }
                if self.recordingSession.state == .recording {
                    await self.recordingSession.stopRecording()
                }
            }
        }

        print("Hotkeys configured:")
        print("  ⌘⇧V - Toggle recording")
        print("  ⌘⇧T - Push to talk")
    }
}
