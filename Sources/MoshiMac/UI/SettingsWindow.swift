// Copyright (c) MoshiMac
// This source code is licensed under the MIT license

import SwiftUI
import KeyboardShortcuts

struct SettingsWindow: View {
    @ObservedObject var preferences = Preferences.shared
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        TabView {
            GeneralSettingsView(preferences: preferences)
                .tabItem {
                    Label("General", systemImage: "gear")
                }

            ShortcutsSettingsView()
                .tabItem {
                    Label("Shortcuts", systemImage: "keyboard")
                }

            AboutSettingsView()
                .tabItem {
                    Label("About", systemImage: "info.circle")
                }
        }
        .frame(width: 500, height: 400)
    }
}

struct GeneralSettingsView: View {
    @ObservedObject var preferences: Preferences

    var body: some View {
        Form {
            Section {
                VStack(alignment: .leading, spacing: 12) {
                    Text("Model Configuration")
                        .font(.headline)

                    HStack {
                        Text("Model Repository:")
                        TextField("kyutai/stt-1b-en_fr-mlx", text: $preferences.modelRepo)
                            .textFieldStyle(.roundedBorder)
                    }

                    Text("Change will take effect after restarting the app")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }

            Section {
                VStack(alignment: .leading, spacing: 12) {
                    Text("Display Options")
                        .font(.headline)

                    Toggle("Show transcript in overlay", isOn: $preferences.showTranscriptInOverlay)
                }
            }

            Section {
                VStack(alignment: .leading, spacing: 12) {
                    Text("Advanced Features")
                        .font(.headline)

                    Toggle("Auto punctuation", isOn: $preferences.autoPunctuation)
                        .help("Automatically add punctuation to transcribed text")

                    Toggle("Voice Activity Detection (VAD)", isOn: $preferences.enableVAD)
                        .help("Automatically detect when you stop speaking")
                }
            }
        }
        .formStyle(.grouped)
        .padding()
    }
}

struct ShortcutsSettingsView: View {
    var body: some View {
        Form {
            Section {
                VStack(alignment: .leading, spacing: 16) {
                    Text("Keyboard Shortcuts")
                        .font(.headline)

                    HStack {
                        Text("Toggle Recording:")
                            .frame(width: 150, alignment: .leading)
                        KeyboardShortcuts.Recorder(for: .toggleRecording)
                    }

                    Text("Press once to start recording, press again to stop")
                        .font(.caption)
                        .foregroundColor(.secondary)

                    Divider()
                        .padding(.vertical, 4)

                    HStack {
                        Text("Push to Talk:")
                            .frame(width: 150, alignment: .leading)
                        KeyboardShortcuts.Recorder(for: .pushToTalk)
                    }

                    Text("Hold to record, release to stop")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
        }
        .formStyle(.grouped)
        .padding()
    }
}

struct AboutSettingsView: View {
    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "mic.circle.fill")
                .font(.system(size: 64))
                .foregroundColor(.blue)

            Text("MoshiMac")
                .font(.title)
                .fontWeight(.bold)

            Text("Version 1.0.0")
                .font(.subheadline)
                .foregroundColor(.secondary)

            VStack(spacing: 8) {
                Text("Speech-to-Text powered by Kyutai")
                    .font(.body)

                Text("Running 100% locally on your Mac")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            Spacer()

            VStack(spacing: 12) {
                Link("Open Source on GitHub", destination: URL(string: "https://github.com/yourusername/moshimac")!)
                    .buttonStyle(.link)

                Text("Built with MLX Swift and Kyutai's state-of-the-art STT models")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
            }
        }
        .padding()
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

#Preview {
    SettingsWindow()
}
