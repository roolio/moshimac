// Copyright (c) MoshiMac
// This source code is licensed under the MIT license

import SwiftUI

struct RecordingOverlay: View {
    @ObservedObject var session: RecordingSession

    var body: some View {
        VStack(spacing: 12) {
            // Status indicator
            HStack {
                Circle()
                    .fill(statusColor)
                    .frame(width: 12, height: 12)
                    .shadow(color: statusColor.opacity(0.5), radius: 4)

                Text(statusText)
                    .font(.system(size: 14, weight: .medium))

                Spacer()

                Text(timeString)
                    .font(.system(size: 14, design: .monospaced))
                    .foregroundColor(.secondary)
            }

            // Progress bar
            if session.state == .recording {
                GeometryReader { geometry in
                    ZStack(alignment: .leading) {
                        // Background
                        RoundedRectangle(cornerRadius: 2)
                            .fill(Color.gray.opacity(0.2))

                        // Progress
                        RoundedRectangle(cornerRadius: 2)
                            .fill(statusColor)
                            .frame(width: progressWidth(in: geometry.size.width))
                    }
                }
                .frame(height: 4)
            }

            // Live transcript
            if !session.transcript.isEmpty && Preferences.shared.showTranscriptInOverlay {
                Text(session.transcript)
                    .font(.system(size: 12))
                    .foregroundColor(.primary)
                    .lineLimit(2)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.top, 4)
            }

            // Hint
            Text(hintText)
                .font(.system(size: 11))
                .foregroundColor(.secondary)
        }
        .padding(16)
        .frame(width: 400)
        .background(.ultraThinMaterial)
        .cornerRadius(12)
        .shadow(color: .black.opacity(0.2), radius: 20, x: 0, y: 8)
    }

    private var statusColor: Color {
        switch session.state {
        case .idle:
            return .gray
        case .recording:
            return .red
        case .processing:
            return .orange
        case .done:
            return .green
        case .error:
            return .red
        }
    }

    private var statusText: String {
        session.state.description
    }

    private var timeString: String {
        let minutes = Int(session.duration) / 60
        let seconds = Int(session.duration) % 60
        let deciseconds = Int((session.duration.truncatingRemainder(dividingBy: 1)) * 10)
        return String(format: "%02d:%02d.%01d", minutes, seconds, deciseconds)
    }

    private func progressWidth(in totalWidth: CGFloat) -> CGFloat {
        let maxDuration: TimeInterval = 60 // 1 minute max
        let progress = min(session.duration / maxDuration, 1.0)
        return totalWidth * progress
    }

    private var hintText: String {
        switch session.state {
        case .idle:
            return "Press ⌘⇧V to start recording"
        case .recording:
            return "Press ⌘⇧V to stop • Speak clearly into your microphone"
        case .processing:
            return "Processing your speech..."
        case .done:
            return "Text inserted!"
        case .error(let message):
            return message
        }
    }
}

#Preview {
    RecordingOverlay(session: RecordingSession(sttEngine: STTEngine()))
}
