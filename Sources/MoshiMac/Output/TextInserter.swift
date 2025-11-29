// Copyright (c) MoshiMac
// This source code is licensed under the MIT license

import AppKit
import CoreGraphics

@MainActor
class TextInserter {
    static let shared = TextInserter()

    private init() {}

    func insertText(_ text: String) async {
        // Try to paste the text directly into the active application
        let success = await pasteText(text)

        if !success {
            // Fallback to clipboard
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(text, forType: .string)
            print("Text copied to clipboard as fallback")
        }
    }

    private func pasteText(_ text: String) async -> Bool {
        // Copy to clipboard first
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)

        // Check if we have accessibility permissions
        let hasPermissions = AXIsProcessTrusted()
        if !hasPermissions {
            print("⚠️  Accessibility permissions not granted - text copied to clipboard")
            print("   Enable accessibility in System Settings > Privacy & Security > Accessibility")
            // Still try to paste, sometimes it works
        }

        // Small delay to ensure clipboard is updated
        try? await Task.sleep(nanoseconds: 50_000_000) // 50ms

        // Try to simulate Cmd+V to paste
        let success = simulatePaste()

        if success {
            print("✅ Text pasted successfully")
        } else {
            print("⚠️  Paste simulation failed - text is in clipboard")
        }

        return success
    }

    private func simulatePaste() -> Bool {
        // Create Cmd+V key event
        guard let source = CGEventSource(stateID: .hidSystemState) else {
            return false
        }

        // Key down for 'v' with Cmd modifier
        guard let keyDownEvent = CGEvent(
            keyboardEventSource: source,
            virtualKey: 0x09, // 'v' key
            keyDown: true
        ) else {
            return false
        }

        keyDownEvent.flags = .maskCommand

        // Key up for 'v'
        guard let keyUpEvent = CGEvent(
            keyboardEventSource: source,
            virtualKey: 0x09,
            keyDown: false
        ) else {
            return false
        }

        keyUpEvent.flags = .maskCommand

        // Post the events
        keyDownEvent.post(tap: .cghidEventTap)
        keyUpEvent.post(tap: .cghidEventTap)

        return true
    }

    private func requestAccessibilityPermissions() async {
        let alert = NSAlert()
        alert.messageText = "Accessibility Access Required"
        alert.informativeText = "MoshiMac needs accessibility access to paste transcribed text directly into applications. Please enable it in System Settings."
        alert.alertStyle = .warning
        alert.addButton(withTitle: "Open System Settings")
        alert.addButton(withTitle: "Cancel")

        if alert.runModal() == .alertFirstButtonReturn {
            if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility") {
                NSWorkspace.shared.open(url)
            }
        }
    }
}
