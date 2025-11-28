// Copyright (c) MoshiMac
// This source code is licensed under the MIT license

import AppKit
import SwiftUI
import AVFoundation

class AppDelegate: NSObject, NSApplicationDelegate {
    private var statusItem: NSStatusItem?
    private var recordingSession: RecordingSession?
    private var hotkeyManager: HotkeyManager?
    private var sttEngine: STTEngine?
    private var overlayWindow: NSWindow?

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Hide from Dock
        NSApp.setActivationPolicy(.accessory)

        // Create menu bar item
        setupMenuBar()

        // Initialize STT engine
        sttEngine = STTEngine()
        recordingSession = RecordingSession(sttEngine: sttEngine!)

        // Setup global hotkeys
        hotkeyManager = HotkeyManager(recordingSession: recordingSession!)
        hotkeyManager?.setup()

        // Initialize engine in background
        Task {
            await sttEngine?.initialize()
            updateMenuBarIcon(ready: true)
        }

        // Request microphone permission
        requestMicrophonePermission()

        print("MoshiMac started successfully")
    }

    private func setupMenuBar() {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)

        if let button = statusItem?.button {
            button.image = NSImage(systemSymbolName: "mic.circle", accessibilityDescription: "MoshiMac")
            button.image?.isTemplate = true
        }

        let menu = NSMenu()

        menu.addItem(NSMenuItem(title: "MoshiMac", action: nil, keyEquivalent: ""))
        menu.addItem(NSMenuItem.separator())

        let statusMenuItem = NSMenuItem(title: "Initializing...", action: nil, keyEquivalent: "")
        statusMenuItem.tag = 100
        menu.addItem(statusMenuItem)

        menu.addItem(NSMenuItem.separator())
        menu.addItem(NSMenuItem(title: "Settings...", action: #selector(openSettings), keyEquivalent: ","))
        menu.addItem(NSMenuItem.separator())
        menu.addItem(NSMenuItem(title: "Quit MoshiMac", action: #selector(quit), keyEquivalent: "q"))

        statusItem?.menu = menu
    }

    private func updateMenuBarIcon(ready: Bool) {
        if let item = statusItem?.menu?.item(withTag: 100) {
            item.title = ready ? "Ready to transcribe" : "Initializing..."
        }
    }

    private func requestMicrophonePermission() {
        AVCaptureDevice.requestAccess(for: .audio) { granted in
            if !granted {
                DispatchQueue.main.async {
                    self.showPermissionAlert()
                }
            }
        }
    }

    private func showPermissionAlert() {
        let alert = NSAlert()
        alert.messageText = "Microphone Access Required"
        alert.informativeText = "MoshiMac needs microphone access to transcribe your voice. Please enable it in System Settings."
        alert.alertStyle = .warning
        alert.addButton(withTitle: "Open System Settings")
        alert.addButton(withTitle: "Cancel")

        if alert.runModal() == .alertFirstButtonReturn {
            if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone") {
                NSWorkspace.shared.open(url)
            }
        }
    }

    @objc private func openSettings() {
        // TODO: Implement settings window
        print("Settings not implemented yet")
    }

    @objc private func quit() {
        NSApplication.shared.terminate(nil)
    }

    func showRecordingOverlay() {
        guard overlayWindow == nil else { return }

        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 400, height: 120),
            styleMask: [.borderless],
            backing: .buffered,
            defer: false
        )

        window.isOpaque = false
        window.backgroundColor = .clear
        window.level = .floating
        window.collectionBehavior = [.canJoinAllSpaces, .stationary]

        let contentView = RecordingOverlay(session: recordingSession!)
        window.contentView = NSHostingView(rootView: contentView)

        // Position at top center of screen
        if let screen = NSScreen.main {
            let screenFrame = screen.visibleFrame
            let windowFrame = window.frame
            let x = screenFrame.midX - windowFrame.width / 2
            let y = screenFrame.maxY - windowFrame.height - 20
            window.setFrameOrigin(NSPoint(x: x, y: y))
        }

        window.makeKeyAndOrderFront(nil)
        overlayWindow = window
    }

    func hideRecordingOverlay() {
        overlayWindow?.close()
        overlayWindow = nil
    }
}
