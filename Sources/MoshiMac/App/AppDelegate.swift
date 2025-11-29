// Copyright (c) MoshiMac
// This source code is licensed under the MIT license

import AppKit
import SwiftUI
import AVFoundation
import Combine

class AppDelegate: NSObject, NSApplicationDelegate {
    private var statusItem: NSStatusItem?
    private var recordingSession: RecordingSession?
    private var hotkeyManager: HotkeyManager?
    private var sttEngine: STTEngine?
    private var overlayWindow: NSWindow?
    private var cancellables = Set<AnyCancellable>()

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Hide from Dock
        NSApp.setActivationPolicy(.accessory)

        // Create menu bar item
        setupMenuBar()

        // Initialize STT engine
        sttEngine = STTEngine()
        recordingSession = RecordingSession(sttEngine: sttEngine!)

        // Observe recording state changes to show/hide overlay
        observeRecordingState()

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

        // Check accessibility permissions
        checkAccessibilityPermissions()

        print("MoshiMac started successfully")
    }

    private func setupMenuBar() {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)

        if let button = statusItem?.button {
            if let image = NSImage(named: "MenuIcon") {
                image.size = NSSize(width: 18, height: 18)
                button.image = image
            } else {
                button.title = "🎙"
            }
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

    private func observeRecordingState() {
        Task { @MainActor in
            guard let recordingSession = recordingSession else { return }

            recordingSession.$state
                .receive(on: DispatchQueue.main)
                .sink { [weak self] state in
                    guard let self = self else { return }

                    switch state {
                    case .recording, .processing:
                        if self.overlayWindow == nil {
                            self.showRecordingOverlay()
                        }
                    case .idle, .done, .error:
                        self.hideRecordingOverlay()
                    }
                }
                .store(in: &self.cancellables)
        }
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

    private func checkAccessibilityPermissions() {
        // Check if we have accessibility permissions
        let options: NSDictionary = [kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String: true]
        let trusted = AXIsProcessTrustedWithOptions(options)

        if !trusted {
            print("⚠️  Accessibility permissions not granted")
            print("   MoshiMac will copy text to clipboard instead of pasting directly")
            print("   Enable in: System Settings > Privacy & Security > Accessibility")
        } else {
            print("✅ Accessibility permissions granted")
        }
    }

    @objc private func openSettings() {
        let settingsView = SettingsWindow()
        let hostingController = NSHostingController(rootView: settingsView)

        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 500, height: 400),
            styleMask: [.titled, .closable, .resizable],
            backing: .buffered,
            defer: false
        )

        window.title = "MoshiMac Settings"
        window.contentViewController = hostingController
        window.center()
        window.makeKeyAndOrderFront(nil)

        // Activate the app to bring window to front
        NSApp.activate(ignoringOtherApps: true)
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
        window.isReleasedWhenClosed = false

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

        window.orderFrontRegardless()
        overlayWindow = window
    }

    func hideRecordingOverlay() {
        overlayWindow?.close()
        overlayWindow = nil
    }
}
