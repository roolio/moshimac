# MoshiMac Development Guide

## Current Status

✅ **Phase 1: Project Structure** - COMPLETE
- Project skeleton created
- Core architecture defined
- UI components implemented
- Build system configured

🚧 **Phase 2: moshi-swift Integration** - IN PROGRESS
- Need to integrate actual ASR class from moshi-swift
- Replace placeholder AudioCapture with MicrophoneCapture
- Connect model loading and inference

⏳ **Phase 3: Testing & Polish** - TODO
- Test on real hardware
- Fine-tune UI/UX
- Add settings panel
- Implement app configs

## Next Steps

### 1. Integrate moshi-swift Core Components

We need to copy/adapt these files from `Vendor/moshi-swift/` into our project:

#### Required Files from MoshiLib:
- `ASR.swift` - Core STT engine
- `LM.swift` - Language model
- `Mimi.swift` - Audio codec
- `Streaming.swift` - Streaming utilities
- `Conv.swift`, `Transformer.swift`, etc. - Model components

#### Required Files from MoshiCLI:
- `AudioRT.swift` - Contains `MicrophoneCapture` class
- `Audio.swift` - Audio utilities

#### Integration Plan:

```swift
// In Sources/MoshiMac/Vendor/
// Copy these files from moshi-swift:
Sources/MoshiMac/Vendor/
├── ASR.swift              # from MoshiLib/ASR.swift
├── LM.swift               # from MoshiLib/LM.swift
├── Mimi.swift             # from MoshiLib/Mimi.swift
├── Streaming.swift        # from MoshiLib/Streaming.swift
├── AudioCapture.swift     # from MoshiCLI/AudioRT.swift (MicrophoneCapture)
└── ... (other model files)
```

### 2. Update STTEngine.swift

Replace the placeholder implementation with actual moshi-swift integration:

```swift
import MLX
import MLXNN

class STTEngine: ObservableObject {
    private var asr: ASR?
    private var mimi: Mimi?
    private var moshi: LM?
    private var audioCapture: MicrophoneCapture?  // Real implementation

    func initialize() async {
        // 1. Download model
        let modelURL = try await downloadFromHub(
            id: "kyutai/stt-1b-en_fr-mlx",
            filename: "model.safetensors"
        )

        // 2. Load Mimi codec
        mimi = try makeMimi(numCodebooks: 32)

        // 3. Load Moshi model
        let cfg = LmConfig.asr1b()
        moshi = try makeMoshi(modelURL, cfg)

        // 4. Warmup
        mimi?.warmup()
        moshi?.warmup()

        // 5. Create ASR
        let vocab = try loadVocab(cfg)
        asr = ASR(moshi!, mimi!, vocab: vocab)
        asr?.reset()

        isReady = true
    }

    func startListening() {
        audioCapture = MicrophoneCapture()
        audioCapture?.startCapturing()

        Task {
            while let pcm = audioCapture?.receive() {
                let pcmArray = MLXArray(pcm)[.newAxis, .newAxis]
                let tokens = asr?.onPcmInput(pcmArray) ?? []

                await MainActor.run {
                    for token in tokens {
                        currentTranscript += token
                    }
                }
            }
        }
    }
}
```

### 3. Add Model Download Utilities

Create `Sources/MoshiMac/Core/ModelManager.swift`:

```swift
import Hub
import Foundation

class ModelManager {
    static func downloadModel(repo: String, filename: String) async throws -> URL {
        // Use HuggingFace Hub to download model
        let hub = HubApi()
        let repoRef = Hub.Repo(id: repo)

        let url = try await Hub.snapshot(from: repoRef, matching: filename) { progress in
            print("Downloading \(filename): \(Int(progress.fractionCompleted * 100))%")
        }

        return url.appending(path: filename)
    }
}
```

### 4. Build & Test

```bash
cd /Users/julien.laugel/Dropbox/code/moshimac
make setup
make build
make run
```

## Testing Checklist

- [ ] App launches and shows menu bar icon
- [ ] Model downloads successfully on first run
- [ ] Microphone permission requested
- [ ] Accessibility permission requested
- [ ] ⌘⇧V hotkey starts/stops recording
- [ ] ⌘⇧T push-to-talk works
- [ ] Recording overlay appears
- [ ] Audio is captured at 24kHz
- [ ] Transcription appears in real-time
- [ ] Text is pasted into active app
- [ ] VAD detects end of speech
- [ ] App can be quit cleanly

## Known Issues

1. **Placeholder STT Engine**: Currently using a mock implementation. Need to integrate real moshi-swift ASR.
2. **No Model Download**: Model download logic not implemented yet.
3. **No Settings Panel**: Settings UI not created yet.
4. **No App Icon**: Need to create and add app icon.

## Architecture Decisions

### Why Swift Package Manager?

- Native Swift tooling
- Simple dependency management
- Easy to build and distribute
- No Xcode project complexity

### Why moshi-swift instead of Python?

- No Python bridge overhead
- Native macOS integration
- Better performance
- Easier distribution (single binary)
- Lower memory footprint

### Why Menu Bar App?

- Always accessible
- Low UI footprint
- Follows macOS conventions
- Easy to invoke with hotkeys

## Performance Targets

- **Cold start**: < 5s (first model load)
- **Warm start**: < 0.5s (app launch)
- **Transcription latency**: < 500ms (80ms frame + processing)
- **Memory usage**: < 2GB (with model loaded)
- **CPU usage**: < 30% (while recording on M2 Max)

## Future Features

- [ ] Multiple model support (switch between 1B and 2.6B)
- [ ] Power Mode (per-app configurations)
- [ ] Personal dictionary
- [ ] Custom commands (e.g., "new line", "period")
- [ ] Punctuation post-processing
- [ ] Export/import settings
- [ ] Keyboard shortcuts customization
- [ ] Recording history
- [ ] Multiple languages
- [ ] Background noise filtering
