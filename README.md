# MoshiMac 🎙

A privacy-first, offline Speech-to-Text application for macOS, powered by [Kyutai](https://kyutai.org/)'s state-of-the-art STT models.

<p align="center">
  <img src="https://img.shields.io/badge/macOS-14%2B-blue" alt="macOS 14+">
  <img src="https://img.shields.io/badge/Swift-5.9%2B-orange" alt="Swift 5.9+">
  <img src="https://img.shields.io/badge/Apple%20Silicon-Optimized-green" alt="Apple Silicon">
  <img src="https://img.shields.io/badge/Privacy-100%25%20Local-purple" alt="100% Local">
</p>

## ✨ Features

- 🎯 **One-Click Recording** - Press `⌘⇧V` to start/stop transcription
- 🔒 **100% Local Processing** - No internet required, complete privacy
- ⚡ **Low Latency** - ~0.5s response time with streaming support
- 🌍 **Multilingual** - English and French support (via stt-1b-en_fr model)
- 🚀 **Apple Silicon Optimized** - Metal-accelerated inference with MLX
- 📋 **Auto-Paste** - Text automatically inserted into active application
- 🎨 **Clean UI** - Menu bar app with floating overlay
- ⚙️ **Customizable** - Configurable shortcuts, display options, and more

## 🚀 Quick Start

### Requirements

- macOS 14.0+
- Apple Silicon (M1/M2/M3/M4)
- ~3 GB free disk space (for models)
- Microphone access
- Xcode 16.0+ (for building)

### Build & Run

```bash
# Clone repository
git clone https://github.com/yourusername/moshimac.git
cd moshimac

# Open in Xcode
open Package.swift

# Build and run with ⌘R
```

On first launch, MoshiMac will:
1. Request microphone permissions
2. Download models (~2.4 GB) from HuggingFace
3. Initialize the STT engine (~10-30 seconds)

## 📖 Usage

### Basic Workflow

1. **Start Recording**: Press `⌘⇧V` (or your custom shortcut)
2. **Speak**: An overlay appears showing recording status
3. **Stop Recording**: Press `⌘⇧V` again
4. **Get Text**: Transcription is automatically pasted (or copied to clipboard)

### Keyboard Shortcuts

- `⌘⇧V` - Toggle recording (press once to start, once to stop)
- `⌘⇧T` - Push-to-talk (hold to record, release to stop)
- `⌘,` - Open settings

### Settings

Access settings from the menu bar icon:
- **Model Configuration**: Choose model repository
- **Display Options**: Show/hide live transcript in overlay
- **Advanced Features**: Auto-punctuation, Voice Activity Detection
- **Shortcuts**: Customize keyboard shortcuts

## 🏗 Architecture

```
Audio Input (24kHz PCM)
    ↓
RustyMimiTokenizer (Rust/Candle FFI)
    • Loads PyTorch Mimi weights
    • Encodes audio → codes [1, 32, steps]
    ↓
ASREngine (Swift/MLX)
    • Language Model (1B parameters)
    • Processes codes → text tokens
    ↓
Vocabulary Decoder (8K tokens)
    • Converts tokens → readable text
    ↓
Text Insertion
    • Clipboard + Auto-paste (Cmd+V simulation)
```

### Key Technologies

- **[MLX Swift](https://github.com/ml-explore/mlx-swift)** - Apple's ML framework for Metal
- **[Kyutai STT Models](https://github.com/kyutai-labs/moshi)** - State-of-the-art speech recognition
- **[rustymimi](https://github.com/kyutai-labs/moshi/tree/main/rust)** - Rust implementation for Mimi codec
- **Candle** - Rust ML framework for loading PyTorch weights
- **C FFI** - Rust ↔ Swift interoperability

## 🎓 Technical Challenges

The most significant challenge was loading Mimi encoder weights. PyTorch weights are not directly compatible with MLX Swift due to structural differences. After several approaches, we created a Rust C FFI binding to use `rustymimi`, which natively loads PyTorch weights via Candle.

For the full development story, see [JOURNEY.md](JOURNEY.md).

## 📦 Models Downloaded

On first run, MoshiMac downloads:

| File | Size | Purpose |
|------|------|---------|
| `model.safetensors` | 1.98 GB | Language Model |
| `mimi-pytorch-e351c8d8@125.safetensors` | 385 MB | Mimi Encoder |
| `tokenizer_spm_8k_0.json` | 158 KB | Vocabulary |
| **Total** | **~2.4 GB** | |

Models are cached in `~/.cache/huggingface/`.

## 🔐 Privacy & Permissions

MoshiMac requires:
- **Microphone Access** (Required) - To capture your voice
- **Accessibility Access** (Optional) - For auto-paste functionality

All processing happens **100% locally** on your Mac. No data is sent to external servers.

## 🛠 Development

### Project Structure

```
moshimac/
├── Sources/MoshiMac/
│   ├── App/              # AppDelegate, main entry
│   ├── Core/             # STTEngine, ASREngine, ModelManager
│   ├── Input/            # HotkeyManager
│   ├── Output/           # TextInserter
│   ├── UI/               # RecordingOverlay, SettingsWindow
│   ├── Vendor/           # moshi-swift files, RustyMimi wrapper
│   └── Resources/
│       └── lib/          # librustymimi_c.dylib
├── rustymimi-c/          # Rust C FFI binding
└── Package.swift
```

### Building rustymimi-c

```bash
cd rustymimi-c
cargo build --release --target aarch64-apple-darwin
cp target/release/librustymimi_c.dylib ../Sources/MoshiMac/Resources/lib/
```

### Dependencies

Managed via Swift Package Manager:
- `mlx-swift` (>= 0.18.0)
- `swift-transformers` (for HuggingFace downloads)
- `KeyboardShortcuts` (global hotkeys)
- `LaunchAtLogin` (optional auto-start)


## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **[Kyutai Labs](https://kyutai.org/)** for the incredible STT models and rustymimi implementation
- **Apple** for MLX Swift and Metal acceleration
- **HuggingFace** for model hosting and distribution
- **Rust/Candle** for PyTorch interoperability

## 📚 Learn More

- [JOURNEY.md](JOURNEY.md) - Complete development story and technical challenges
- [XCODE_SETUP.md](XCODE_SETUP.md) - Xcode configuration guide
- [RUSTYMIMI_INTEGRATION.md](RUSTYMIMI_INTEGRATION.md) - RustyMimi integration details

## Citation

If you use MoshiMac in research, please cite the Kyutai STT paper:

```bibtex
@techreport{kyutai2025streaming,
      title={Streaming Sequence-to-Sequence Learning with Delayed Streams Modeling},
      author={Neil Zeghidour and Eugene Kharitonov and Manu Orsini and Václav Volhejn and Gabriel de Marmiesse and Edouard Grave and Patrick Pérez and Laurent Mazaré and Alexandre Défossez},
      year={2025},
      eprint={2509.08753},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.08753},
}
```

---

**Made with ❤️ and lots of debugging**

*Transcribe locally, stay private* 🔒
