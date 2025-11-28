# MoshiMac

A native macOS voice-to-text application powered by Kyutai's state-of-the-art STT models via moshi-swift.

## Features

- 🎤 **Global Hotkeys**: Activate transcription from anywhere with `⌘⇧V` or `⌘⇧T`
- ⚡ **Real-time Streaming**: True streaming transcription with <80ms latency
- 🔒 **100% Local**: All processing happens on-device using Metal acceleration
- 🎯 **Direct Text Insertion**: Transcribed text automatically inserted into active application
- 🌍 **Multi-language**: Supports English and French (using kyutai/stt-1b-en_fr-mlx)
- 🧠 **Semantic VAD**: Intelligent voice activity detection

## Requirements

- macOS 14.0+
- Apple Silicon (M1, M2, M3, etc.) for optimal performance
- Microphone access permission
- Accessibility access permission (for direct text insertion)

## Installation

### Building from Source

1. Clone the repository with submodules:
```bash
git clone --recursive https://github.com/yourusername/moshimac.git
cd moshimac
```

2. Build using Swift Package Manager:
```bash
swift build -c release
```

Or open the project in Xcode:
```bash
open Package.swift
```

## Usage

### Global Hotkeys

- **⌘⇧V** (Toggle Recording): Press once to start recording, press again to stop
- **⌘⇧T** (Push-to-Talk): Hold to record, release to stop

### First Run

On first run, MoshiMac will:
1. Request microphone access
2. Request accessibility access (needed to paste text)
3. Download the STT model (~2GB) from HuggingFace

## Architecture

```
MoshiMac/
├── App/                  # Application lifecycle
├── Core/                 # Core STT engine and recording logic
├── Input/                # Hotkey management
├── Output/               # Text insertion logic
├── UI/                   # SwiftUI views
├── Models/               # Data models
└── Vendor/
    └── moshi-swift/      # Kyutai's moshi-swift (submodule)
```

## How It Works

1. **Audio Capture**: Captures audio from microphone at 24kHz mono
2. **Streaming Processing**: Audio is processed in 80ms chunks (1920 samples)
3. **Real-time Transcription**: Each chunk is transcribed using the Kyutai STT model
4. **Text Output**: Transcribed text is automatically pasted into the active application

## Configuration

The app uses sensible defaults, but you can configure:

- Model repository (default: `kyutai/stt-1b-en_fr-mlx`)
- Show live transcript in overlay
- Enable/disable VAD
- Auto-punctuation

## Credits

- **Kyutai STT Models**: [kyutai-labs/delayed-streams-modeling](https://github.com/kyutai-labs/delayed-streams-modeling)
- **moshi-swift**: [kyutai-labs/moshi-swift](https://github.com/kyutai-labs/moshi-swift)
- **MLX Swift**: [ml-explore/mlx-swift](https://github.com/ml-explore/mlx-swift)

## License

MIT License - see LICENSE file for details

Model weights are released under CC-BY 4.0 license by Kyutai.

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
