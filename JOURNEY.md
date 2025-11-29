# MoshiMac Development Journey 🚀

Ce document retrace le parcours complet du développement de MoshiMac, de la conception initiale à l'application fonctionnelle.

## 🎯 Objectif Initial

Créer une application macOS de Speech-to-Text :
- 100% locale (privacy-first)
- Utilisant les modèles state-of-the-art de Kyutai
- Interface simple type VoiceInk (activation par raccourci clavier)
- Optimisée pour Apple Silicon avec Metal

## 📅 Timeline du Développement

### Phase 1 : Setup Initial ✅
**Durée** : Rapide
**Objectif** : Structure de base du projet

- Création du package Swift avec SPM
- Configuration des dépendances MLX Swift
- Setup git et structure de dossiers
- Intégration des fichiers vendor de moshi-swift

**Défis rencontrés** :
- Résolution des conflits de dépendances (Hub vs Transformers)
- Ajustements de compatibilité MLX Swift 0.18+
  - `Foundation.sqrt()` au lieu de `sqrt()`
  - Suppression des `override` keywords
  - Syntaxe `verify: [.all]` au lieu de `verify: .all`

### Phase 2 : La Grande Galère Mimi 😅
**Durée** : Plusieurs heures de debugging intense
**Objectif** : Charger les poids Mimi pour l'encodage audio

#### Tentative 1 : Charger les poids PyTorch directement
**Échec** : Incompatibilité structurelle PyTorch → MLX Swift

```
Error: keyNotFound(path: ["quantizer", "rvq_first", "vq", "layers", "0", "_codebook", "embedding"])
```

Les poids PyTorch de Mimi utilisent une structure de clés différente de ce que MLX Swift attend.

**Leçons apprises** :
- Les poids PyTorch ne peuvent pas être simplement mappés vers MLX
- La structure des modèles diffère entre frameworks
- Tentative de créer un mapping manuel → trop complexe et incomplet

#### Tentative 2 : Utiliser des poids aléatoires (temporaire)
**Succès partiel** : L'app démarre mais transcription gibberish

```swift
let mimi = Mimi(MimiConfig.mimi_2024_07(numCodebooks: 32))
// Pas de chargement de poids → poids aléatoires
```

**Résultat** :
- ✅ L'app compile et tourne
- ✅ Le pipeline fonctionne end-to-end
- ❌ La transcription ne produit que du charabia
- ❌ Pas utilisable en production

#### Tentative 3 : Essayer différents fichiers de tokenizer
**Échec** : Même problème d'incompatibilité

On a essayé plusieurs fichiers :
1. `mimi-e351c8d8@125.safetensors`
2. `mimi-pytorch-e351c8d8@125.safetensors`
3. Différents repos HuggingFace

Tous avec la même erreur de structure de clés.

### Phase 3 : La Solution RustyMimi 🦀
**Durée** : Intense mais efficace
**Objectif** : Créer un binding Rust pour utiliser les vrais poids

#### Décision Architecturale

Après analyse, trois options :
1. **Créer un binding rustymimi (Rust/Candle)** ⭐ CHOISI
2. Convertir manuellement PyTorch → MLX (trop complexe)
3. Utiliser un bridge Python (lent, dépendances)

**Pourquoi Rust ?**
- `rustymimi` de Kyutai charge nativement les poids PyTorch
- Candle (framework Rust) compatible avec safetensors PyTorch
- Performance native (compilé)
- Interopérabilité C FFI bien établie

#### Implémentation du Binding C FFI

**Fichier** : `rustymimi-c/src/lib.rs`

```rust
#[repr(C)]
pub struct MimiTokenizer {
    mimi: Mimi,
    // Internal streaming state
}

#[no_mangle]
pub extern "C" fn mimi_tokenizer_new(
    path: *const c_char,
    num_codebooks: usize
) -> *mut MimiTokenizer

#[no_mangle]
pub extern "C" fn mimi_encode_step(
    tokenizer: *mut MimiTokenizer,
    pcm_data: *const f32,
    samples: usize,
    out_codes: *mut *mut u32,
    out_codebooks: *mut usize,
    out_steps: *mut usize
) -> i32

#[no_mangle]
pub extern "C" fn mimi_reset(tokenizer: *mut MimiTokenizer)

#[no_mangle]
pub extern "C" fn mimi_tokenizer_free(tokenizer: *mut MimiTokenizer)
```

**Compilation** :
```bash
cd rustymimi-c
cargo build --release --target aarch64-apple-darwin
cp target/release/librustymimi_c.dylib ../Sources/MoshiMac/Resources/lib/
```

**Taille** : 2.1 MB (très raisonnable)

#### Swift Wrapper

**Fichier** : `Sources/MoshiMac/Vendor/RustyMimi.swift`

Défis rencontrés :
1. **Visibilité des symboles C** : Résolu avec `@_silgen_name`
2. **Conversion MLXArray** : Aplatir en 1D puis reshaper
3. **Gestion mémoire** : `deinit` pour libérer le tokenizer Rust

```swift
@_silgen_name("mimi_tokenizer_new")
func mimi_tokenizer_new(_ path: UnsafePointer<CChar>, _ numCodebooks: Int) -> OpaquePointer?

public class RustyMimiTokenizer {
    private var tokenizer: OpaquePointer?

    public init(weightsPath: String, numCodebooks: Int = 32) throws {
        tokenizer = mimi_tokenizer_new(weightsPath, numCodebooks)
        // ...
    }

    public func encodeStep(_ pcm: MLXArray) throws -> MLXArray {
        // Conversion PCM → codes audio via Rust
        // ...
    }

    deinit {
        if let tokenizer = tokenizer {
            mimi_tokenizer_free(tokenizer)
        }
    }
}
```

#### Intégration ASR

**Nouveau fichier** : `Sources/MoshiMac/Core/ASREngine.swift`

Remplace l'ancien `ASR.swift` qui attendait un `Mimi` MLX.

```swift
class ASREngine {
    let lm: LM
    let vocab: [Int: String]
    let mimi: RustyMimiTokenizer  // ← Rust au lieu de MLX

    func onPcmInput(_ pcm: MLXArray) -> [String] {
        let codes = try mimi.encodeStep(pcm)  // ← Rust encode
        // Le reste du pipeline reste identique
        // codes → LM → tokens → texte
    }
}
```

### Phase 4 : Debugging Final 🐛

**Problèmes rencontrés** :

1. **Erreur compilation** : `Cannot find 'mimi_tokenizer_new' in scope`
   - **Solution** : Ajout des déclarations `@_silgen_name`

2. **Erreur type** : `Cannot convert value of type '[[[Int32]]]' to expected argument type '[Int]'`
   - **Solution** : Aplatir en 1D avant de créer MLXArray
   ```swift
   var flatCodes: [Int32] = []
   for cb in 0..<outCodebooks {
       for step in 0..<outSteps {
           flatCodes.append(Int32(codes[0][cb][step]))
       }
   }
   let mlxArray = MLXArray(flatCodes)
   return mlxArray.reshaped([1, outCodebooks, outSteps])
   ```

3. **Metal shaders manquants**
   - **Cause** : `swift build` ne compile pas les shaders Metal
   - **Solution** : MUST use Xcode for building

### Phase 5 : UI Complete 🎨

Une fois la transcription fonctionnelle, focus sur l'UI :

1. **Menu Bar App**
   - Icône dans la barre système
   - Menu avec status et actions

2. **Recording Overlay**
   - Fenêtre flottante transparente
   - Timer, status, transcription live

3. **Settings Window**
   - Configuration modèle
   - Raccourcis clavier
   - Préférences d'affichage

4. **Text Insertion**
   - Copie automatique vers clipboard
   - Simulation Cmd+V pour collage auto
   - Gestion permissions Accessibility

## 📊 Architecture Finale

```
┌─────────────────────────────────────────────────┐
│                 MoshiMac App                    │
├─────────────────────────────────────────────────┤
│                                                 │
│  Microphone (24kHz PCM)                        │
│         ↓                                       │
│  RustyMimiTokenizer (Rust/Candle FFI)         │
│    • Charge mimi-pytorch-e351c8d8@125.st      │
│    • Encode PCM → codes [1, 32, steps]        │
│         ↓                                       │
│  ASREngine (Swift/MLX)                         │
│    • LM (Language Model 1B params)             │
│    • Traite codes → text tokens                │
│         ↓                                       │
│  Vocabulaire (8000 tokens)                     │
│    • Décode tokens → texte                     │
│         ↓                                       │
│  TextInserter                                  │
│    • Copie vers clipboard                      │
│    • Colle dans app active                     │
│                                                 │
└─────────────────────────────────────────────────┘
```

## 🎓 Leçons Apprises

### Techniques

1. **Incompatibilité inter-frameworks** : PyTorch ≠ MLX Swift
   - Les poids ne sont pas directement compatibles
   - La structure des modèles diffère
   - Le mapping manuel est complexe et fragile

2. **C FFI est puissant mais délicat**
   - `@_silgen_name` évite le bridging header
   - Gestion mémoire critique (leaks possibles)
   - Conversion de types Swift ↔ C nécessite attention

3. **MLXArray est particulier**
   - Créer un array 3D nécessite aplatissement 1D
   - `reshaped()` après création
   - Types stricts (Int32, Float, etc.)

4. **Metal shaders nécessitent Xcode**
   - `swift build` insuffisant
   - Xcode compile `.metal` files automatiquement

5. **Permissions macOS sont complexes**
   - Microphone : Requis
   - Accessibility : Optionnel mais mieux
   - Peuvent nécessiter redémarrage app

### Stratégiques

1. **Quand bloquer, changer d'approche**
   - PyTorch → MLX mapping = impasse
   - Rust FFI = solution élégante

2. **Utiliser les bons outils**
   - Candle pour PyTorch en Rust
   - MLX pour inference sur Metal
   - Chacun dans son domaine

3. **Documentation et debug messages**
   - Console logs avec emojis (✅ ⚠️ 📋)
   - Messages clairs pour l'utilisateur
   - Fallbacks gracieux

## 📦 Fichiers Téléchargés

Au premier lancement, l'app télécharge :

| Fichier | Taille | Usage |
|---------|--------|-------|
| `model.safetensors` | 1.98 GB | Language Model (LM) |
| `mimi-pytorch-e351c8d8@125.safetensors` | 385 MB | Mimi Encoder (Rust) |
| `tokenizer_spm_8k_0.json` | 158 KB | Vocabulaire 8K tokens |
| **Total** | **~2.4 GB** | |

Cache HuggingFace : `~/.cache/huggingface/`

## 🎉 Résultat Final

**MoshiMac fonctionne ! 🎊**

Workflow utilisateur :
1. ⌘⇧V pour commencer l'enregistrement
2. Parler dans le micro
3. ⌘⇧V pour arrêter
4. Texte transcrit collé automatiquement (ou dans clipboard)

Performance :
- Transcription précise en français et anglais
- Latence ~0.5s (streaming)
- 100% local, pas d'internet requis
- Optimisé Metal pour Apple Silicon

## 🙏 Remerciements

- **Kyutai** pour les modèles STT state-of-the-art
- **Apple** pour MLX Swift et Metal
- **Rust/Candle** pour l'interop PyTorch
- **Claude** pour le pair programming intensif 🤖

---

**Développé avec détermination et beaucoup de debugging** 💪

_"En codant, on apprend. En débuggant, on grandit."_
