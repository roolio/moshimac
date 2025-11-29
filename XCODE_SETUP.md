# Configuration Xcode pour RustyMimi

## ⚠️ Étapes Importantes

### 1. Ouvrir le projet dans Xcode
```bash
cd /Users/julien.laugel/Dropbox/code/moshimac
open Package.swift
```

### 2. Configuration du Runtime Library Path

Pour que l'app trouve `librustymimi_c.dylib` au runtime :

**Dans Xcode :**
1. Sélectionnez le projet "moshimac" dans le navigateur
2. Sélectionnez la target "MoshiMac"
3. Allez dans l'onglet **Build Settings**
4. Cherchez "Runpath Search Paths" (ou `LD_RUNPATH_SEARCH_PATHS`)
5. Ajoutez :
   ```
   @executable_path
   @loader_path
   ```

### 3. Ajouter un Build Phase (Optionnel mais recommandé)

Pour copier automatiquement la dylib :

1. Dans la target "MoshiMac", allez dans **Build Phases**
2. Cliquez sur **+** → **New Run Script Phase**
3. Nommez-la "Copy RustyMimi Library"
4. Ajoutez le script :
   ```bash
   DYLIB_SOURCE="${SRCROOT}/Sources/MoshiMac/Resources/lib/librustymimi_c.dylib"

   if [ -f "$DYLIB_SOURCE" ]; then
       echo "Copying rustymimi dylib"
       cp "$DYLIB_SOURCE" "${BUILT_PRODUCTS_DIR}/"
   fi
   ```

### 4. Alternative : Copie Manuelle

Si vous ne voulez pas configurer le build phase :

```bash
cp Sources/MoshiMac/Resources/lib/librustymimi_c.dylib \
   ~/Library/Developer/Xcode/DerivedData/moshimac-*/Build/Products/Debug/
```

## 🔨 Build et Run

1. Dans Xcode, sélectionnez le scheme **MoshiMac**
2. Sélectionnez **My Mac** comme destination
3. Appuyez sur **⌘B** pour builder
4. Si le build réussit, appuyez sur **⌘R** pour run

## 🐛 Erreurs Possibles

### Error: dyld: Library not loaded: librustymimi_c.dylib

**Solution 1** : Vérifier que la dylib est dans le même dossier que l'exécutable
```bash
ls ~/Library/Developer/Xcode/DerivedData/moshimac-*/Build/Products/Debug/
```

**Solution 2** : Copier manuellement :
```bash
cp Sources/MoshiMac/Resources/lib/librustymimi_c.dylib \
   ~/Library/Developer/Xcode/DerivedData/moshimac-*/Build/Products/Debug/
```

**Solution 3** : Ajouter au DYLD_LIBRARY_PATH (temporaire) :
```bash
export DYLD_LIBRARY_PATH="${DYLD_LIBRARY_PATH}:${PWD}/Sources/MoshiMac/Resources/lib"
```

### Error: Symbol not found

**Cause** : La dylib n'a pas été compilée pour la bonne architecture

**Solution** : Recompiler rustymimi-c :
```bash
cd rustymimi-c
~/.cargo/bin/cargo build --release --target aarch64-apple-darwin
```

### Error: Cannot find 'mimi_tokenizer_new' in scope

**Cause** : Les déclarations C ne sont pas visibles

**Solution** : Vérifier que `RustyMimi.swift` contient les déclarations `@_silgen_name`

## ✅ Test Rapide

Une fois l'app lancée, vous devriez voir dans les logs :
```
Initializing STT Engine...
Downloading model from kyutai/stt-1b-en_fr-mlx...
Using cached file: ...
Downloading Mimi weights...
Loading Mimi tokenizer from Rust...
Mimi tokenizer loaded successfully!
Loading models into memory...
Warming up LM...
STT Engine ready!
```

Si vous voyez "Mimi tokenizer loaded successfully!", c'est bon ! 🎉

## 📊 Taille Fichiers

- `librustymimi_c.dylib` : 2.1 MB
- `mimi-pytorch-e351c8d8@125.safetensors` : 385 MB
- `model.safetensors` : 1.98 GB

## 🚀 Prochaine Étape

Si le build et le lancement fonctionnent, testez la transcription :
1. Parlez dans le micro après avoir appuyé sur ⌘⇧V
2. Vérifiez les logs pour voir les tokens générés
3. Le texte devrait apparaître dans l'UI

Bonne chance ! 🍀
