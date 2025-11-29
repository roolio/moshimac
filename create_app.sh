#!/bin/bash
# Create MoshiMac.app bundle from existing Xcode build

set -e

APP_NAME="MoshiMac"
APP_BUNDLE="$PWD/$APP_NAME.app"
CONTENTS="$APP_BUNDLE/Contents"

# Find the executable from Xcode's DerivedData
DERIVED_DATA="$HOME/Library/Developer/Xcode/DerivedData"
EXECUTABLE=$(find "$DERIVED_DATA" -path "*moshimac*/Build/Products/Debug/MoshiMac" -type f 2>/dev/null | head -1)

if [ -z "$EXECUTABLE" ]; then
    # Try Release
    EXECUTABLE=$(find "$DERIVED_DATA" -path "*moshimac*/Build/Products/Release/MoshiMac" -type f 2>/dev/null | head -1)
fi

if [ -z "$EXECUTABLE" ]; then
    echo "❌ Error: Could not find MoshiMac executable in DerivedData"
    echo "   Please build the project in Xcode first (Cmd+B)"
    exit 1
fi

echo "Found executable: $EXECUTABLE"

# Create app bundle structure
rm -rf "$APP_BUNDLE"
mkdir -p "$CONTENTS/MacOS"
mkdir -p "$CONTENTS/Resources"
mkdir -p "$CONTENTS/Frameworks"

# Copy executable
cp "$EXECUTABLE" "$CONTENTS/MacOS/$APP_NAME"

# Copy dylib
cp "Sources/MoshiMac/Resources/lib/librustymimi_c.dylib" "$CONTENTS/Frameworks/"

# Copy MLX Metal bundle (required for Metal shaders)
MLX_BUNDLE=$(dirname "$EXECUTABLE")/mlx-swift_Cmlx.bundle
if [ -d "$MLX_BUNDLE" ]; then
    cp -R "$MLX_BUNDLE" "$CONTENTS/Resources/"
    echo "Copied MLX Metal bundle"
else
    echo "⚠️  Warning: MLX Metal bundle not found at $MLX_BUNDLE"
fi

# Copy app icon
if [ -f "Sources/MoshiMac/Resources/AppIcon.icns" ]; then
    cp "Sources/MoshiMac/Resources/AppIcon.icns" "$CONTENTS/Resources/"
    echo "Copied app icon"
fi

# Fix dylib path in executable
install_name_tool -change "librustymimi_c.dylib" "@executable_path/../Frameworks/librustymimi_c.dylib" "$CONTENTS/MacOS/$APP_NAME" 2>/dev/null || true

# Create Info.plist
cat > "$CONTENTS/Info.plist" << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>MoshiMac</string>
    <key>CFBundleIdentifier</key>
    <string>com.moshimac.app</string>
    <key>CFBundleName</key>
    <string>MoshiMac</string>
    <key>CFBundleDisplayName</key>
    <string>MoshiMac</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>LSMinimumSystemVersion</key>
    <string>14.0</string>
    <key>LSUIElement</key>
    <true/>
    <key>NSMicrophoneUsageDescription</key>
    <string>MoshiMac needs microphone access to transcribe your voice.</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
EOF

# Sign the app (ad-hoc signing for local use)
codesign --force --deep --sign - "$APP_BUNDLE" 2>/dev/null || echo "⚠️  Signing skipped (install may require right-click > Open)"

echo ""
echo "✅ App bundle created: $APP_BUNDLE"
echo ""
echo "To install:"
echo "  mv \"$APP_BUNDLE\" /Applications/"
echo ""
echo "First launch: right-click → Open (to bypass Gatekeeper)"
