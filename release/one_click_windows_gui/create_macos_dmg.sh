#!/bin/bash
# =============================================================================
# build_dmg.sh
# Builds a macOS DMG installer for Picasso 0.9.6
# Requirements: PyInstaller, create-dmg (brew install create-dmg)
# Usage: bash build_dmg.sh
# =============================================================================

set -e  # Exit immediately on any error

APP_NAME="Picasso"
VERSION="0.9.6"
BUNDLE_NAME="Picasso.app"
DMG_NAME="Picasso-macOS-0.9.6"
SPEC_FILE="picasso.spec"
DIST_DIR="dist"
BUILD_DIR="build"
STAGING_DIR="dmg_staging"
APPS_LINK="/Applications"

# -----------------------------------------------------------------------------
# Step 1: Run PyInstaller to build the .app bundle
# -----------------------------------------------------------------------------
echo ">>> Building .app bundle with PyInstaller..."
pyinstaller "$SPEC_FILE" \
    --distpath "$DIST_DIR" \
    --workpath "$BUILD_DIR" \
    --noconfirm

APP_PATH="$DIST_DIR/$BUNDLE_NAME"

if [ ! -d "$APP_PATH" ]; then
    echo "ERROR: PyInstaller did not produce $APP_PATH"
    exit 1
fi
echo ">>> .app bundle created at $APP_PATH"

# -----------------------------------------------------------------------------
# Step 2: Create tool-specific launcher scripts inside the .app
#
# Since all tools share one .app, we create small wrapper shell scripts
# that launch picassow with the right argument. These can be placed on the
# Dock or opened from Finder as aliases.
#
# The main .app bundle itself launches without arguments (shows a tool picker
# or defaults to the first tool — depends on your picasso_pyinstaller.py).
# -----------------------------------------------------------------------------
echo ">>> Creating per-tool launcher scripts..."

MACOS_DIR="$APP_PATH/Contents/MacOS"
TOOLS=("design" "simulate" "localize" "filter" "render" "average" "spinna" "server")

for tool in "${TOOLS[@]}"; do
    launcher="$MACOS_DIR/picasso_$tool"
    cat > "$launcher" <<EOF
#!/bin/bash
# Launcher for picasso $tool
DIR="\$(cd "\$(dirname "\$0")" && pwd)"
"\$DIR/picassow" $tool "\$@"
EOF
    chmod +x "$launcher"
    echo "    Created launcher: picasso_$tool"
done

# -----------------------------------------------------------------------------
# Step 3: Stage the DMG contents
# -----------------------------------------------------------------------------
echo ">>> Staging DMG contents..."

rm -rf "$STAGING_DIR"
mkdir -p "$STAGING_DIR"

# Copy the .app bundle into staging
cp -r "$APP_PATH" "$STAGING_DIR/$BUNDLE_NAME"

# Create a symlink to /Applications so users can drag-and-drop
ln -s "$APPS_LINK" "$STAGING_DIR/Applications"

# Optional: copy a background image if you have one
# mkdir -p "$STAGING_DIR/.background"
# cp assets/dmg_background.png "$STAGING_DIR/.background/background.png"

# -----------------------------------------------------------------------------
# Step 4: Build the DMG with create-dmg
# -----------------------------------------------------------------------------
echo ">>> Building DMG with create-dmg..."

# Remove any previous DMG
rm -f "${DMG_NAME}.dmg"

create-dmg \
    --volname "$APP_NAME $VERSION" \
    --volicon "logos/localize.icns" \
    --window-pos 200 120 \
    --window-size 600 400 \
    --icon-size 100 \
    --icon "$BUNDLE_NAME" 150 185 \
    --hide-extension "$BUNDLE_NAME" \
    --app-drop-link 450 185 \
    --no-internet-enable \
    "${DMG_NAME}.dmg" \
    "$STAGING_DIR"

# -----------------------------------------------------------------------------
# Step 5: Cleanup staging directory
# -----------------------------------------------------------------------------
echo ">>> Cleaning up staging directory..."
rm -rf "$STAGING_DIR"

echo ""
echo "============================================="
echo " Build complete!"
echo " Output: ${DMG_NAME}.dmg"
echo "============================================="
