#!/bin/bash
# =============================================================================
# build_dmg.sh
# Builds a macOS DMG installer for Picasso 0.9.6
# Requirements: PyInstaller, create-dmg (brew install create-dmg)
# Usage: bash create_macos_dmg.sh
# =============================================================================

set -e  # Exit immediately on any error

APP_NAME="Picasso"
VERSION="0.9.6"
MAIN_BUNDLE_NAME="Picasso.app"
DMG_NAME="Picasso-macOS-0.9.6"
SPEC_FILE="../pyinstaller/picassow.spec"
DIST_DIR="../pyinstaller/dist"
BUILD_DIR="../pyinstaller/build"
STAGING_DIR="macos_dmg_staging"
APPS_LINK="/Applications"
# Tool definitions: name, argument, icon filename (without .icns extension)
declare -a TOOLS=(
    "Design:design:design"
    "Simulate:simulate:simulate"
    "Localize:localize:localize"
    "Filter:filter:filter"
    "Render:render:render"
    "Average:average:average"
    "SPINNA:spinna:spinna"
    "Server:server:server"
    "Nanotron:nanotron:nanotron"
    "Toraw:toraw:toraw"
)
# -----------------------------------------------------------------------------
# Step 1: Run PyInstaller to build the main .app bundle
# -----------------------------------------------------------------------------
echo ">>> Building main .app bundle with PyInstaller..."
pyinstaller "$SPEC_FILE" \
    --distpath "$DIST_DIR" \
    --workpath "$BUILD_DIR" \
    --noconfirm

MAIN_APP_PATH="$DIST_DIR/$MAIN_BUNDLE_NAME"

if [ ! -d "$MAIN_APP_PATH" ]; then
    echo "ERROR: PyInstaller did not produce $MAIN_APP_PATH"
    exit 1
fi
echo ">>> Main .app bundle created at $MAIN_APP_PATH"

# Get the path to the main executable
MAIN_EXECUTABLE="$MAIN_APP_PATH/Contents/MacOS/picasso"
if [ ! -f "$MAIN_EXECUTABLE" ]; then
    echo "ERROR: Main executable not found at $MAIN_EXECUTABLE"
    exit 1
fi

# Get the resources directory for icons
MAIN_RESOURCES="$MAIN_APP_PATH/Contents/Resources"

# -----------------------------------------------------------------------------
# Step 2: Create separate .app bundles for each tool
# -----------------------------------------------------------------------------
echo ">>> Creating individual .app bundles for each tool..."

for tool_def in "${TOOLS[@]}"; do
    IFS=':' read -r display_name argument icon_name <<< "$tool_def"
    
    app_name="Picasso ${display_name}.app"
    app_path="$DIST_DIR/$app_name"
    
    echo "    Creating $app_name..."
    
    # Create .app bundle structure
    mkdir -p "$app_path/Contents/MacOS"
    mkdir -p "$app_path/Contents/Resources"
    
    # Create launcher script that calls the main executable with the tool argument
    launcher_script="$app_path/Contents/MacOS/launcher"
    cat > "$launcher_script" <<EOF
#!/bin/bash
# Launcher for Picasso $display_name
SCRIPT_DIR="\$(cd "\$(dirname "\$0")" && pwd)"
MAIN_EXEC="/Applications/$MAIN_BUNDLE_NAME/Contents/MacOS/picasso"

# Check if main Picasso.app exists
if [ ! -f "\$MAIN_EXEC" ]; then
    osascript -e 'display dialog "Picasso.app not found in Applications folder. Please install the main Picasso application first." buttons {"OK"} default button "OK" with icon stop'
    exit 1
fi

# Launch the tool
exec "\$MAIN_EXEC" $argument "\$@"
EOF
    chmod +x "$launcher_script"
    
    # Copy the appropriate icon
    icon_source="$MAIN_RESOURCES/picasso/gui/icons/${icon_name}.icns"
    if [ -f "$icon_source" ]; then
        cp "$icon_source" "$app_path/Contents/Resources/icon.icns"
    else
        echo "    WARNING: Icon not found: $icon_source"
        # Fall back to main icon if tool-specific icon doesn't exist
        if [ -f "$MAIN_RESOURCES/icon.icns" ]; then
            cp "$MAIN_RESOURCES/icon.icns" "$app_path/Contents/Resources/icon.icns"
        fi
    fi
    
    # Create Info.plist
    cat > "$app_path/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>launcher</string>
    <key>CFBundleIconFile</key>
    <string>icon.icns</string>
    <key>CFBundleIdentifier</key>
    <string>org.jungmannlab.picasso.${argument}</string>
    <key>CFBundleName</key>
    <string>Picasso ${display_name}</string>
    <key>CFBundleDisplayName</key>
    <string>Picasso ${display_name}</string>
    <key>CFBundleShortVersionString</key>
    <string>${VERSION}</string>
    <key>CFBundleVersion</key>
    <string>${VERSION}</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSPrincipalClass</key>
    <string>NSApplication</string>
</dict>
</plist>
EOF
done

echo ">>> Individual tool bundles created successfully"

# -----------------------------------------------------------------------------
# Step 3: Stage the DMG contents
# -----------------------------------------------------------------------------
echo ">>> Staging DMG contents..."

rm -rf "$STAGING_DIR"
mkdir -p "$STAGING_DIR"

# Copy the main .app bundle
cp -r "$MAIN_APP_PATH" "$STAGING_DIR/$MAIN_BUNDLE_NAME"

# Copy all tool-specific .app bundles
for tool_def in "${TOOLS[@]}"; do
    IFS=':' read -r display_name argument icon_name <<< "$tool_def"
    app_name="Picasso ${display_name}.app"
    cp -r "$DIST_DIR/$app_name" "$STAGING_DIR/$app_name"
done

# Note: Applications symlink is created automatically by create-dmg via --app-drop-link

# -----------------------------------------------------------------------------
# Step 4: Build the DMG with create-dmg
# -----------------------------------------------------------------------------
echo ">>> Building DMG with create-dmg..."

# Remove any previous DMG
rm -f "${DMG_NAME}.dmg"

# Build the DMG
# Note: With multiple apps, we'll let create-dmg handle the layout automatically
# or you can manually position each icon if you prefer more control
create-dmg \
    --volname "$APP_NAME $VERSION" \
    --volicon "../logos/localize.icns" \
    --window-pos 200 120 \
    --window-size 800 500 \
    --icon-size 80 \
    --text-size 12 \
    --app-drop-link 650 250 \
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
echo ""
echo " The DMG contains:"
echo "   - Picasso.app (main bundle)"
echo "   - Picasso Design.app"
echo "   - Picasso Simulate.app"
echo "   - Picasso Localize.app"
echo "   - Picasso Filter.app"
echo "   - Picasso Render.app"
echo "   - Picasso Average.app"
echo "   - Picasso SPINNA.app"
echo "   - Picasso Server.app"
echo "   - Picasso Nanotron.app"
echo "   - Picasso Toraw.app"
echo "============================================="
