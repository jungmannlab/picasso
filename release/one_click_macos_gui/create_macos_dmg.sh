#!/bin/bash
# =============================================================================
# create_macos_dmg.sh
# Builds a macOS DMG installer for Picasso
# Requirements: conda, create-dmg >= 1.3.0 (brew install create-dmg)
# Usage: bash create_macos_dmg.sh, from an environment in which
#        "import picasso" works (the version is read before the build
#        environment exists)
# =============================================================================

set -e  # Exit immediately on any error

# Initialize conda for this shell session
eval "$(conda shell.bash hook)"

APP_NAME="Picasso"
VERSION=$(python3 -c "from picasso.version import __version__; print(__version__)")
MAIN_BUNDLE_NAME="Picasso.app"
DMG_NAME="Picasso-v$VERSION-macOS-Apple-Silicon"
PYINSTALLER_FILE="../pyinstaller/picasso_pyinstaller.py"
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
)

# -----------------------------------------------------------------------------
# Step 0: Create a conda environment and prepare the package
# -----------------------------------------------------------------------------
echo ">>> Setting up conda environment and preparing package..."
# Create the conda environment. A previous run that failed part-way
# (set -e) leaves the environment behind, and conda refuses to create
# an existing one, so remove any leftover first.
echo "Creating conda environment 'installer'..."
conda env remove -n installer -y > /dev/null 2>&1 || true
conda create -n installer python=3.14.4 -y
conda activate installer
pip install build
cd ../..
python -m build
pip install dist/picassosr-$VERSION-py3-none-any.whl[installer]
cd release/one_click_macos_gui

# -----------------------------------------------------------------------------
# Step 1: Clean previous build artifacts and run PyInstaller
# -----------------------------------------------------------------------------
echo ">>> Cleaning previous build artifacts..."
if [ -d "$DIST_DIR" ]; then
    echo "Removing $DIST_DIR..."
    rm -rf "$DIST_DIR"
fi
if [ -d "$BUILD_DIR" ]; then
    echo "Removing $BUILD_DIR..."
    rm -rf "$BUILD_DIR"
fi

echo ">>> Building main .app bundle with PyInstaller..."
pyinstaller "$PYINSTALLER_FILE" \
    --onedir \
    --windowed \
    --collect-all picasso \
    --collect-all streamlit \
    --copy-metadata streamlit \
    --copy-metadata imageio \
    --collect-submodules matplotlib.backends \
    --name picasso \
    --icon ../logos/localize.icns \
    --osx-bundle-identifier org.jungmannlab.picasso \
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

# ---------------------------------------------------------------------------
# Step 1a: Drop a qt.conf into the bundle.
# On macOS 26 (Tahoe), Qt's static initializers call CFBundleGetMainBundle()
# during dlopen of QtCore. If that returns NULL (which happens when Python is
# the launching process and the Info.plist identifier isn't a reverse-DNS
# string), Qt passes NULL into CFBundleCopyBundleURL and the app crashes
# instantly with SIGSEGV in __CFCheckCFInfoPACSignature. Shipping a qt.conf
# short-circuits that lookup: Qt reads paths from the file instead of asking
# CoreFoundation. Prefix = . pins Qt to the bundle's own Resources dir.
# ---------------------------------------------------------------------------
cat > "$MAIN_RESOURCES/qt.conf" <<'EOF'
[Paths]
Prefix = .
EOF
echo ">>> Wrote qt.conf to $MAIN_RESOURCES/qt.conf"

# ---------------------------------------------------------------------------
# Step 1b: Fix HDF5 library conflict (h5py 3.15.1 vs tables 3.10.1), which
# happened at version 0.9..6
# Both h5py and tables ship their own libhdf5.310.dylib from different HDF5
# versions. PyInstaller may bundle the older (tables) copy, causing a symbol
# mismatch at runtime. Force h5py's versions into the bundle.
# ---------------------------------------------------------------------------
H5PY_DYLIBS=$(python3 -c "import h5py, os; print(os.path.join(os.path.dirname(h5py.__file__), '.dylibs'))" 2>/dev/null || true)
if [ -d "$H5PY_DYLIBS" ]; then
    FRAMEWORKS_DIR="$MAIN_APP_PATH/Contents/Frameworks"
    # Fall back to _internal for --onedir non-macOS-bundle layouts
    if [ ! -d "$FRAMEWORKS_DIR" ]; then
        FRAMEWORKS_DIR="$MAIN_APP_PATH/Contents/MacOS/_internal"
    fi
    if [ -d "$FRAMEWORKS_DIR" ]; then
        echo ">>> Fixing HDF5 libraries: copying h5py's dylibs into bundle..."
        for dylib in "$H5PY_DYLIBS"/libhdf5*.dylib; do
            if [ -f "$dylib" ]; then
                cp -f "$dylib" "$FRAMEWORKS_DIR/"
                echo "    Copied $(basename "$dylib")"
            fi
        done
    fi
fi

# ---------------------------------------------------------------------------
# Step 1c: Re-sign the main bundle (ad-hoc). PyInstaller signs the bundle
# when it builds, but the HDF5 dylib swap above invalidates that signature.
# An invalid signature on Apple Silicon triggers PAC/codesigning faults at
# load time, including crashes inside CoreFoundation. --deep --force resigns
# everything inside the bundle with an ad-hoc identity.
# ---------------------------------------------------------------------------
echo ">>> Re-signing $MAIN_APP_PATH (ad-hoc)..."
codesign --force --deep --sign - "$MAIN_APP_PATH"

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
MAIN_APP="/Applications/$MAIN_BUNDLE_NAME"
MAIN_EXEC="\$MAIN_APP/Contents/MacOS/picasso"

if [ ! -f "\$MAIN_EXEC" ]; then
    osascript -e 'display dialog "Picasso.app not found in Applications folder. Please install the main Picasso application first." buttons {"OK"} default button "OK" with icon stop'
    exit 1
fi

# Set DYLD_LIBRARY_PATH so PyInstaller can find its bundled libraries
INTERNAL_DIR="\$MAIN_APP/Contents/MacOS/_internal"
if [ -d "\$INTERNAL_DIR" ]; then
    export DYLD_LIBRARY_PATH="\$INTERNAL_DIR:\$DYLD_LIBRARY_PATH"
fi

# Launch natively. LaunchServices starts a script-only bundle under
# Rosetta unless its Info.plist says otherwise (see LSArchitecturePriority
# below), and a translated shell passes an x86_64 spawn preference on to
# every descendant: the arm64-only Picasso binary still runs natively,
# but the "uname -p" that Python's platform.processor() spawns then
# answers "i386", rubicon-objc (used by wgpu) believes it is on Intel and
# looks up objc_msgSendSuper_stret, which does not exist on arm64 - and
# GPU rendering silently falls back to the CPU. "arch -arm64" resets the
# preference even if a user ticked "Open using Rosetta".
exec arch -arm64 "\$MAIN_EXEC" $argument "\$@"
EOF
    chmod +x "$launcher_script"
    
    # Copy the appropriate icon
    icon_source="../logos/${icon_name}.icns"
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
    <!-- a bundle whose executable is a shell script has no Mach-O
         architecture, and LaunchServices then runs it under Rosetta;
         declare arm64 so the launcher (and everything it spawns) is
         native - see the note in the launcher script -->
    <key>LSArchitecturePriority</key>
    <array>
        <string>arm64</string>
    </array>
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

# Remove any previous DMG, and the read-write scratch image a failed
# create-dmg run leaves behind
rm -f "${DMG_NAME}.dmg" rw.*."${DMG_NAME}.dmg"

# Build the DMG
# Note: With multiple apps, we'll let create-dmg handle the layout automatically
# or you can manually position each icon if you prefer more control.
# create-dmg lays the window out through a Finder AppleScript after
# mounting the image; with an image of this size (~1.7 GB) Finder has not
# registered the volume within the tool's default 2 s wait and the script
# died with "Can't get disk (-1728)" on macOS 26, hence the longer wait
# (create-dmg >= 1.3.0) and the hdiutil retries.
create-dmg \
    --volname "$APP_NAME $VERSION" \
    --volicon "../logos/localize.icns" \
    --window-pos 200 120 \
    --window-size 800 500 \
    --icon-size 80 \
    --text-size 12 \
    --app-drop-link 650 250 \
    --no-internet-enable \
    --applescript-sleep-duration 15 \
    --hdiutil-retries 5 \
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
echo "============================================="

# Delete the conda environment
conda deactivate
conda remove -n installer --all -y
# Delete the .spec file generated by PyInstaller
rm -f picasso.spec
