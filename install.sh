#!/bin/bash
# LibUIPC Auto-Install Script for Linux/macOS

set -e  # Exit on error

echo "🚀 LibUIPC Auto-Installer for Linux/macOS"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not found"
    exit 1
fi

# Run the auto installer
echo "Starting automatic installation..."
python3 auto_install.py "$@"

echo ""
echo "Installation completed!"
echo ""
echo "To test the installation:"
echo "  python3 -c 'import uipc; print(\"✅ LibUIPC imported successfully!\")'"