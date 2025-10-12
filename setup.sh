#!/bin/bash

# This script helps set up the development environment for the NeuralNetwork project.
# It detects the operating system and automatically installs necessary dependencies.

set -e  # Exit on error
set -u  # Exit on undefined variable

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Download file with progress
download_file() {
    local url="$1"
    local output="$2"
    log_info "Downloading $output..."
    if command_exists wget; then
        wget -q --show-progress -O "$output" "$url"
    else
        curl -L --progress-bar -o "$output" "$url"
    fi
}

# Install Python packages
install_python_packages() {
    local python_cmd="$1"
    log_info "Installing Python packages (matplotlib, numpy)..."
    "$python_cmd" -m pip install --upgrade pip --quiet 2>/dev/null || true
    
    # Try without --break-system-packages first, then with it if needed
    if "$python_cmd" -m pip install matplotlib numpy --quiet 2>/dev/null; then
        log_info "Python packages installed successfully."
    elif "$python_cmd" -m pip install matplotlib numpy --quiet --break-system-packages 2>/dev/null; then
        log_info "Python packages installed successfully (using --break-system-packages)."
    else
        log_error "Failed to install Python packages."
        log_warn "You may need to install them manually:"
        log_warn "  $python_cmd -m pip install matplotlib numpy --break-system-packages"
        log_warn "Or use a virtual environment or system package manager."
    fi
}

# Linux setup
setup_linux() {
    log_info "Detected Linux."
    
    if [ ! -f /etc/os-release ]; then
        log_error "Could not detect Linux distribution."
        log_warn "Please install: build-essential, cmake, libcurl-devel, python3-devel manually."
        exit 1
    fi
    
    . /etc/os-release
    local os_name="$NAME"
    
    case "$os_name" in
        "Ubuntu"|"Debian GNU/Linux"|"Kali GNU/Linux")
            log_info "Detected Debian-based system: $os_name"
            sudo apt-get update -qq
            sudo apt-get install -y -qq build-essential cmake libcurl4-openssl-dev python3-dev python3-pip
            ;;
        "Fedora"|"CentOS Linux"|"Red Hat Enterprise Linux")
            log_info "Detected Red Hat-based system: $os_name"
            sudo dnf groupinstall -y -q 'Development Tools'
            sudo dnf install -y -q cmake libcurl-devel python3-devel python3-pip
            ;;
        "Arch Linux")
            log_info "Detected Arch Linux"
            sudo pacman -Syu --noconfirm --quiet base-devel cmake curl python python-pip
            ;;
        *)
            log_error "Unsupported Linux distribution: $os_name"
            log_warn "Please install: build-essential, cmake, libcurl-devel, python3-devel manually."
            exit 1
            ;;
    esac
    
    # Install Python packages
    if command_exists python3; then
        install_python_packages python3
    fi
    
    log_info "Linux setup complete!"
}

# macOS setup
setup_macos() {
    log_info "Detected macOS."
    
    if ! command_exists brew; then
        log_error "Homebrew not found."
        echo "Please install Homebrew first: https://brew.sh/"
        exit 1
    fi
    
    log_info "Installing dependencies via Homebrew..."
    brew install cmake curl python
    
    # Install Python packages
    if command_exists python3; then
        install_python_packages python3
    fi
    
    log_info "macOS setup complete!"
}

# Windows setup
setup_windows() {
    log_info "Detected Windows (Git Bash/MSYS/Cygwin)."
    
    local needs_restart=false
    
    # Install Python
    if command_exists python || command_exists python3; then
        log_info "Python is already installed."
        local python_cmd=$(command -v python || command -v python3)
        install_python_packages "$python_cmd"
    else
        log_info "Installing Python 3.11.7..."
        local python_version="3.11.7"
        local python_installer="python-${python_version}-amd64.exe"
        local python_url="https://www.python.org/ftp/python/${python_version}/${python_installer}"
        
        download_file "$python_url" "$python_installer"
        
        log_info "Running Python installer (silent)..."
        ./"$python_installer" /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
        rm "$python_installer"
        
        needs_restart=true
        log_info "Python installed."
    fi
    
    # Install CMake
    if command_exists cmake; then
        log_info "CMake is already installed: $(cmake --version | head -n1)"
    else
        log_info "Installing CMake 3.28.1..."
        local cmake_version="3.28.1"
        local cmake_installer="cmake-${cmake_version}-windows-x86_64.msi"
        local cmake_url="https://github.com/Kitware/CMake/releases/download/v${cmake_version}/${cmake_installer}"
        
        download_file "$cmake_url" "$cmake_installer"
        
        log_info "Running CMake installer (silent)..."
        msiexec //i "$cmake_installer" //quiet ADD_CMAKE_TO_PATH=System
        rm "$cmake_installer"
        
        needs_restart=true
        log_info "CMake installed."
    fi
    
    # Check for Visual Studio
    local vs_paths=(
        "/c/Program Files/Microsoft Visual Studio/2022/Community/VC/Auxiliary/Build/vcvars64.bat"
        "/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvars64.bat"
        "/c/Program Files/Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvars64.bat"
    )
    
    local vs_found=false
    for path in "${vs_paths[@]}"; do
        if [ -f "$path" ]; then
            vs_found=true
            break
        fi
    done
    
    if [ "$vs_found" = true ] || command_exists cl.exe 2>/dev/null; then
        log_info "Visual Studio C++ compiler is already installed."
    else
        log_info "Installing Visual Studio Build Tools 2022..."
        log_warn "This may take 10-15 minutes depending on your internet connection."
        
        local vs_installer="vs_buildtools.exe"
        local vs_url="https://aka.ms/vs/17/release/vs_buildtools.exe"
        
        download_file "$vs_url" "$vs_installer"
        
        log_info "Running Visual Studio Build Tools installer..."
        ./"$vs_installer" --quiet --wait --norestart --nocache \
            --installPath "C:\\BuildTools" \
            --add Microsoft.VisualStudio.Workload.VCTools \
            --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 \
            --add Microsoft.VisualStudio.Component.Windows11SDK.22621 \
            --includeRecommended
        
        rm "$vs_installer"
        needs_restart=true
        log_info "Visual Studio Build Tools installed."
    fi
    
    # Setup vcpkg
    if [ -d "vcpkg" ]; then
        log_info "vcpkg directory already exists."
    else
        log_info "Cloning vcpkg..."
        git clone --depth 1 -q https://github.com/microsoft/vcpkg.git
    fi
    
    if [ -f "vcpkg/vcpkg.exe" ]; then
        log_info "vcpkg already bootstrapped."
    else
        log_info "Bootstrapping vcpkg..."
        cd vcpkg && ./bootstrap-vcpkg.sh -disableMetrics > /dev/null && cd ..
    fi
    
    log_info "Installing curl via vcpkg..."
    ./vcpkg/vcpkg install curl:x64-windows --clean-after-build > /dev/null 2>&1
    
    log_info "Windows setup complete!"
    
    if [ "$needs_restart" = true ]; then
        log_warn "Please restart your terminal for PATH changes to take effect."
    fi
    
    echo ""
    echo "To build the project, run:"
    echo 'cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE="vcpkg/scripts/buildsystems/vcpkg.cmake" -DNN_DISABLE_EIGEN_PARALLELISM=OFF -DNN_ENABLE_FAST_MATH=ON -DNN_EPOCHS=10 -DNN_PRINT_EVERY=100 -DBATCH_SIZE=128'
    echo "cmake --build build --config Release --parallel"
}

# Main script
main() {
    echo "================================================"
    echo "  NeuralNetwork Project - Setup Script"
    echo "================================================"
    echo ""
    
    read -p "This script will install necessary system packages. Continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_warn "Installation cancelled."
        exit 0
    fi
    
    log_info "Detecting operating system..."
    
    case "$OSTYPE" in
        linux-gnu*)
            setup_linux
            ;;
        darwin*)
            setup_macos
            ;;
        msys|cygwin|win32)
            setup_windows
            exit 0
            ;;
        *)
            log_error "Unsupported operating system: $OSTYPE"
            log_warn "Please install: C++ compiler, CMake, Curl, Python 3 manually."
            exit 1
            ;;
    esac
    
    # Print build instructions (Linux/macOS)
    local num_cores=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    echo ""
    echo "To build the project, run:"
    echo "cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNN_DISABLE_EIGEN_PARALLELISM=OFF -DNN_ENABLE_FAST_MATH=ON -DNN_EPOCHS=10 -DNN_PRINT_EVERY=100 -DBATCH_SIZE=128"
    echo "cmake --build build --config Release -- -j${num_cores}"
}

main "$@"