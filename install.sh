#!/bin/bash

# TACAS Artifact Installation Script
# Template for installing various tools and dependencies

set -e  # Exit on error

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install system dependencies
install_dependencies() {
    cd "$SCRIPT_DIR"
    print_info "Installing system dependencies..."

    sudo apt-get update
    # install boost for ranker
    sudo apt-get install libboost-all-dev
    # install python3-venv for virtual environment
    sudo apt-get install python3-venv

    print_info "Creating Python virtual environment..."
    python3 -m venv .venv
    
    print_info "Activating virtual environment..."
    source .venv/bin/activate
    
    print_info "Installing Python packages from requirements.txt..."
    pip install --upgrade pip
    pip install -r ba-compl-eval/requirements.txt
    
    print_info "Dependencies installation completed"
}

# Function to install Spot
install_spot() {
    cd "$SCRIPT_DIR"
    print_info "Installing Spot..."
    
    cd bin
    tar -xvf spot-2.14.2.tar.gz
    cd spot-2.14.2
    ./configure --enable-max-accsets=128
    make
    sudo make install
    sudo ldconfig /usr/local/lib
    
    print_info "Spot installation completed"
}

# Function to install Ranker
install_ranker() {
    cd "$SCRIPT_DIR"
    print_info "Installing Ranker..."
    
    cp bin/ranker.zip ba-compl-eval/bin/
    cd ba-compl-eval/bin/
    unzip ranker.zip
    cd ranker
    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release ../src/
    make
    
    print_info "Ranker installation completed"
}

# Function to install Kofola
install_kofola() {
    cd "$SCRIPT_DIR"
    print_info "Installing Kofola..."

    cp bin/kofola.zip ba-compl-eval/bin/
    cd ba-compl-eval/bin/
    unzip kofola.zip
    cd kofola
    mkdir build && cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make

    print_info "Kofola installation completed"
}

# Function to install Kofola TACAS23 version
install_kofola_tacas23() {
    cd "$SCRIPT_DIR"
    print_info "Installing Kofola TACAS23..."

    cp bin/kofola-tacas23.tar.gz ba-compl-eval/bin/
    cd ba-compl-eval/bin/
    tar -xvf kofola-tacas23.tar.gz
    cd kofola-tacas23
    ./configure
    make
    
    print_info "Kofola TACAS23 installation completed"
}

# Function to install Rabit
install_rabit() {
    cd "$SCRIPT_DIR"
    print_info "Installing Rabit..."
    
    mkdir -p ba-compl-eval/bin/rabit
    cp bin/RABIT.jar ba-compl-eval/bin/rabit/
    
    print_info "Rabit installation completed"
}

# Function to install Forklift
install_forklift() {
    cd "$SCRIPT_DIR"
    print_info "Installing Forklift..."

    cp bin/FORKLIFT.zip ba-compl-eval/bin/
    cd ba-compl-eval/bin/
    unzip FORKLIFT.zip
    cd FORKLIFT
    make
    
    print_info "Forklift installation completed"
}

# Function to install Bait
install_bait() {
    cd "$SCRIPT_DIR"
    print_info "Installing Bait..."
    
    mkdir -p ba-compl-eval/bin/BAIT
    cp bin/bait.jar ba-compl-eval/bin/BAIT/
    
    print_info "Bait installation completed"
}

# Main installation function
main() {
    print_info "Starting TACAS Artifact installation..."
    print_info "This script will install the following tools:"
    print_info "  - System dependencies"
    print_info "  - Spot"
    print_info "  - Ranker"
    print_info "  - Kofola"
    print_info "  - Kofola TACAS23"
    print_info "  - Rabit"
    print_info "  - Forklift"
    print_info "  - Bait"
    echo
    
    # Install dependencies first
    install_dependencies
    
    # Install tools
    install_spot
    install_ranker
    install_kofola
    install_kofola_tacas23
    install_rabit
    install_forklift
    install_bait
    
    print_info "All installations completed successfully!"
}

# Check if script is run with sudo when needed
check_permissions() {
    # This function can be used to check if certain operations need elevated privileges
    # Uncomment and modify as needed
    # if [[ $EUID -eq 0 ]]; then
    #     print_warning "This script is running as root"
    # fi
    return 0
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Parse command line arguments if needed
    case "${1:-}" in
        --help|-h)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --help, -h    Show this help message"
            echo "  --deps-only   Install only dependencies"
            exit 0
            ;;
        --deps-only)
            install_dependencies
            exit 0
            ;;
        "")
            # No arguments, run main installation
            check_permissions
            main
            ;;
        *)
            print_error "Unknown option: $1"
            print_info "Use --help for usage information"
            exit 1
            ;;
    esac
fi
