#!/bin/bash
#
# Build script for M4 network simulator backends
#
# Usage:
#   ./build.sh           # Build both NS3 and FlowSim
#   ./build.sh ns3       # Build NS3 only
#   ./build.sh flowsim   # Build FlowSim only
#

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BACKENDS_DIR="$SCRIPT_DIR/backends"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

function print_header() {
    echo ""
    echo "================================================================================"
    echo "$1"
    echo "================================================================================"
    echo ""
}

function print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

function print_error() {
    echo -e "${RED}❌ $1${NC}"
}

function print_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

function build_ns3() {
    print_header "Building NS3 (UNISON)"
    
    cd "$BACKENDS_DIR/UNISON"
    
    print_info "Cleaning previous build..."
    ./ns3 clean
    
    print_info "Configuring NS3 with optimized profile..."
    ./ns3 configure --build-profile=optimized --enable-mtp
    
    print_info "Building NS3..."
    ./ns3 build
    
    # Check if binary exists
    if [ -f "build/scratch/ns3.39-twelve-optimized" ]; then
        print_success "NS3 built successfully!"
        print_info "Binary: build/scratch/ns3.39-twelve-optimized"
    else
        print_error "NS3 build failed - binary not found"
        return 1
    fi
}

function build_flowsim() {
    print_header "Building FlowSim"
    
    cd "$BACKENDS_DIR/flowsim"
    
    print_info "Cleaning previous build..."
    make clean
    
    print_info "Building FlowSim..."
    make -j$(nproc)
    
    # Check if binary exists
    if [ -f "main" ]; then
        print_success "FlowSim built successfully!"
        print_info "Binary: main"
    else
        print_error "FlowSim build failed - binary not found"
        return 1
    fi
}

# Main script
case "${1:-all}" in
    ns3)
        build_ns3
        ;;
    flowsim)
        build_flowsim
        ;;
    all|"")
        build_ns3
        build_flowsim
        ;;
    *)
        echo "Usage: $0 [ns3|flowsim|all]"
        echo ""
        echo "Examples:"
        echo "  $0           # Build both"
        echo "  $0 ns3       # Build NS3 only"
        echo "  $0 flowsim   # Build FlowSim only"
        exit 1
        ;;
esac

print_header "Build Complete!"
print_info "Run sweeps with: python run.py [ns3|flowsim|all] --jobs 32"

