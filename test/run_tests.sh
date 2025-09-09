#!/bin/bash
# LibUIPC Installation Test Runner

set -e

echo "🚀 LibUIPC Installation Test Suite"
echo ""

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is required but not found"
    echo "Please install Docker and try again"
    exit 1
fi

# Function to build and test in container
test_in_container() {
    local dockerfile=$1
    local test_name=$2
    local method=${3:-auto}
    
    echo "🐳 Testing in $test_name container..."
    
    # Build image
    echo "Building $test_name test image..."
    docker build -f test/$dockerfile -t libuipc-test-$test_name .
    
    # Run test
    echo "Running installation test..."
    docker run --rm \
        --gpus all \
        -v $(pwd):/workspace/source \
        libuipc-test-$test_name \
        python3 test_installation.py --method $method
    
    echo "✅ $test_name test completed"
    echo ""
}

# Parse arguments
METHOD="auto"
DISTRO="all"

while [[ $# -gt 0 ]]; do
    case $1 in
        --method)
            METHOD="$2"
            shift 2
            ;;
        --distro)
            DISTRO="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--method auto|pip] [--distro ubuntu|centos|all]"
            echo ""
            echo "Options:"
            echo "  --method    Installation method to test (auto|pip)"
            echo "  --distro    Distribution to test (ubuntu|centos|all)"
            echo "  --help      Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Test configuration:"
echo "  Method: $METHOD"
echo "  Distribution: $DISTRO"
echo ""

# Run tests
if [[ "$DISTRO" == "all" || "$DISTRO" == "ubuntu" ]]; then
    test_in_container "Dockerfile.ubuntu" "ubuntu" "$METHOD"
fi

if [[ "$DISTRO" == "all" || "$DISTRO" == "centos" ]]; then
    test_in_container "Dockerfile.centos" "centos" "$METHOD"  
fi

echo "🎉 All tests completed successfully!"