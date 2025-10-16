#!/bin/bash
set -e

# Script to build the shots-on-goal container image

IMAGE_NAME="shots-on-goal"
IMAGE_TAG="latest"

# Detect container runtime
if command -v container &> /dev/null; then
    RUNTIME="container"
    echo "Using 'container' runtime"
elif command -v docker &> /dev/null; then
    RUNTIME="docker"
    echo "Using 'docker' runtime"
else
    echo "Error: Neither 'container' nor 'docker' command found"
    exit 1
fi

# Build the image
echo "Building ${IMAGE_NAME}:${IMAGE_TAG}..."
$RUNTIME build -t ${IMAGE_NAME}:${IMAGE_TAG} .

# Verify the build
echo ""
echo "Build complete! Verifying..."
$RUNTIME run --rm ${IMAGE_NAME}:${IMAGE_TAG} bazel --version
$RUNTIME run --rm ${IMAGE_NAME}:${IMAGE_TAG} rg --version

echo ""
echo "Successfully built ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""
echo "To use this image with shots_on_goal.py, specify it with --image flag:"
echo "  python3 shots_on_goal.py --image ${IMAGE_NAME}:${IMAGE_TAG} \"goal\" /path/to/repo"
