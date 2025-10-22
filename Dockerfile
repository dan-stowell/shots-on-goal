FROM ubuntu:22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install build essentials and common tools
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    python3 \
    python3-pip \
    wget \
    unzip \
    ca-certificates \
    ripgrep \
    && rm -rf /var/lib/apt/lists/*

# Install Bazelisk (manages Bazel versions automatically)
# Using ARM64 binary since we're on Darwin/Apple Silicon
RUN curl -Lo /usr/local/bin/bazel \
    https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-arm64 \
    && chmod +x /usr/local/bin/bazel

# Verify installations
RUN bazel --version && rg --version

# Create workspace directory
WORKDIR /workspace

# Set git config to avoid warnings
RUN git config --global user.email "shots-on-goal@example.com" \
    && git config --global user.name "Shots on Goal"

# Keep container running by default
CMD ["sleep", "infinity"]
