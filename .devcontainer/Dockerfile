# Base image
FROM python:3.10-slim

# Install required packages for poetry, sudo and non-root user creation
RUN apt-get update && apt-get install -y \
    curl \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user 'vscode' and add to sudoers group
RUN useradd -m vscode && echo "vscode ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/nopasswd && chmod 0440 /etc/sudoers.d/nopasswd

# Set default shell of vscode user to bash
RUN chsh -s /bin/bash vscode

# Switch to vscode user
USER vscode

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Add poetry to the PATH (for all shells)
ENV PATH="/home/vscode/.local/bin:$PATH"

# Set the default working directory
WORKDIR /workspace

# Set default user to vscode
USER vscode