#!/bin/bash

set -e

echo "🚀 Starting Voice Assistant..."
export PORT=7860

# Warm all models into memory before accepting any calls
python3 preload.py

# Start your app
python3 sockets.py