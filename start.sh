#!/bin/bash

set -euo pipefail

echo "🚀 Starting Voice Assistant..."
export PORT=7860

exec python3 sockets.py