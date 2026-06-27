#!/bin/bash

set -euo pipefail

echo "🚀 Starting Voice Assistant..."

# Use existing PORT if Hugging Face provides one, otherwise default to 7860
export PORT="${PORT:-7860}"

echo ""
echo "============================================================"
echo "   ENVIRONMENT DEBUG SNAPSHOT"
echo "============================================================"
echo "Timestamp: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "Python:    $(python3 --version 2>/dev/null || true)"
echo "Working directory: $(pwd)"
echo "PORT: ${PORT}"
echo "============================================================"
echo ""

echo "🔧 App-related environment variables:"
echo "------------------------------------------------------------"

env | sort | grep -E \
  '^(PORT|SAFETY_|TRANSFER_|OPERATOR_|TWILIO_|TELNYX_|STT_|WEBRTC_|WHISPER_|OPENAI_WHISPER_|SONIOX_|TTS_|LLM_|GGUF_|N_CTX|N_GPU_LAYERS|QWEN_|HF_HOME|TRANSFORMERS_CACHE|CUDA_|PYTORCH_)=' \
  | awk -F= '
    BEGIN { OFS="=" }
    $1 ~ /(KEY|TOKEN|SECRET|PASSWORD|PASS|AUTH|CREDENTIAL|COOKIE)/ {
      print $1, "***REDACTED***"
      next
    }
    {
      print $0
    }
  '

echo ""
echo "🌍 Full environment snapshot with secrets redacted:"
echo "------------------------------------------------------------"

env | sort | awk -F= '
  BEGIN { OFS="=" }
  $1 ~ /(KEY|TOKEN|SECRET|PASSWORD|PASS|AUTH|CREDENTIAL|COOKIE)/ {
    print $1, "***REDACTED***"
    next
  }
  {
    print $0
  }
'

echo ""
echo "============================================================"
echo "   STARTING APPLICATION"
echo "============================================================"
echo ""

exec python3 sockets.py