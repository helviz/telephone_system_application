#!/bin/bash

# Target the application PID and Cloudflare PID for graceful shutdown
cleanup() {
    echo "Stopping all processes..."
    kill $(jobs -p) 2>/dev/null
    exit 0
}
trap cleanup SIGINT SIGTERM

# 1. Start the FastAPI/Uvicorn application in the background
echo "🚀 Starting Python Application Server..."
python test_phone.py &
APP_PID=$!

# 2. Start Cloudflare Tunnel if a token is provided
if [ -z "$TUNNEL_TOKEN" ]; then
    echo "⚠️ TUNNEL_TOKEN environment variable is missing."
    echo "Running application in local mode without Cloudflare Tunnel."
    wait $APP_PID
else
    echo "☁️ Starting Cloudflare Tunnel..."
    cloudflared tunnel run --token "$TUNNEL_TOKEN" &
    TUNNEL_PID=$!

    # Wait for either the app or the tunnel to exit
    wait -n

    # Clean up the remaining process if one fails
    kill $APP_PID $TUNNEL_PID 2>/dev/null || true
fi