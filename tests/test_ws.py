import asyncio
import json
import websockets


#URL = "wss://helviz-gsm-voice.hf.space/twilio/voice"
URL = "wss://helviz-gsm-voice.hf.space/twilio/voice"

async def test_connection():
    print(f"Connecting to {URL}...")
    try:
        async with websockets.connect(URL) as websocket:
            print("✅ Connection successfully established!")

            # Simulate Twilio's initial "connected" or "start" event
            mock_start_event = {
                "event": "connected",
                "protocol": "Call",
                "version": "1.0.0"
            }

            print("Sending mock handshake payload...")
            await websocket.send(json.dumps(mock_start_event))

            # Wait to see if your backend sends a response or closes the connection
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                print(f"Received response from backend: {response}")
            except asyncio.TimeoutError:
                print("No immediate response from server, but connection remains open (normal for raw voice streams).")

    except Exception as e:
       print(f"❌ Connection failed: {e}")

asyncio.run(test_connection())