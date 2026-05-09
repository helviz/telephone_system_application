import asyncio
import os
from llmModule.GGUFStrategy import GGUFStrategy


async def test_local_llm():
    # 1. Configuration
    # Ensure this path matches where you saved your Qwen/Llama model file
    MODEL_PATH = "/home/twomoelvis/PycharmProjects/jayden-telynx-voice-agent/models/TinyV-Qwen3-1.7B-Think.i1-IQ1_M.gguf"
    SYSTEM_PROMPT = "You are a helpful assistant. Keep your answers brief."

    # Check if the model file actually exists before starting
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at {MODEL_PATH}")
        print("Please download a GGUF model from Hugging Face and place it in the /models folder.")
        return

    print(f"--- Initializing Local Model: {os.path.basename(MODEL_PATH)} ---")

    # 2. Initialize the Strategy
    try:
        # This may take a few seconds as it loads the model into RAM
        local_model = GGUFStrategy(model_path=MODEL_PATH, system_prompt=SYSTEM_PROMPT)
    except Exception as e:
        print(f"Failed to load the model: {e}")
        return

    # 3. Test Input
    user_query = "How do I check my current IP address on Arch Linux?"
    print(f"\nUser: {user_query}")

    # 4. Execute the Stream
    try:
        # Since GGUFStrategy.generate_stream prints to terminal internally,
        # we just need to iterate through it to keep it running.
        async for chunk in local_model.generate_stream(user_query):
            # We use asyncio.sleep(0) to allow the event loop to switch
            # tasks if needed (e.g., if we were running a mic or UI)
            await asyncio.sleep(0)
    except Exception as e:
        print(f"\n[!] Inference Error: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(test_local_llm())
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")