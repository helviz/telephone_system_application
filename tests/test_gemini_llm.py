import asyncio
import os
from llmModule.LLM import LLM


async def run_test(provider="gemini", lang="en"):
    """
    Tests the LLM Factory and the resulting strategy's streaming capability.
    """
    print(f"\n" + "=" * 50)
    print(f"TESTING PROVIDER: {provider.upper()} | LANG: {lang.upper()}")
    print("=" * 50)

    # 1. Initialize through the Factory
    # This automatically handles .env loading and system prompt selection
    model = LLM.get_model(provider=provider, lang=lang)

    if model is None:
        print(f"Error: Could not initialize provider '{provider}'. Check your .env or model path.")
        return

    # 2. Define a test query
    test_queries = {
        "en": "Briefly explain what a kernel is.",
        "fr": "Expliquez brièvement ce qu'est un noyau.",
        "sw": "Eleza kwa kifupi kiini (kernel) ni nini."
    }

    user_input = test_queries.get(lang, test_queries["en"])
    print(f"User Request: {user_input}\n")

    # 3. Consume the stream
    # Note: The Strategy itself handles the printing to terminal (flush=True)
    try:
        # We must iterate through the async generator to trigger the internal prints
        async for _ in model.generate_stream(user_input):
            # No additional print needed here because it's handled inside the Strategy
            await asyncio.sleep(0)  # Yield control to the event loop
    except Exception as e:
        print(f"\n[!] Test Failed with error: {e}")


async def main():
    # Test 1: Gemini in English
    await run_test(provider="gemini", lang="en")

    # Test 2: Gemini in Swahili
    # This verifies your SYSTEM_PROMPTS dictionary in LLM.py works
    await run_test(provider="gemini", lang="sw")

    # Optional: Test 3: Qwen (Local GGUF)
    # Ensure your .gguf file exists at the path defined in LLM.py before enabling
    # await run_test(provider="qwen", lang="en")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTest cancelled by user.")