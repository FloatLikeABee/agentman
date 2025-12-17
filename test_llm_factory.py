"""
Test script for LLM Factory
Demonstrates how to use different LLM providers
"""
from src.llm_factory import LLMFactory, LLMProvider

# Import all callers to register them with the factory
import gemini_caller
import qwen_caller
import glm_caller


def test_gemini():
    """Test Gemini caller"""
    print("\n=== Testing Gemini ===")
    caller = LLMFactory.create_caller(
        provider=LLMProvider.GEMINI,
        api_key="AIzaSyAt19tBj232GyyUbM95MlZzZarqZcTKmsc",
        model="gemini-2.5-flash"
    )
    
    response = caller.generate("What is artificial intelligence? Answer in one sentence.")
    print(f"Response: {response}")


def test_qwen():
    """Test Qwen caller"""
    print("\n=== Testing Qwen ===")
    caller = LLMFactory.create_caller(
        provider=LLMProvider.QWEN,
        api_key="sk-fc88e8c463e94a43bc41f1094a28fa1f",
        model="qwen3-max"
    )
    
    response = caller.generate("What is artificial intelligence? Answer in one sentence.")
    print(f"Response: {response}")


def test_glm():
    """Test GLM caller"""
    print("\n=== Testing GLM ===")
    caller = LLMFactory.create_caller(
        provider=LLMProvider.GLM,
        api_key="0aa9ad7dd2114ef3bfb16bea056a977a.9nVmCTUa8l6SBCK6",
        model="glm-4.6"
    )
    
    response = caller.generate("What is artificial intelligence? Answer in one sentence.")
    print(f"Response: {response}")


def test_chat():
    """Test chat functionality"""
    print("\n=== Testing Chat with Gemini ===")
    caller = LLMFactory.create_caller(
        provider=LLMProvider.GEMINI,
        api_key="AIzaSyAt19tBj232GyyUbM95MlZzZarqZcTKmsc",
        model="gemini-2.5-flash"
    )
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
    ]
    
    response = caller.chat(messages)
    print(f"Response: {response}")


def test_streaming():
    """Test streaming functionality"""
    print("\n=== Testing Streaming with Qwen ===")
    caller = LLMFactory.create_caller(
        provider=LLMProvider.QWEN,
        api_key="sk-fc88e8c463e94a43bc41f1094a28fa1f",
        model="qwen3-max"
    )
    
    print("Streaming response: ", end="", flush=True)
    for chunk in caller.stream("Tell me a short joke."):
        print(chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    print("LLM Factory Test Suite")
    print("=" * 50)
    
    # List available providers
    providers = LLMFactory.get_available_providers()
    print(f"\nAvailable providers: {providers}")
    
    # Test each provider
    try:
        test_gemini()
    except Exception as e:
        print(f"Gemini test failed: {e}")
    
    try:
        test_qwen()
    except Exception as e:
        print(f"Qwen test failed: {e}")
    
    try:
        test_glm()
    except Exception as e:
        print(f"GLM test failed: {e}")
    
    try:
        test_chat()
    except Exception as e:
        print(f"Chat test failed: {e}")
    
    try:
        test_streaming()
    except Exception as e:
        print(f"Streaming test failed: {e}")
    
    print("\n" + "=" * 50)
    print("Test suite completed!")

