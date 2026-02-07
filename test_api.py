#!/usr/bin/env python3
"""
Gemini API Key Test
"""
import sys

# Load API key
try:
    import toml
    with open('secrets.toml', 'r') as f:
        secrets = toml.load(f)
    api_key = secrets.get('GEMINI_API_KEY')
    print(f"✅ API Key loaded: {api_key[:20]}...")
except Exception as e:
    print(f"❌ Error loading API Key: {e}")
    sys.exit(1)

# Test google.genai (new)
print("\n" + "="*50)
print("TEST 1: google-genai (new library)")
print("="*50)
try:
    from google import genai
    print("✅ google.genai imported successfully")

    try:
        client = genai.Client(api_key=api_key)
        print("✅ Client created successfully")

        # Test content generation
        test_prompt = "Say 'Hello, I am working!' in one sentence."
        print(f"\n🧪 Testing with prompt: {test_prompt}")

        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=test_prompt
        )
        print(f"✅ Response received: {response.text}")

    except Exception as e:
        print(f"❌ Error using client: {type(e).__name__}: {e}")

except ImportError as e:
    print(f"❌ google.genai not installed: {e}")
except Exception as e:
    print(f"❌ Unexpected error: {type(e).__name__}: {e}")

# Test google.generativeai (old)
print("\n" + "="*50)
print("TEST 2: google-generativeai (old library)")
print("="*50)
try:
    import google.generativeai as genai_old
    print("✅ google.generativeai imported successfully")

    try:
        genai_old.configure(api_key=api_key)
        print("✅ API configured successfully")

        # Test content generation
        model = genai_old.GenerativeModel("gemini-2.0-flash-exp")
        print("✅ Model created successfully")

        test_prompt = "Say 'Hello, I am working!' in one sentence."
        print(f"\n🧪 Testing with prompt: {test_prompt}")

        response = model.generate_content(test_prompt)
        print(f"✅ Response received: {response.text}")

    except Exception as e:
        print(f"❌ Error using old library: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

except ImportError as e:
    print(f"❌ google.generativeai not installed: {e}")
except Exception as e:
    print(f"❌ Unexpected error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50)
print("TEST COMPLETED")
print("="*50)
