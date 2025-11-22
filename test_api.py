import google.generativeai as genai

# PASTE YOUR KEY HERE
api_key = "AIzaSyDcxKuacRKHNlgcx6R8stpi6Llr3bJFvy8"

try:
    genai.configure(api_key=api_key)
    
    # List of potential model names to test
    candidates = ["gemini-1.5-flash", "gemini-pro", "gemini-1.5-pro-latest"]
    
    print("Testing connection...")
    
    for model_name in candidates:
        print(f"Trying model: {model_name}...")
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content("Hello, are you working?")
            print(f"✅ SUCCESS! The correct model name is: '{model_name}'")
            print(f"Response: {response.text}")
            break # Stop after finding one that works
        except Exception as e:
            print(f"❌ Failed: {e}")

except Exception as e:
    print(f"Critical Error: {e}")