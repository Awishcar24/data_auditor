import google.generativeai as genai

# PASTE YOUR KEY HERE
api_key = "AIzaSyDcxKuacRKHNlgcx6R8stpi6Llr3bJFvy8"

try:
    genai.configure(api_key=api_key)
    
    print("Searching for available models...")
    models = genai.list_models()
    
    found_any = False
    for m in models:
        # We only want models that can generate text (generateContent)
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
            found_any = True
            
    if not found_any:
        print("No text generation models found for this API key.")
        
except Exception as e:
    print(f"Error: {e}")