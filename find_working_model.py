import google.generativeai as genai

# --- PASTE YOUR KEY HERE ---
API_KEY = "AIzaSyA0050pG6FQyLLZeQfqtqoLt13QX6HMwOk" 

genai.configure(api_key=API_KEY)

# List of all possible model names to try
candidates = [
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
    "gemini-1.5-pro",
    "gemini-1.5-pro-latest",
    "gemini-pro",
    "gemini-1.0-pro",
    "gemini-1.0-pro-latest",
    "models/gemini-1.5-flash",
    "models/gemini-pro"
]

print("🔍 Hunting for a working model...")

for model_name in candidates:
    print(f"Testing: {model_name}...", end=" ")
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content("Hello")
        print("✅ SUCCESS!")
        print(f"\n🎉 USE THIS NAME IN YOUR APP:  {model_name}")
        print("-" * 40)
        break # Stop after finding the first working one
    except Exception as e:
        print("❌ Failed")

print("Done.")