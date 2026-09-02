#!/usr/bin/env python
"""Check which Groq models are actually available for this API key"""
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("ERROR: No GROQ_API_KEY found in .env")
    exit(1)

try:
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    models = client.models.list()
    
    print("\n=== Available Groq Models ===")
    for model in models.data:
        print(f"  - {model.id}")
        if hasattr(model, 'context_window'):
            print(f"    Context: {model.context_window}")
    
    print("\n=== Recommended for your use: ===")
    # Filter for text models suitable for reasoning
    text_models = [m for m in models.data if 'llama' in m.id.lower() and 'vision' not in m.id.lower()]
    if text_models:
        print(f"  Use: {text_models[0].id}")
    else:
        print("  Use the first model from the list above")
        
except Exception as e:
    print(f"ERROR: {e}")
