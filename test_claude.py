import os
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")

if not api_key:
    print("ERROR: Anthropic API key not found in .env file")
    exit(1)

print("Initializing Anthropic client...")
client = Anthropic(api_key=api_key)

print("Sending test request to Claude 3.7 Sonnet...")
try:
    response = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=300,
        system="You are an expert headline writer for a news aggregator.",
        messages=[
            {"role": "user", "content": "Generate 3 alternative headline variants for this article title: 'Study finds coffee may boost productivity'"}
        ]
    )
    
    print("\nResponse from Claude:")
    print(response.content[0].text)
    print("\nAPI call successful!")
    
except Exception as e:
    print(f"Exception occurred: {e}")
