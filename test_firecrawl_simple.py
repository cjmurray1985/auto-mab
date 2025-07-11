import os
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("FIRECRAWL_API_KEY")

if not api_key:
    print("ERROR: Firecrawl API key not found in .env file")
    exit(1)

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

payload = {
    "url": "https://news.yahoo.com/",
    "formats": ["html"]
}

print("Sending request to Firecrawl API...")
try:
    response = requests.post(
        "https://api.firecrawl.dev/v1/scrape",
        headers=headers,
        json=payload
    )
    
    print(f"Response status code: {response.status_code}")
    if response.status_code == 200:
        print("Connection successful!")
        print(f"Response contains {len(response.text)} characters")
    else:
        print(f"Error response: {response.text}")
        
except Exception as e:
    print(f"Exception occurred: {e}")
