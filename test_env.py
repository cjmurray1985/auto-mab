import os
from dotenv import load_dotenv

load_dotenv()

firecrawl_key = os.getenv("FIRECRAWL_API_KEY")
anthropic_key = os.getenv("ANTHROPIC_API_KEY")

print(f"Firecrawl API key loaded: {'Yes' if firecrawl_key else 'No'}")
print(f"Anthropic API key loaded: {'Yes' if anthropic_key else 'No'}")
