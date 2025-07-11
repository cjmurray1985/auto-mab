import os
import json
import pandas as pd
import random
from datetime import datetime
from dotenv import load_dotenv
from anthropic import Anthropic
import requests
from bs4 import BeautifulSoup

# Load environment variables
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")

# Initialize Anthropic client
client = Anthropic(api_key=ANTHROPIC_API_KEY)

def extract_article_metadata(url):
    """Extract article metadata using Firecrawl API with improved parsing"""
    headers = {
        "Authorization": f"Bearer {FIRECRAWL_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "url": url,
        "formats": ["html"]
    }
    
    try:
        response = requests.post(
            "https://api.firecrawl.dev/v1/scrape",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            data = response.json()
            html_content = data.get('html', '')
            
            # Use BeautifulSoup for better HTML parsing
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract title - first try og:title, then regular title
            title = None
            og_title = soup.find('meta', property='og:title')
            if og_title and og_title.get('content'):
                title = og_title.get('content')
            else:
                title_tag = soup.find('title')
                if title_tag:
                    title = title_tag.text.strip()
            
            # Extract description - first try og:description, then meta description
            description = None
            og_desc = soup.find('meta', property='og:description')
            if og_desc and og_desc.get('content'):
                description = og_desc.get('content')
            else:
                meta_desc = soup.find('meta', attrs={"name": "description"})
                if meta_desc and meta_desc.get('content'):
                    description = meta_desc.get('content')
            
            # Try to extract a category (might need customization for different sites)
            category = "News"  # Default category
            # Look for category in URL path
            path_parts = url.split('/')
            if len(path_parts) > 3:
                possible_category = path_parts[3]
                if possible_category and possible_category not in ['index.html', 'home', '']:
                    category = possible_category.capitalize()
            
            print(f"Extracted metadata for {url}:")
            print(f"  Title: {title}")
            print(f"  Description: {description}")
            print(f"  Category: {category}")
            
            return {
                "original_title": title or "Unknown Title",
                "description": description or "No description available",
                "category": category,
                "url": url
            }
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Exception during metadata extraction: {e}")
        return None

def generate_headline_variants(article_metadata):
    """Generate headline variants using Claude 3.7 Sonnet with improved prompt"""
    system_prompt = """You are an expert headline writer for a news aggregator, 
    skilled at crafting compelling page titles that maximize click-through rates 
    while maintaining accuracy and journalistic integrity."""
    
    user_message = f"""
    Create 3 alternative headline variants for this article:
    
    Original Title: {article_metadata['original_title']}
    Category: {article_metadata['category']}
    Description: {article_metadata['description']}
    
    Guidelines:
    - Create headlines that would increase reader engagement and click-through rates
    - Maintain factual accuracy based on the original title and description
    - Keep headlines concise (under 70 characters if possible)
    - Each variant should have a different angle or approach
    - Do not use clickbait tactics that mislead readers
    - Make sure each headline is distinct and compelling
    
    Format your response as a numbered list with just the 3 variants.
    """
    
    try:
        response = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=300,
            temperature=0.7,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}]
        )
        
        # Extract the variants from Claude's response
        variant_text = response.content[0].text
        
        # Improved parsing for the numbered list
        variants = []
        for line in variant_text.strip().split('\n'):
            line = line.strip()
            if line and any(line.startswith(f"{i}.") or line.startswith(f"{i}. ") for i in range(1, 10)):
                # Split on the first period-space after the number
                parts = line.split('. ', 1)
                if len(parts) > 1:
                    variant = parts[1].strip()
                    if variant:
                        variants.append(variant)
        
        # Ensure we have exactly 3 variants
        while len(variants) < 3:
            variants.append(None)
        
        return variants[:3]
    except Exception as e:
        print(f"Exception during headline generation: {e}")
        return [None, None, None]

def prepare_for_review(url, metadata, variants):
    """Prepare data for editorial review with improved structure"""
    original_title = metadata['original_title']
    
    # Combine original and generated variants
    all_titles = [original_title] + [v for v in variants if v]
    
    # Shuffle the titles for blind review
    shuffled_titles = random.sample(all_titles, len(all_titles))
    
    # Pad to ensure we have 4 slots (original + 3 variants)
    while len(shuffled_titles) < 4:
        shuffled_titles.append(None)
    
    return {
        'article_url': url,
        'category': metadata['category'],
        'original_position': shuffled_titles.index(original_title) + 1 if original_title in shuffled_titles else 0,
        'variant_1': shuffled_titles[0],
        'variant_2': shuffled_titles[1],
        'variant_3': shuffled_titles[2],
        'variant_4': shuffled_titles[3]
    }

def main():
    # For POC, use a set of example URLs - added more real article URLs
    urls = [
        "https://news.yahoo.com/",
        "https://news.yahoo.com/putin-says-western-enemies-trying-130646694.html",
        "https://finance.yahoo.com/news/stock-market-today-dow-drops-211625825.html"
    ]
    
    results = []
    
    for url in urls:
        print(f"\nProcessing {url}...")
        
        # Extract metadata
        metadata = extract_article_metadata(url)
        if not metadata:
            print(f"Skipping {url} due to metadata extraction failure")
            continue
        
        # Generate headline variants
        print("Generating headline variants...")
        variants = generate_headline_variants(metadata)
        print("Generated variants:")
        for i, variant in enumerate(variants, 1):
            print(f"  {i}. {variant}")
        
        # Prepare for review
        review_data = prepare_for_review(
            url, 
            metadata, 
            variants
        )
        
        results.append(review_data)
        
        print(f"Completed processing for {url}")
        
    # Export results to CSV
    if results:
        df = pd.DataFrame(results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"headline_variants_{timestamp}.csv"
        df.to_csv(output_file, index=False)
        print(f"\nExported results to {output_file}")
        
        # Display the results
        print("\nGenerated headline variants for editorial review:")
        print(df.to_string())
    else:
        print("No results to export")

if __name__ == "__main__":
    main()
