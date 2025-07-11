import os
import json
import pandas as pd
import random
from datetime import datetime
from dotenv import load_dotenv
from anthropic import Anthropic
import requests

# Load environment variables
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")

# Initialize Anthropic client
client = Anthropic(api_key=ANTHROPIC_API_KEY)

def extract_article_metadata(url):
    """Extract article metadata using Firecrawl API"""
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
            # Parse the HTML to extract metadata
            data = response.json()
            
            # This is a simplified extraction process
            # In a full implementation, you'd use BeautifulSoup or similar
            # to properly parse the HTML and extract metadata
            html_content = data.get('html', '')
            
            # Simple extraction example (would need refinement)
            title = extract_title_from_html(html_content)
            description = extract_description_from_html(html_content)
            category = extract_category_from_html(html_content)
            
            return {
                "original_title": title,
                "description": description,
                "category": category,
                "url": url
            }
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Exception during metadata extraction: {e}")
        return None

def extract_title_from_html(html):
    """Extract title from HTML (simplified example)"""
    # In a real implementation, use BeautifulSoup or similar
    # This is just a simplified example
    if "<title>" in html and "</title>" in html:
        start = html.find("<title>") + len("<title>")
        end = html.find("</title>")
        return html[start:end].strip()
    return "Unknown Title"

def extract_description_from_html(html):
    """Extract description from HTML (simplified example)"""
    # Look for meta description
    if 'name="description"' in html:
        start = html.find('name="description"')
        content_start = html.find('content="', start) + len('content="')
        content_end = html.find('"', content_start)
        if content_start > 0 and content_end > 0:
            return html[content_start:content_end].strip()
    return "No description available"

def extract_category_from_html(html):
    """Extract category from HTML (simplified example)"""
    # This would need customization based on the site structure
    # A real implementation might look for category in URL, breadcrumbs, etc.
    return "General"

def generate_headline_variants(article_metadata):
    """Generate headline variants using Claude 3.7 Sonnet"""
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
        
        # Simple parsing for the numbered list (would need refinement)
        variants = []
        for line in variant_text.strip().split('\n'):
            if line.strip() and any(line.strip().startswith(str(i) + '.') for i in range(1, 10)):
                variant = line.strip().split('.', 1)[1].strip()
                if variant:
                    variants.append(variant)
        
        # Ensure we have exactly 3 variants
        while len(variants) < 3:
            variants.append(None)
        
        return variants[:3]
    except Exception as e:
        print(f"Exception during headline generation: {e}")
        return [None, None, None]

def prepare_for_review(url, original_title, variants):
    """Prepare data for editorial review"""
    # Combine original and generated variants
    all_titles = [original_title] + [v for v in variants if v]
    
    # Shuffle the titles for blind review
    shuffled_titles = random.sample(all_titles, len(all_titles))
    
    # Pad to ensure we have 4 slots (original + 3 variants)
    while len(shuffled_titles) < 4:
        shuffled_titles.append(None)
    
    return {
        'article_url': url,
        'variant_1': shuffled_titles[0],
        'variant_2': shuffled_titles[1],
        'variant_3': shuffled_titles[2],
        'variant_4': shuffled_titles[3]
    }

def main():
    # For POC, use a small set of example URLs
    urls = [
        "https://news.yahoo.com/",
        # Add more example URLs here
    ]
    
    results = []
    
    for url in urls:
        print(f"Processing {url}...")
        
        # Extract metadata
        metadata = extract_article_metadata(url)
        if not metadata:
            print(f"Skipping {url} due to metadata extraction failure")
            continue
        
        # Generate headline variants
        variants = generate_headline_variants(metadata)
        
        # Prepare for review
        review_data = prepare_for_review(
            url, 
            metadata['original_title'], 
            variants
        )
        
        results.append(review_data)
        
        print(f"Generated variants for {url}")
        
    # Export results to CSV
    if results:
        df = pd.DataFrame(results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"headline_variants_{timestamp}.csv"
        df.to_csv(output_file, index=False)
        print(f"Exported results to {output_file}")
    else:
        print("No results to export")

if __name__ == "__main__":
    main()
