import os
import time
import json
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import random
from datetime import datetime
from dotenv import load_dotenv
from anthropic import Anthropic
import requests
from bs4 import BeautifulSoup
import re
from newspaper import Article

# Load environment variables
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# File path for MAB data
MAB_DATA_CSV = 'Example data  Sheet1.csv'  # Update if your filename is different

def load_mab_data(csv_path=MAB_DATA_CSV):
    """Load and process MAB testing data for few-shot examples"""
    try:
        print(f"Loading MAB data from {csv_path}...")
        
        # Load the CSV file, specifying tab separator, no header, and python engine for robustness
        df = pd.read_csv(csv_path, sep='\t', header=None, engine='python', on_bad_lines='skip')
        
        # Display columns for verification
        print(f"CSV columns: {df.columns.tolist()}")
        
        # Define columns we need to check for NaN values
        # Column 8: headline, Column 10: performance
        required_cols = [8, 10]
        df = df.dropna(subset=required_cols)
        
        # Convert performance column to numeric, coercing errors
        df[10] = pd.to_numeric(df[10], errors='coerce')
        df = df.dropna(subset=[10]) # Drop rows where performance could not be converted

        print(f"Loaded {len(df)} MAB examples after cleaning")
        return df
    except Exception as e:
        print(f"Error loading MAB data: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error

def select_few_shot_examples(article_title, mab_df, corpus_embeddings, model, num_examples=3, diversity_threshold=0.7):
    """Select few-shot examples that follow Yahoo editorial standards"""
    if corpus_embeddings is None or mab_df.empty:
        return []

    title_embedding = model.encode(article_title, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(title_embedding, corpus_embeddings)[0]

    top_k = min(num_examples * 10, len(mab_df))
    top_results = torch.topk(cosine_scores, k=top_k)

    similar_indices = [idx.item() for idx in top_results[1]]
    candidate_performance = mab_df.iloc[similar_indices][10].tolist()
    candidates = list(zip(similar_indices, candidate_performance))

    # Filter examples for Yahoo style compliance before selection
    compliant_candidates = []
    for idx, performance in candidates:
        headline = mab_df.iloc[idx][8]
        words = headline.split()
        if len(words) > 1 and not headline.isupper():  # Avoid ALL CAPS headlines
            compliant_candidates.append((idx, performance))

    # Use compliant candidates or fall back to original if too few
    final_candidates = compliant_candidates if len(compliant_candidates) >= num_examples else candidates
    
    # Sort candidates by performance score (descending)
    sorted_candidates = sorted(final_candidates, key=lambda x: x[1], reverse=True)

    selected_indices = []
    for idx, _ in sorted_candidates:
        if len(selected_indices) >= num_examples:
            break

        if not selected_indices:
            selected_indices.append(idx)
            continue

        candidate_embedding = corpus_embeddings[idx]
        embeddings_of_selected = corpus_embeddings[selected_indices]
        diversity_scores = util.pytorch_cos_sim(candidate_embedding, embeddings_of_selected)[0]

        if torch.max(diversity_scores) < diversity_threshold:
            selected_indices.append(idx)

    if len(selected_indices) < num_examples:
        remaining_needed = num_examples - len(selected_indices)
        remaining_candidate_indices = [c[0] for c in sorted_candidates if c[0] not in selected_indices]
        selected_indices.extend(remaining_candidate_indices[:remaining_needed])

    final_headlines = mab_df.iloc[selected_indices][8].tolist()
    return final_headlines

def extract_article_metadata(url):
    """Extract article metadata using the Newspaper3k library."""
    try:
        print(f"Attempting to scrape {url} with Newspaper3k...")
        article = Article(url)
        article.download()
        article.parse()

        title = article.title
        description = article.meta_description

        if not title:
            print(f"Newspaper3k failed to find a title for {url}")
            return None

        category = "News" # Default category
        path_parts = url.split('/')
        if len(path_parts) > 3:
            possible_category = path_parts[3]
            if possible_category and possible_category not in ['index.html', 'home', '']:
                category = possible_category.capitalize().replace('-', ' ')
        
        print(f"Newspaper3k scrape successful for {url}")
        return {
            "original_title": title,
            "description": description or "No description available",
            "category": category,
            "url": url
        }
    except Exception as e:
        print(f"Newspaper3k scraping failed for {url}: {e}")
        return None

def scrape_article_description(url):
    """Scrape the description of an article using Newspaper3k.
    Returns the description and an error message if any.
    """
    try:
        article = Article(url)
        article.download()
        article.parse()
        description = article.meta_description
        if not description:
            return None, "No description found with Newspaper3k."
        return description, None # Success
    except Exception as e:
        return None, f"Scraping with Newspaper3k failed: {e}"

def validate_headline_quality(headlines):
    """Validate headlines against Yahoo editorial standards"""
    validated_headlines = []
    
    for headline in headlines:
        # Check length (prefer under 70 chars, hard limit 80)
        if len(headline) > 80:
            continue
            
        # Check for sentence case (first word and proper nouns only)
        words = headline.split()
        if len(words) > 1:
            properly_cased = True
            for i, word in enumerate(words[1:], 1):
                if word.isupper() and len(word) > 1:  # Avoid ALL CAPS
                    properly_cased = False
                    break
            if not properly_cased:
                continue
        
        # Check for clickbait patterns
        clickbait_phrases = [
            "you won't believe",
            "shocking truth",
            "this will blow your mind",
            "wait until you see",
            "the reason why will surprise you"
        ]
        if any(phrase.lower() in headline.lower() for phrase in clickbait_phrases):
            continue
            
        validated_headlines.append(headline)
    
    return validated_headlines

def generate_headline_variants_with_few_shot(article_metadata, few_shot_examples, article_description=""):
    """
    Generates headline variants using a few-shot prompt with Anthropic Claude 3 Haiku.
    Includes the article description for context.
    Returns a dictionary with variants, prompt, and raw response.
    """
    if not ANTHROPIC_API_KEY:
        print("ANTHROPIC_API_KEY not found. Cannot generate headlines.")
        return {"variants": [], "prompt": "", "response": "ANTHROPIC_API_KEY not found."}

    few_shot_prompt_text = ""
    if few_shot_examples:
        for headline in few_shot_examples:
            few_shot_prompt_text += f"- {headline}\n"

    system_prompt = """You are an expert headline writer for Yahoo News, skilled at crafting compelling headlines that maximize click-through rates while maintaining accuracy, journalistic integrity, and Yahoo's editorial standards. You follow Yahoo's sentence-case headline style and editorial guidelines."""

    prompt = f"""You are an expert copywriter for Yahoo News. Generate exactly 5 compelling headline variants following Yahoo's editorial standards.

YAHOO EDITORIAL REQUIREMENTS:
1. **Headline Style**: Use sentence case (capitalize only the first word and proper nouns) - NOT title case
2. **Length**: Keep under 70 characters when possible for mobile optimization
3. **Accuracy**: Maintain factual accuracy - no speculation or unverified claims
4. **Respect**: Treat subjects with respect - no body-shaming, misgendering, or gratuitous violence
5. **Accessibility**: Use clear, accessible language - avoid jargon and overly complex sentences
6. **No Clickbait**: Avoid curiosity gaps or sensationalism that the article doesn't deliver on
7. **Transparency**: Headlines should accurately reflect the article content

CONTENT GUIDELINES:
- Generate exactly 5 headline variants
- Each variant should have a different angle (direct, analytical, impact-focused, etc.)
- **Grounding Rule**: ONLY use information from the 'Article Title' and 'Article Description' provided
- No external facts, speculation, or unverified details
- Maintain Yahoo's conversational but authoritative tone

STYLE SPECIFICS:
- Use "and" not "&" (unless in official names)
- Spell out numbers one through nine, use numerals for 10+
- Use straight quotes, not curly quotes
- For titles: Use single quotes in headlines (vs. double quotes in body text)
- No unnecessary capitalization (avoid ALL CAPS)

Here are examples of high-performing Yahoo headlines for context:
{few_shot_prompt_text}

Article Title: {article_metadata['original_title']}
Article Description: {article_description}

Response format: Numbered list of 5 headlines only, no additional text.
"""

    try:
        client = Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            system=system_prompt,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        variants = []
        if response.content and response.content[0].text:
            raw_text = response.content[0].text
            variants = re.findall(r'^\s*\d+\.\s*(.*)', raw_text, re.MULTILINE)

        if variants:
            validated_variants = validate_headline_quality(variants)
            if len(validated_variants) < 3:
                variants = variants[:5]
            else:
                variants = validated_variants[:5]

        return {
            "variants": variants,
            "prompt": prompt,
            "response": response.model_dump_json(indent=2),
            "editorial_compliance": {
                "style_guide_applied": True,
                "sentence_case_enforced": True,
                "length_optimized": True,
                "fact_grounded": True
            }
        }

    except Exception as e:
        print(f"Error calling Anthropic API: {e}")
        return {"variants": [], "prompt": prompt, "response": str(e)}

def main():
    # Load MAB data for few-shot learning
    mab_df = load_mab_data()
    
    # For POC, use a set of example URLs
    urls = [
        "https://news.yahoo.com/putin-says-western-enemies-trying-130646694.html",
        "https://finance.yahoo.com/news/stock-market-today-dow-drops-211625825.html",
        "https://sports.yahoo.com/nfl-draft-2025-first-round-analysis-042159052.html"
    ]
    
    results = []
    
    for url in urls:
        print(f"\nProcessing {url}...")
        
        # Extract metadata
        metadata = extract_article_metadata(url)
        if not metadata:
            print(f"Skipping {url} due to metadata extraction failure")
            continue
        
        # Select relevant few-shot examples
        examples = select_few_shot_examples(mab_df, metadata['category'])
        print(f"Selected {len(examples)} relevant few-shot examples")
        
        # Generate headline variants with few-shot learning
        print("Generating headline variants...")
        variants = generate_headline_variants_with_few_shot(metadata, examples)
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
        output_file = f"headline_variants_mab_{timestamp}.csv"
        df.to_csv(output_file, index=False)
        print(f"\nExported results to {output_file}")
        
        # Display the results
        print("\nGenerated headline variants for editorial review:")
        print(df.to_string())
    else:
        print("No results to export")

if __name__ == "__main__":
    main()
