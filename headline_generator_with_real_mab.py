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

def select_few_shot_examples(
    article_title, mab_df, corpus_embeddings, model, num_examples=3, min_similarity=0.5, diversity_threshold=0.7
):
    """
    Selects a diverse set of high-performing few-shot examples as a simple list of strings.
    """
    if corpus_embeddings is None or mab_df.empty:
        return []

    title_embedding = model.encode(article_title, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(title_embedding, corpus_embeddings)[0]
    
    top_k = min(num_examples * 10, len(mab_df))
    top_results = torch.topk(cosine_scores, k=top_k)

    similar_indices = [
        idx.item() for score, idx in zip(top_results[0], top_results[1]) if score >= min_similarity
    ]

    # This list will hold the final headline strings
    final_headlines = []

    if len(similar_indices) >= num_examples:
        # Create a list of tuples: (integer_index, performance_score)
        candidate_performance = mab_df.iloc[similar_indices][11].tolist()
        candidates = list(zip(similar_indices, candidate_performance))
        
        # Sort candidates by performance score (descending)
        sorted_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)

        selected_indices = [] # This will store the valid integer indices
        for idx, _ in sorted_candidates:
            if len(selected_indices) >= num_examples:
                break
            
            if not selected_indices:
                selected_indices.append(idx)
                continue
            
            # 'idx' is the correct integer index for corpus_embeddings
            candidate_embedding = corpus_embeddings[idx]
            
            # 'selected_indices' also contains correct integer indices
            embeddings_of_selected = corpus_embeddings[selected_indices]
            
            diversity_scores = util.pytorch_cos_sim(candidate_embedding, embeddings_of_selected)[0]

            if torch.max(diversity_scores) < diversity_threshold:
                selected_indices.append(idx)
        
        # If we still don't have enough examples, fill with the next best performers
        if len(selected_indices) < num_examples:
            remaining_needed = num_examples - len(selected_indices)
            # Get the indices of the remaining candidates
            remaining_candidate_indices = [c[0] for c in sorted_candidates if c[0] not in selected_indices]
            selected_indices.extend(remaining_candidate_indices[:remaining_needed])

        # Get the final headlines using the selected integer indices
        final_headlines = mab_df.iloc[selected_indices][8].tolist()
    else:
        # Fallback to global top performers
        top_performers = mab_df.sort_values(by=11, ascending=False).head(num_examples)
        final_headlines = top_performers[8].tolist()

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

def generate_headline_variants_with_few_shot(article_metadata, few_shot_examples, article_description=""):
    """
    Generates headline variants using a few-shot prompt with Anthropic Claude 3 Haiku.
    Includes the article description for context.
    Returns a dictionary with variants, prompt, and raw response.
    """
    if not ANTHROPIC_API_KEY:
        print("ANTHROPIC_API_KEY not found. Cannot generate headlines.")
        return {"variants": [], "prompt": "", "response": "ANTHROPIC_API_KEY not found."}

    # Construct the few-shot examples part of the prompt
    few_shot_prompt_text = ""
    if few_shot_examples:
        for headline in few_shot_examples:
            few_shot_prompt_text += f"- {headline}\n"

    # Construct the main prompt
    prompt = f"""You are an expert copywriter specializing in viral news headlines. Your task is to generate exactly 5 compelling headline variants for the provided article.

Follow these rules strictly:
1.  Generate exactly 5 headline variants.
2.  The variants should be diverse in their angle and tone (e.g., some direct, some intriguing, some controversial).
3.  **Grounding Rule**: You MUST ONLY use information found in the 'Article Title' and 'Article Description' provided below. Do not introduce any external facts, names, or details.
4.  Your response MUST be ONLY a numbered list of the 5 new headline variants. Do not include any introductory text, conversational filler, or any text other than the numbered list of headlines.

Here are some examples of high-performing headlines for similar articles:
{few_shot_prompt_text}

Now, based on the article below, provide your response.

Article Title: {article_metadata['original_title']}
Article Description: {article_description}

Your response:
"""

    try:
        client = Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        variants = []
        if response.content and response.content[0].text:
            raw_text = response.content[0].text
            # Use regex to robustly find numbered list items
            variants = re.findall(r'^\s*\d+\.\s*(.*)', raw_text, re.MULTILINE)

        return {
            "variants": variants,
            "prompt": prompt,
            "response": response.model_dump_json(indent=2)
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
