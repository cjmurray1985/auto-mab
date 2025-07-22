import os
import time
import json
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import re
from dotenv import load_dotenv
from anthropic import Anthropic
from newspaper import Article
import streamlit as st
import traceback

# Load environment variables
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# File path for MAB data
MAB_DATA_CSV = 'Example data  Sheet1.csv'

def load_mab_data(csv_path=MAB_DATA_CSV):
    """Load and process MAB testing data for few-shot examples"""
    try:
        df = pd.read_csv(csv_path, sep='\t', header=None, engine='python', on_bad_lines='skip')

        # Filter by winning variant (last column must be TRUE)
        last_col_index = df.columns[-1]
        df[last_col_index] = df[last_col_index].astype(str)
        df = df[df[last_col_index].str.upper() == 'TRUE']

        required_cols = [8, 10]
        df = df.dropna(subset=required_cols)
        df[10] = pd.to_numeric(df[10], errors='coerce')
        df = df.dropna(subset=[10])
        
        # Rename columns for clarity
        df = df.rename(columns={8: 'headline', 10: 'performance'})
        print(f"Loaded {len(df)} winning MAB examples after cleaning and filtering")
        return df
    except Exception as e:
        print(f"Error loading MAB data: {e}")
        return pd.DataFrame()

def validate_headline_quality(headlines):
    """Validates a list of headlines against editorial standards."""
    validation_results = []
    for h in headlines:
        fail_reasons = []
        warn_reasons = []
        if len(h) > 100:
            fail_reasons.append(f"Exceeds 100 characters (is {len(h)})")
        elif len(h) > 70:
            warn_reasons.append(f"Exceeds 70 characters (is {len(h)})")
        words = h.split()
        for word in words[1:]:
            if word.isupper() and len(word) > 1:
                fail_reasons.append(f"Contains improperly capitalized word: '{word}'")
                break
        if fail_reasons:
            status = "failure"
            reason = ", ".join(fail_reasons)
        elif warn_reasons:
            status = "warning"
            reason = ", ".join(warn_reasons)
        else:
            status = "valid"
            reason = ""
        validation_results.append({"headline": h, "status": status, "reason": reason})
    return validation_results

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

def select_few_shot_examples(article_title, mab_data, embeddings, model, top_n=5):
    """Selects the best few-shot examples using a hybrid semantic and compliance strategy."""
    query_embedding = model.encode([article_title], convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, embeddings)[0]
    top_indices = torch.topk(similarities, k=top_n * 10).indices.tolist() # Increased pool to 50
    similar_headlines = [mab_data.iloc[idx].to_dict() for idx in top_indices]
    
    compliant_examples = []
    seen_headlines = set()
    for example in similar_headlines:
        headline_text = example['headline']
        if headline_text in seen_headlines:
            continue  # Skip duplicate

        validation_result = validate_headline_quality([headline_text])[0]
        if validation_result['status'] == 'valid':
            compliant_examples.append(example)
            seen_headlines.add(headline_text)
            
    if len(compliant_examples) >= 3:
        return compliant_examples[:top_n], False  # Strategy A: Good examples
    else:
        return compliant_examples, True  # Strategy B: Limited examples

def create_adaptive_prompt(article_metadata, few_shot_examples, examples_are_limited, article_description=""):
    """Adjusts the prompt based on the quality of the available few-shot examples."""
    examples_str = "\n".join([f"- {ex['headline']}" for ex in few_shot_examples])
    if not few_shot_examples:
        examples_str = "No compliant examples were found for similar articles."

    # The full, detailed prompt is now the base
    base_prompt = f"""YAHOO EDITORIAL REQUIREMENTS:
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
"""

    # The adaptive guidance is now appended to the full base prompt
    if examples_are_limited:
        guidance = f"""\n**Guidance on Using Examples:**
IMPORTANT: The examples below are the only compliant headlines we could find for similar articles. They may not be perfectly relevant. Your task is to:
1. EXTRACT GENERAL ENGAGEMENT PRINCIPLES from them (e.g., posing a question, using a strong verb), rather than copying their specific style.
2. APPLY THOSE PRINCIPLES to the specific content of the article you are working on.
3. PRIORITIZE THE EDITORIAL STANDARDS above all else. Your generated headlines must be fully compliant.

Think of these as loose inspiration for engagement techniques, not as templates to follow.
"""
    else:
        guidance = f"""\n**High-Performing Headline Examples:**
Here are some examples of successful, compliant headlines for similar articles. Study their structure and tone to inform your own creations:
"""

    # Combine all parts for the final prompt
    final_prompt = base_prompt + guidance + f"""\n**Headline Examples:**\n{examples_str}\n\n**Article to Process:**\n- **Title:** {article_metadata['original_title']}\n- **Description:** {article_description}\n\nNow, generate 5 headline variants that strictly follow all editorial standards."""
    
    return final_prompt

@st.cache_data
def generate_headline_variants_with_few_shot(article_metadata, few_shot_examples, examples_are_limited, api_key, article_description=""):
    """Generates headline variants using a few-shot prompt with adaptive guidance and retry logic."""
    if not api_key:
        return {"error": "API key not found."}

    prompt = create_adaptive_prompt(article_metadata, few_shot_examples, examples_are_limited, article_description)
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            client = Anthropic(api_key=ANTHROPIC_API_KEY)
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            raw_text = response.content[0].text
            # Improved regex to handle variations in the AI's response format
            variants = re.findall(r'^\s*\d+\.\s*(.*)', raw_text, re.MULTILINE)
            if not variants:
                 variants = raw_text.strip().split('\n')

            validated_variants = validate_headline_quality(variants)
            
            return {
                "variants": validated_variants,
                "prompt": prompt,
                "response": response.to_json()
            }
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                print("All retry attempts failed.")
                error_trace = traceback.format_exc()
                return {"error": f"API Error after multiple retries. See details below.\n\n{error_trace}", "prompt": prompt}
    return {"error": "Exited retry loop unexpectedly.", "prompt": prompt} # Fallback

def create_zero_shot_prompt(article_metadata, article_description=""):
    """Creates a zero-shot prompt without any examples."""
    
    # The base prompt with all editorial and content guidelines
    base_prompt = f"""YAHOO EDITORIAL REQUIREMENTS:
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
"""

    # Combine the base prompt with the article details
    final_prompt = base_prompt + f"""\n\n**Article to Process:**\n- **Title:** {article_metadata['original_title']}\n- **Description:** {article_description}\n\nNow, generate 5 headline variants that strictly follow all editorial standards."""
    
    return final_prompt

@st.cache_data
def generate_headline_variants_zero_shot(article_metadata, api_key, article_description=""):
    """Generates headline variants using a zero-shot prompt with retry logic."""
    if not api_key:
        return {"error": "API key not found."}

    prompt = create_zero_shot_prompt(article_metadata, article_description)
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            client = Anthropic(api_key=ANTHROPIC_API_KEY)
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            raw_text = response.content[0].text
            # Improved regex to handle variations in the AI's response format
            variants = re.findall(r'^\s*\d+\.\s*(.*)', raw_text, re.MULTILINE)
            if not variants:
                 variants = raw_text.strip().split('\n')

            validated_variants = validate_headline_quality(variants)
            
            return {
                "variants": validated_variants,
                "prompt": prompt,
                "response": response.to_json()
            }
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                print("All retry attempts failed.")
                error_trace = traceback.format_exc()
                return {"error": f"API Error after multiple retries. See details below.\n\n{error_trace}", "prompt": prompt}
    return {"error": "Exited retry loop unexpectedly.", "prompt": prompt} # Fallback

