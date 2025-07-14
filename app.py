import streamlit as st
import requests
from lxml import etree
import pandas as pd
from sentence_transformers import SentenceTransformer
import concurrent.futures

from headline_generator_with_real_mab import (
    load_mab_data,
    select_few_shot_examples,
    scrape_article_description,
    generate_headline_variants_with_few_shot,
    ANTHROPIC_API_KEY,
    MAB_DATA_CSV
)

st.set_page_config(page_title="Auto-MAB POC", page_icon="📰")

@st.cache_resource
def load_embedding_model():
    """Loads the SentenceTransformer model and caches it."""
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_and_embed_data(csv_path):
    """Loads MAB data and computes headline embeddings."""
    mab_df = load_mab_data(csv_path)
    if not mab_df.empty:
        model = load_embedding_model()
        headlines = mab_df['headline'].astype(str).tolist()
        st.info("Creating headline embeddings for semantic search... This is a one-time process.")
        corpus_embeddings = model.encode(headlines, convert_to_tensor=True, show_progress_bar=True)
        return mab_df, corpus_embeddings
    return pd.DataFrame(), None

# Function to fetch and parse sitemap
def get_urls_from_sitemap(sitemap_url):
    """Fetches a sitemap (or sitemap index) and extracts URLs and titles."""
    urls = []
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(sitemap_url, headers=headers, timeout=20)
        response.raise_for_status()
        
        content = response.content
        root = etree.fromstring(content)
        
        # Define namespaces, including the default sitemap namespace
        ns = {
            'sitemap': 'http://www.sitemaps.org/schemas/sitemap/0.9',
            'news': 'http://www.google.com/schemas/sitemap-news/0.9'
        }
        
        # Check if it's a sitemap index file
        if root.tag.endswith('sitemapindex'):
            st.info(f"Found sitemap index: {sitemap_url}. Parsing sub-sitemaps...")
            sitemap_links = root.xpath('//sitemap:sitemap/sitemap:loc', namespaces=ns)
            for link in sitemap_links:
                # Recursively call the function for each sub-sitemap
                urls.extend(get_urls_from_sitemap(link.text))
            return urls

        # If it's a regular sitemap, parse for URLs
        url_elements = root.xpath('//sitemap:url', namespaces=ns)
        
        for url_element in url_elements:
            loc_element = url_element.find('sitemap:loc', namespaces=ns)
            if loc_element is None:
                continue
            loc = loc_element.text
            
            news_title_element = url_element.find('news:news/news:title', namespaces=ns)
            
            if news_title_element is not None and news_title_element.text:
                title = news_title_element.text
                urls.append({"url": loc, "title": title})
                
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching sitemap: {e}")
    except etree.XMLSyntaxError as e:
        st.error(f"Error parsing sitemap XML: {e}")
    return urls

def process_article_url(article, mab_df, corpus_embeddings, model):
    """
    Processes a single article URL to scrape it and generate headline variants.
    This function is designed to be run in a thread pool.
    """
    url = article['url']
    title = article['title']

    # Scrape description
    article_description, scrape_error = scrape_article_description(url)

    if not article_description:
        return {"url": url, "title": title, "variants": None, "error": f"Scraping failed: {scrape_error}"}

    # Select few-shot examples
    few_shot_examples, examples_are_limited = select_few_shot_examples(title, mab_df, corpus_embeddings, model)

    # Generate variants
    article_metadata = {
        "original_title": title,
        "description": article_description,
        "category": "News",  # Assuming default
        "url": url
    }
    generation_result = generate_headline_variants_with_few_shot(article_metadata, few_shot_examples, examples_are_limited, ANTHROPIC_API_KEY, article_description)

    variants = generation_result.get("variants")
    prompt = generation_result.get("prompt")
    raw_response = generation_result.get("response")
    editorial_compliance = generation_result.get("editorial_compliance")

    # Prioritize returning a specific error from the backend if it exists
    if generation_result.get("error"):
        return {
            "url": url, 
            "title": title, 
            "variants": None, 
            "error": generation_result.get("error"), 
            "prompt": prompt, 
            "response": raw_response
        }

    return {
        "url": url, 
        "title": title, 
        "variants": variants, 
        "error": None, 
        "prompt": prompt, 
        "response": raw_response, 
        "editorial_compliance": editorial_compliance
    }

# Streamlit UI
st.title("Headline Variant Generator")

# Custom CSS to override button and expander colors
st.markdown(f"""
<style>
    /* The nuclear option: Target the dynamically generated class directly. */
    .st-emotion-cache-1rwb540 {{
        background-color: rgb(126, 31, 255) !important;
        color: white !important;
        border: 1px solid rgb(126, 31, 255) !important;
    }}
    .st-emotion-cache-1rwb540:hover {{
        background-color: rgb(100, 25, 200) !important;
        color: white !important;
        border: 1px solid rgb(100, 25, 200) !important;
    }}
    .st-emotion-cache-1rwb540:active {{
        background-color: rgb(80, 20, 160) !important;
        color: white !important;
        border: 1px solid rgb(80, 20, 160) !important;
    }}

    /* Target the 'Download' button directly */
    div[data-testid=\"stDownloadButton\"] button {{
        background-color: rgb(126, 31, 255) !important;
        color: white !important;
        border: 1px solid rgb(126, 31, 255) !important;
    }}
    div[data-testid=\"stDownloadButton\"] button:hover {{
        background-color: rgb(100, 25, 200) !important;
        color: white !important;
        border: 1px solid rgb(100, 25, 200) !important;
    }}

    /* Expander header color */
    .st-expander > summary {{
        background-color: rgba(126, 31, 255, 0.1) !important;
    }}
    .st-expander > summary:hover {{
        background-color: rgba(126, 31, 255, 0.2) !important;
    }}
</style>
""", unsafe_allow_html=True)

st.write("Enter a news sitemap URL to generate headline variants for the articles within.")

max_articles = st.sidebar.number_input("Number of articles to process", min_value=1, max_value=20, value=5, step=1)

with st.form(key='sitemap_form'):
    sitemap_url = st.text_input("Enter News Sitemap URL", "https://www.yahoo.com/news-sitemap.xml")
    submit_button = st.form_submit_button(label='Generate Headlines')

if submit_button:
    if sitemap_url:
        with st.spinner("Loading MAB data and processing sitemap..."):
            mab_df, corpus_embeddings = load_and_embed_data(MAB_DATA_CSV)
            if mab_df.empty:
                st.error("Failed to load MAB data. Please check the CSV file.")
            else:
                articles = get_urls_from_sitemap(sitemap_url)
                if not articles:
                    st.warning("No articles with <news:title> tags found in the sitemap.")
                else:
                    articles_to_process = articles[:max_articles]
                    st.info(f"Found {len(articles)} articles. Processing the first {len(articles_to_process)}.")

                    st.subheader("Generated Headlines")
                    all_results_for_csv = []
                    
                    # Use a ThreadPoolExecutor to process articles in parallel
                    with st.spinner(f"Processing {len(articles_to_process)} articles concurrently..."):
                        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                            # Prepare futures
                            model = load_embedding_model()
                            future_to_article = {
                                executor.submit(process_article_url, article, mab_df, corpus_embeddings, model): article
                                for article in articles_to_process
                            }

                            results = []
                            progress_bar = st.progress(0)
                            num_completed = 0
                            total_articles = len(articles_to_process)

                            for future in concurrent.futures.as_completed(future_to_article):
                                try:
                                    result = future.result()
                                    results.append(result)
                                except Exception as e:
                                    article_info = future_to_article[future]
                                    results.append({
                                        "url": article_info['url'],
                                        "title": article_info['title'],
                                        "variants": None,
                                        "error": f"An exception occurred: {e}"
                                    })
                                finally:
                                    num_completed += 1
                                    progress_bar.progress(num_completed / total_articles)

                    # Process and display results
                    for result in sorted(results, key=lambda r: articles_to_process.index(next(a for a in articles_to_process if a['url'] == r['url']))):
                        if result['variants']:
                            # Store results for CSV download
                            result_row = {"URL": result['url'], "Original Title": result['title']}
                            for j, variant in enumerate(result['variants']):
                                result_row[f"Variant {j+1}"] = variant
                            all_results_for_csv.append(result_row)

                            # Display in an expandable section
                            with st.expander(f"**{result['title']}**"):
                                st.markdown(f"<small><a href='{result['url']}' target='_blank' style='text-decoration: none;'>🔗 View Article</a></small>", unsafe_allow_html=True)
                                st.markdown("**Generated Variants:**")
                                for variant_data in result['variants']:
                                    headline = variant_data['headline']
                                    status = variant_data.get('status', 'failure')
                                    reason = variant_data.get('reason', 'N/A')

                                    # Use columns to ensure consistent formatting for all headlines
                                    col1, col2 = st.columns([1, 19])
                                    with col1:
                                        st.write("•") # Bullet point
                                    with col2:
                                        st.markdown(f"`{headline}`")
                                        if status != 'valid':
                                            st.markdown(f"<small>🚩 **Flagged:** {reason}</small>", unsafe_allow_html=True)

                                # Add a nested expander for the prompt and response
                                with st.expander("View Prompt & Response"):
                                    st.markdown("**Prompt sent to AI:**")
                                    st.code(result.get('prompt', 'Prompt not available.'), language='text')
                                    st.markdown("**Raw response from AI:**")
                                    st.json(result.get('response', 'Response not available.'))
                        else:
                            with st.expander(f"❌ **Error processing:** {result['title']}", expanded=True):
                                st.error("An unrecoverable error occurred after multiple retries. See the full error log below.")
                                st.markdown("**Full Error Log:**")
                                st.code(result.get('error', 'No error log available.'), language='log')
                                st.markdown("**Prompt that caused the error:**")
                                st.code(result.get('prompt', 'Prompt not available.'), language='text')

                    # Add a download button for the CSV
                    if all_results_for_csv:
                        csv_df = pd.DataFrame(all_results_for_csv)
                        csv_data = csv_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download All Headlines as CSV",
                            data=csv_data,
                            file_name='headline_variants.csv',
                            mime='text/csv',
                        )

                    if all_results_for_csv:
                        st.success("Headline generation complete!")
                        results_df = pd.DataFrame(all_results_for_csv)
                        csv = results_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download all results as CSV",
                            data=csv,
                            file_name='headline_variants.csv',
                            mime='text/csv',
                        )
                    else:
                        st.warning("Could not generate headlines for any of the articles.")
    else:
        st.warning("Please enter a sitemap URL.")
