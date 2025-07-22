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
    generate_headline_variants_zero_shot,  # Import the new zero-shot function
    ANTHROPIC_API_KEY,
    MAB_DATA_CSV
)

st.set_page_config(layout="wide", page_title="Auto-MAB POC", page_icon="📰")

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

def process_article_url(article, mab_df, corpus_embeddings, model, comparison_mode):
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

    # --- Few-Shot Generation ---
    few_shot_result = generate_headline_variants_with_few_shot(article_metadata, few_shot_examples, examples_are_limited, ANTHROPIC_API_KEY, article_description)

    # --- Zero-Shot Generation (if in comparison mode) ---
    zero_shot_result = None
    if comparison_mode:
        zero_shot_result = generate_headline_variants_zero_shot(article_metadata, ANTHROPIC_API_KEY, article_description)

    # Prioritize returning an error from the few-shot result if it exists
    if few_shot_result.get("error"):
        return {
            "url": url,
            "title": title,
            "few_shot_variants": None,
            "error": few_shot_result.get("error"),
            "few_shot_prompt": few_shot_result.get("prompt"),
            "response": few_shot_result.get("response")
        }

    return {
        "url": url,
        "title": title,
        "few_shot_variants": few_shot_result.get("variants"),
        "zero_shot_variants": zero_shot_result.get("variants") if zero_shot_result else None,
        "error": None,
        "few_shot_prompt": few_shot_result.get("prompt"),
        "zero_shot_prompt": zero_shot_result.get("prompt") if zero_shot_result else None,
        "response": few_shot_result.get("response"),
        "editorial_compliance": few_shot_result.get("editorial_compliance")
    }

# Streamlit UI
st.title("Auto-MAB Headline Generator")

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

    /* Responsive A/B comparison container */
    .comparison-container {{
        display: flex;
        flex-direction: row;
        gap: 2rem; /* space between columns */
    }}
    .comparison-column {{
        flex: 1;
    }}

    /* On smaller screens, stack the columns */
    @media (max-width: 1100px) {{
        .comparison-container {{
            flex-direction: column;
        }}
    }}

    /* Ensure code blocks wrap text properly */
    .comparison-column code {{
        white-space: normal !important;
        word-break: break-word;
    }}
</style>
""", unsafe_allow_html=True)

st.write("Enter a news sitemap URL to generate headline variants for the articles within.")

max_articles = st.sidebar.number_input("Number of articles to process", min_value=1, max_value=20, value=5, step=1)
comparison_mode = st.sidebar.checkbox("Enable A/B Comparison (Few-Shot vs. Zero-Shot)", value=True)
show_flags = st.sidebar.checkbox("Show Validation Flags", value=False)

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
                                executor.submit(process_article_url, article, mab_df, corpus_embeddings, model, comparison_mode): article
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
                        if result.get('few_shot_variants'):
                            # Store results for CSV download
                            result_row = {"URL": result['url'], "Original Title": result['title']}
                            for j, variant in enumerate(result['few_shot_variants']):
                                result_row[f"Variant {j+1}"] = variant['headline']
                            all_results_for_csv.append(result_row)

                            # Display in an expandable section
                            with st.expander(f"**{result['title']}**"):
                                st.markdown(f"<small><a href='{result['url']}' target='_blank' style='text-decoration: none;'>🔗 View Article</a></small>", unsafe_allow_html=True)
                                
                                if comparison_mode and result.get('zero_shot_variants'):
                                    # Generate HTML for the few-shot column
                                    few_shot_html = "<div class='comparison-column'><b>Few-Shot Variants (with examples):</b><ul>"
                                    for variant in result['few_shot_variants']:
                                        few_shot_html += f"<li><code>{variant['headline']}</code></li>"
                                        if show_flags and variant['status'] != 'valid':
                                            few_shot_html += f"<small> &nbsp; &nbsp; 🚩 <b>Flagged:</b> {variant['reason']}</small>"
                                    few_shot_html += "</ul></div>"

                                    # Generate HTML for the zero-shot column
                                    zero_shot_html = "<div class='comparison-column'><b>Zero-Shot Variants (no examples):</b><ul>"
                                    for variant in result['zero_shot_variants']:
                                        zero_shot_html += f"<li><code>{variant['headline']}</code></li>"
                                        if show_flags and variant['status'] != 'valid':
                                            zero_shot_html += f"<small> &nbsp; &nbsp; 🚩 <b>Flagged:</b> {variant['reason']}</small>"
                                    zero_shot_html += "</ul></div>"

                                    # Display the two columns inside the responsive container
                                    st.markdown(f"<div class='comparison-container'>{few_shot_html}{zero_shot_html}</div>", unsafe_allow_html=True)
                                else:
                                    st.markdown("**Generated Variants:**")
                                    for variant_data in result['few_shot_variants']:
                                        headline = variant_data['headline']
                                        status = variant_data.get('status', 'failure')
                                        reason = variant_data.get('reason', 'N/A')
                                        st.markdown(f"- `{headline}`")
                                        if show_flags and status != 'valid':
                                            st.markdown(f"<small> &nbsp; &nbsp; 🚩 **Flagged:** {reason}</small>", unsafe_allow_html=True)

                                # Add a nested expander for the prompt and response
                                with st.expander("View Prompts & Response"):
                                    if comparison_mode and result.get('zero_shot_prompt'):
                                        st.markdown("**Few-Shot Prompt:**")
                                        st.code(result.get('few_shot_prompt', 'Prompt not available.'), language='text')
                                        st.markdown("**Zero-Shot Prompt:**")
                                        st.code(result.get('zero_shot_prompt', 'Prompt not available.'), language='text')
                                    else:
                                        st.markdown("**Prompt sent to AI:**")
                                        st.code(result.get('few_shot_prompt', 'Prompt not available.'), language='text')
                                    
                                    st.markdown("**Raw response from AI (Few-Shot):**")
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
