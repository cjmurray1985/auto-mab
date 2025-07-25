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

def process_article_url(article, mab_df, corpus_embeddings, model, test_mode):
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

    article_metadata = {
        "original_title": title,
        "description": article_description,
        "category": "News",
        "url": url
    }

    # --- Column A Generation ---
    # Always generate compliant few-shot for Column A or standard mode
    few_shot_compliant_examples, examples_are_limited = select_few_shot_examples(title, mab_df, corpus_embeddings, model, filter_by_compliance=True)
    col_a_result = generate_headline_variants_with_few_shot(article_metadata, few_shot_compliant_examples, examples_are_limited, ANTHROPIC_API_KEY, article_description)
    col_a_title = "Few-Shot (Compliant)"

    # --- Column B Generation (for A/B tests) ---
    col_b_result = None
    col_b_title = None
    if test_mode == "A/B Test: Compliant vs. Unfiltered Few-Shot":
        few_shot_unfiltered_examples, _ = select_few_shot_examples(title, mab_df, corpus_embeddings, model, filter_by_compliance=False)
        col_b_result = generate_headline_variants_with_few_shot(article_metadata, few_shot_unfiltered_examples, False, ANTHROPIC_API_KEY, article_description)
        col_b_title = "Few-Shot (Unfiltered)"
    elif test_mode == "A/B Test: Compliant Few-Shot vs. Zero-Shot":
        col_b_result = generate_headline_variants_zero_shot(article_metadata, ANTHROPIC_API_KEY, article_description)
        col_b_title = "Zero-Shot"

    # --- Consolidate Results ---
    if col_a_result.get("error"):
        return {"url": url, "title": title, "error": col_a_result.get("error"), "prompt": col_a_result.get("prompt")}

    return {
        "url": url,
        "title": title,
        "col_a_title": col_a_title,
        "col_a_variants": col_a_result.get("variants"),
        "col_a_prompt": col_a_result.get("prompt"),
        "col_b_title": col_b_title,
        "col_b_variants": col_b_result.get("variants") if col_b_result else None,
        "col_b_prompt": col_b_result.get("prompt") if col_b_result else None,
        "response": col_a_result.get("response"), # For simplicity, just show one response
        "error": None
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

TEST_MODES = {
    "Standard": "Standard Generation (Few-Shot Compliant)",
    "Compliant vs. Unfiltered": "A/B Test: Compliant vs. Unfiltered Few-Shot",
    "Compliant vs. Zero-Shot": "A/B Test: Compliant Few-Shot vs. Zero-Shot",
}

test_mode_key = st.sidebar.selectbox(
    "Select Test Mode",
    options=TEST_MODES.keys(),
    index=0,  # Default to Standard
    help="""
- **Standard:** Generates one set of headlines using compliant few-shot examples.
- **Compliant vs. Unfiltered:** Compares headlines from compliant examples vs. all semantically similar examples.
- **Compliant vs. Zero-Shot:** Compares headlines from compliant examples vs. no examples.
"""
)
test_mode = TEST_MODES[test_mode_key]

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
                                executor.submit(process_article_url, article, mab_df, corpus_embeddings, model, test_mode): article
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
                        if result.get('col_a_variants'):
                            # Store results for CSV download (from column A)
                            result_row = {"URL": result['url'], "Original Title": result['title']}
                            for j, variant in enumerate(result['col_a_variants']):
                                result_row[f"Variant {j+1}"] = variant['headline']
                            all_results_for_csv.append(result_row)

                            # Display in an expandable section
                            with st.expander(f"**{result['title']}**"):
                                st.markdown(f"<small><a href='{result['url']}' target='_blank' style='text-decoration: none;'>🔗 View Article</a></small>", unsafe_allow_html=True)

                                # A/B Test View
                                if result.get('col_b_variants'):
                                    # Generate HTML for Column A
                                    col_a_html = f"<div class='comparison-column'><b>{result['col_a_title']}:</b><ul>"
                                    for variant in result['col_a_variants']:
                                        col_a_html += f"<li><code>{variant['headline']}</code></li>"
                                        if show_flags and variant['status'] != 'valid':
                                            col_a_html += f"<small> &nbsp; &nbsp; 🚩 <b>Flagged:</b> {variant['reason']}</small>"
                                    col_a_html += "</ul></div>"

                                    # Generate HTML for Column B
                                    col_b_html = f"<div class='comparison-column'><b>{result['col_b_title']}:</b><ul>"
                                    for variant in result['col_b_variants']:
                                        col_b_html += f"<li><code>{variant['headline']}</code></li>"
                                        if show_flags and variant['status'] != 'valid':
                                            col_b_html += f"<small> &nbsp; &nbsp; 🚩 <b>Flagged:</b> {variant['reason']}</small>"
                                    col_b_html += "</ul></div>"

                                    st.markdown(f"<div class='comparison-container'>{col_a_html}{col_b_html}</div>", unsafe_allow_html=True)
                                
                                # Standard View
                                else:
                                    st.markdown(f"**{result['col_a_title']}:**")
                                    for variant_data in result['col_a_variants']:
                                        st.markdown(f"- `{variant_data['headline']}`")
                                        if show_flags and variant_data['status'] != 'valid':
                                            st.markdown(f"<small> &nbsp; &nbsp; 🚩 **Flagged:** {variant_data['reason']}</small>", unsafe_allow_html=True)

                                # Expander for prompts
                                with st.expander("View Prompts & Response"):
                                    st.markdown(f"**Prompt for: {result['col_a_title']}**")
                                    st.code(result.get('col_a_prompt', 'Prompt not available.'), language='text')
                                    if result.get('col_b_prompt'):
                                        st.markdown(f"**Prompt for: {result['col_b_title']}**")
                                        st.code(result.get('col_b_prompt', 'Prompt not available.'), language='text')
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
