import streamlit as st
import requests
from lxml import etree
import pandas as pd
from sentence_transformers import SentenceTransformer

from headline_generator_with_real_mab import (
    load_mab_data,
    select_few_shot_examples,
    scrape_article_description,
    generate_headline_variants_with_few_shot,
    ANTHROPIC_API_KEY,
    MAB_DATA_CSV
)

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
        headlines = mab_df[8].astype(str).tolist()
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
    div[data-testid="stDownloadButton"] button {{
        background-color: rgb(126, 31, 255) !important;
        color: white !important;
        border: 1px solid rgb(126, 31, 255) !important;
    }}
    div[data-testid="stDownloadButton"] button:hover {{
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

max_articles = st.sidebar.number_input("Number of articles to process", min_value=1, max_value=100, value=5, step=1)

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
                    progress_bar = st.progress(0)
                    num_articles = len(articles_to_process)

                    for i, article in enumerate(articles_to_process):
                        article_metadata = {
                            "original_title": article['title'],
                            "description": "",
                            "category": "News",
                            "url": article['url']
                        }
                        
                        article_description = None
                        variants = None

                        # Scrape article description with its own spinner for clarity
                        with st.spinner(f"Step 1/2: Scraping description for *{article['title']}*..."):
                            article_description, scrape_error = scrape_article_description(article['url'])

                        # Generate headlines only if scraping was successful
                        if article_description:
                            with st.spinner(f"Step 2/2: Generating variants for *{article['title']}*..."):
                                model = load_embedding_model()
                                few_shot_examples = select_few_shot_examples(
                                    article['title'], mab_df, corpus_embeddings, model
                                )
                                variants = generate_headline_variants_with_few_shot(
                                    article_metadata, few_shot_examples, article_description
                                )
                        else:
                            st.error(f"Could not scrape description for *{article['title']}*. Reason: {scrape_error}")
                        
                        if variants:
                            # Store results for CSV download
                            result_row = {"URL": article['url'], "Original Title": article['title']}
                            for j, variant in enumerate(variants):
                                result_row[f"Variant {j+1}"] = variant
                            all_results_for_csv.append(result_row)

                            # Display in an expandable section
                            with st.expander(f"**{article['title']}**"):
                                st.markdown(f"<small><a href='{article['url']}' target='_blank' style='text-decoration: none;'>🔗 View Article</a></small>", unsafe_allow_html=True)
                                st.markdown("**Generated Variants:**")
                                for k, variant in enumerate(variants):
                                    st.markdown(f"> {k+1}. {variant}")
                        elif article_description: # Only show this warning if scraping succeeded but generation failed
                            st.warning(f"Could not generate variants for: *{article['title']}*.")
                        
                        # Update overall progress bar
                        progress_bar.progress((i + 1) / num_articles)

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
