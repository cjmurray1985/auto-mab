# AI-Powered Headline Variant Generator

This project is a web-based tool designed to help content editors and marketers generate optimized headline variants for news articles. It uses a combination of historical performance data from a Multi-Armed Bandit (MAB) system, semantic search for few-shot example selection, and generative AI to create contextually relevant and high-performing headline suggestions.

## Core Features

- **Sitemap Integration**: Automatically parses news sitemaps to discover and process articles.
- **Article Scraping**: Uses `Newspaper3k` to scrape article titles and descriptions for rich context.
- **Context-Aware Few-Shot Prompting**: Employs semantic search to find historically high-performing headlines that are contextually similar to the new article, using them as dynamic few-shot examples.
- **Unified & Transparent Prompting**: Consolidates all editorial guidelines and instructions into a single, transparent prompt that is displayed in the UI for full visibility.
- **AI-Powered Generation**: Leverages the Anthropic Claude 3 Haiku model to generate five diverse headline variants.
- **Detailed Validation & Flagging**:
    - Validates every generated headline against editorial rules (e.g., length, capitalization).
    - Instead of a simple pass/fail, non-compliant headlines are **flagged (🚩)** with a clear reason.
    - This system acknowledges that flagged headlines may still be viable for MAB testing.
- **Robust API Interaction**: Automatically retries API calls up to 3 times to handle transient network or service errors gracefully.
- **Interactive UI**: Built with Streamlit, the app provides a user-friendly interface to input sitemaps, monitor progress, and review generated headlines with their validation status.
- **Transparent Error Reporting**: If an unrecoverable error occurs, the UI displays the full technical traceback for effective debugging.
- **High-Performance Concurrent Processing**: Processes multiple articles in parallel using a `ThreadPoolExecutor` to deliver results quickly.

## Tech Stack

- **Backend**: Python
- **Web Framework**: Streamlit
- **Core Libraries**:
    - `pandas`: For data manipulation.
    - `requests` & `lxml`: For fetching and parsing sitemaps.
    - `newspaper3k`: For scraping article content.
    - `sentence-transformers`: For semantic search and headline embeddings.
    - `anthropic`: For interacting with the Claude API.
    - `python-dotenv`: For managing environment variables.

## Setup and Installation

Follow these steps to get the application running on your local machine.

**1. Clone the Repository**

```bash
git clone <repository_url>
cd <repository_name>
```

**2. Create a Virtual Environment**

It is highly recommended to use a virtual environment to manage project dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**3. Install Dependencies**

Install all the required Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

**4. Create an Environment File**

You will need to provide an API key for the Anthropic service. Create a file named `.env` in the root of the project directory and add your key like this:

```
ANTHROPIC_API_KEY="your_anthropic_api_key_here"
```

## How to Run the Application

Once the setup is complete, you can launch the Streamlit application with the following command:

```bash
streamlit run app.py
```

Your web browser should automatically open with the application running. The first time you run it, it will download the sentence-transformer model, which may take a moment. This is a one-time process.

## How It Works

1.  **Sitemap Input**: The user provides a URL to a news sitemap.
2.  **Parsing**: The app fetches and parses the sitemap, handling nested sitemap indexes, to extract a list of all article URLs.
3.  **Data Loading & Embedding**: The historical MAB data from `Example data  Sheet1.csv` is loaded into a pandas DataFrame. The `sentence-transformers` model then converts every headline in this dataset into a numerical vector (embedding). This process is cached for efficiency.
4.  **Concurrent Headline Generation**:
    - The application processes up to 10 articles at a time in parallel. For each article:
    - **Scraping**: The article's title and description are scraped using `Newspaper3k`.
    - **Few-Shot Selection**: The app finds the most semantically similar headlines from the historical data to use as few-shot examples.
    - **Prompting**: A unified prompt is constructed containing all editorial rules, the article context, and the few-shot examples. This full prompt is visible in the UI.
    - **Generation & Validation**: The AI generates 5 new headline variants. Each variant is immediately validated against editorial guidelines (e.g., length, capitalization).
5.  **Display Results**: The generated variants are displayed in the Streamlit UI. Headlines that pass all checks are shown normally, while non-compliant ones are **flagged (🚩)** with a clear explanation on a separate line. This allows editors to quickly assess quality while still considering all variants for MAB testing.
