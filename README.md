# AI-Powered Headline Variant Generator

This project is a web-based tool designed to help content editors and marketers generate optimized headline variants for news articles. It uses a sophisticated combination of semantic search, editorial compliance filtering, and adaptive AI prompting to create contextually relevant and high-performing headline suggestions.

## Core Features

- **Sitemap Integration**: Automatically parses news sitemaps to discover and process articles.
- **Article Scraping**: Uses `Newspaper3k` to scrape article titles and descriptions for rich context.
- **Hybrid Few-Shot Example Selection**:
    - Employs semantic search to find the top 50 most contextually similar headlines from a historical MAB dataset.
    - Filters these results for strict editorial compliance to identify the best possible examples.
    - Ensures all selected examples are unique by performing deduplication.
- **Adaptive Prompt Construction**:
    - Dynamically adjusts the AI prompt based on the quality and quantity of available examples.
    - If 3 or more high-quality examples are found, it instructs the AI to study their structure and tone.
    - If examples are limited, it provides special guidance, instructing the AI to extract general engagement principles rather than copying specific styles, ensuring robust performance even with sparse data.
- **AI-Powered Generation**: Leverages the Anthropic Claude 3 Haiku model to generate five diverse headline variants based on precise, multi-layered editorial instructions.
- **Detailed Validation & Flagging**:
    - Validates every generated headline against strict editorial rules (e.g., length, capitalization, style).
    - Assigns a detailed status (`valid`, `warning`, or `failure`) to each variant.
    - Non-compliant headlines are flagged (🚩) with a clear reason, allowing editors to review all suggestions while understanding their compliance level.
- **Robust API Interaction**: Automatically retries API calls up to 3 times to handle transient network or service errors gracefully.
- **Interactive UI**: Built with Streamlit, the app provides a user-friendly interface to input sitemaps, monitor progress, and review generated headlines with their validation status. The full, final prompt sent to the AI is displayed for complete transparency.
- **High-Performance Concurrent Processing**: Processes multiple articles in parallel using a `ThreadPoolExecutor` to deliver results quickly.

## Tech Stack

- **Backend**: Python
- **Web Framework**: Streamlit
- **Core Libraries**:
    - `pandas`: For data manipulation.
    - `requests` & `lxml`: For fetching and parsing sitemaps.
    - `newspaper3k`: For scraping article content.
    - `sentence-transformers` & `torch`: For semantic search and GPU-accelerated embeddings.
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
3.  **Data Loading & Embedding**: The historical MAB data from `Example data  Sheet1.csv` is loaded. The `sentence-transformers` model then converts every headline in this dataset into a numerical vector (embedding). This process is cached for efficiency.
4.  **Concurrent Headline Generation**:
    - The application processes multiple articles at a time in parallel. For each article:
    - **Scraping**: The article's title and description are scraped.
    - **Hybrid Few-Shot Selection**: The app finds the top 50 most semantically similar headlines from the historical data. It then filters this list for editorial compliance and uniqueness to select the best possible examples.
    - **Adaptive Prompting**: A final prompt is constructed containing the precise editorial rules, the article context, and the dynamically selected few-shot examples. The prompt adapts its guidance based on the quality of the examples found.
    - **Generation & Validation**: The AI generates 5 new headline variants. Each is immediately validated, and a status (`valid`, `warning`, `failure`) is assigned.
5.  **Display Results**: The generated variants are displayed in the Streamlit UI. Non-compliant headlines are flagged (🚩) with a clear explanation, allowing editors to quickly assess quality while still considering all variants for MAB testing.
