import streamlit as st
import requests
import validators
import json
import xml.etree.ElementTree as ET
import psycopg2
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import json
from typing import List
import json
import config
from nltk.tokenize import sent_tokenize


# UI Configuration
st.set_page_config(page_title="Website Crawler", layout="centered")
st.markdown("""
    <h1 style='text-align: center;'>🌐 Website Crawler</h1>
    <hr>
""", unsafe_allow_html=True)

# PostgreSQL connection setup
DB_CONFIG = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "123456",
    "host": "localhost",
    "port": "5432"
}

def get_table_name(url):
    """Generate a table name based on the website domain."""
    domain = urlparse(url).netloc.replace(".", "_").replace("-", "_")
    return f"scraped_{domain}"

def save_to_postgres(scraped_data, table_name):
    """Save all scraped data to PostgreSQL after processing."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id SERIAL PRIMARY KEY,
                url TEXT UNIQUE,
                title TEXT,
                description TEXT,
                content TEXT,
                scrape_time TIMESTAMP DEFAULT NOW()
            )
        """)

        for data in scraped_data:
            cur.execute(f"""
                INSERT INTO {table_name} (url, title, description, content, scrape_time)
                VALUES (%s, %s, %s, %s, NOW())
                ON CONFLICT (url) DO NOTHING
            """, (data["url"], data["title"], data["description"], data["content"]))

        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        st.error("Error saving data to DB.")

def fetch_sitemap_from_robots(url):
    """Fetch sitemap URLs from robots.txt, ignoring .xml.gz files."""
    parsed_url = urlparse(url)
    robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
    
    try:
        response = requests.get(robots_url, timeout=10)
        response.raise_for_status()
        lines = response.text.split("\n")

        # Extract only valid sitemap URLs, ignoring .xml.gz files
        sitemap_urls = [line.split(": ")[1].strip() for line in lines 
                        if line.lower().startswith("sitemap:") and not line.strip().lower().endswith(".xml.gz")]

        if sitemap_urls:
            st.success(f"✅ Found {len(sitemap_urls)} valid sitemaps in robots.txt")
        else:
            st.warning("⚠️ No valid sitemaps found in robots.txt. Using default `/sitemap.xml` path.")
            sitemap_urls = [f"{parsed_url.scheme}://{parsed_url.netloc}/sitemap.xml"]

        return sitemap_urls

    except requests.exceptions.RequestException:
        st.error("Could not fetch robots.txt. Trying default `/sitemap.xml`.")
        return [f"{parsed_url.scheme}://{parsed_url.netloc}/sitemap.xml"]

def fetch_sitemap_links(sitemap_urls):
    """Fetch and parse the first 25 valid links from each sitemap file."""
    all_links = []

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    for sitemap_url in sitemap_urls:
        try:
            response = requests.get(sitemap_url, timeout=10, headers=headers)
            response.raise_for_status()
            root = ET.fromstring(response.content)

            # Extract <loc> links, ignoring .xml.gz files, and limit to 25 per sitemap
            links = [elem.text for elem in root.iter('{http://www.sitemaps.org/schemas/sitemap/0.9}loc') if not elem.text.endswith(".xml.gz")][:25]
            all_links.extend(links)
        except requests.exceptions.RequestException:
            st.warning(f"⚠️ Could not fetch sitemap: {sitemap_url}")

    if all_links:
        st.success(f"✅ Total {len(all_links)} pages found in sitemaps")
    
    return all_links

def scrape_page(url):
    """Scrape title, meta description, and main content from a webpage."""
    try:

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        response = requests.get(url, timeout=10, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract Title
        title = soup.title.string.strip() if soup.title else "No Title"

        # Extract Meta Description
        meta_desc = soup.find("meta", attrs={"name": "description"})
        description = meta_desc["content"].strip() if meta_desc and "content" in meta_desc.attrs else "No Description"

        # Extract Main Content (First 5 Paragraphs)
        paragraphs = soup.find_all("p")
        content = " ".join([p.get_text(strip=True) for p in paragraphs[:5]])

        return {
            "url": url,
            "title": title,
            "description": description,
            "body": content
        }

    except Exception as e:
        return {"url": url, "error": str(e)}

def semantic_chunking(text, max_chunk_size=512):
    """
    Splits text into semantically meaningful chunks while ensuring 
    each chunk is within the token limit.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split())  # Approximate token count
        if current_length + sentence_length > max_chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(sentence)
        current_length += sentence_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def get_embeddings(text: str) -> List[float]:
    """Get embeddings using Bedrock's embedding model."""
    response = config.bedrock_runtime.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        body=json.dumps({
            "inputText": text
        })
    )
    
    response_body = json.loads(response['body'].read())
    return response_body['embedding']

def insert_data(articles):
    """
    Insert scraped data into OpenSearch.
    """
    
    progress_bar = st.progress(0)
    existing_urls = 0
    new_urls = 0
    # Process and store each chunk separately
    for i, article in enumerate(articles):

        existing_data = config.client.search(index=config.index_name, body={"query": {"match_phrase": {"url": article['url']}}})
        if existing_data['hits']['total']['value'] > 0:
            print(f"Data for {article['url']} already exists. Skipping...")
            existing_urls += 1
        else:
            new_urls += 1
            chunks = semantic_chunking(article['body'])
            source = article['url'].replace('https://', '').replace('www.', "").split('/')[0].lstrip().rstrip()

            for j, chunk in enumerate(chunks):
                embedding = get_embeddings(chunk)  # Generate embedding for the chunk
                chunk_data = {
                    "article_id": i,
                    "chunk_id": j,
                    "chunk_text": chunk,
                    "embedding": embedding,
                    "source": source,
                    "title": article['title'],
                    "url": article['url']
                }

                response = config.client.index(index=config.index_name, body=chunk_data)
                print(response)
            progress_bar.progress((i + 1) / len(articles))

    progress_bar.empty()

    st.success(f"✅ {new_urls} new URLs added to OpenSearch. Skipped {existing_urls} existing URLs.")

def scrape_pages(links):
    """Scrape multiple pages with a progress bar."""
    scraped_data = []
    progress_bar = st.progress(0)
    existing_urls = 0
    new_urls = 0
    for idx, url in enumerate(links):
        
        existing_data = config.client.search(index=config.index_name, body={"query": {"match_phrase": {"url": url}}})
        if existing_data['hits']['total']['value'] > 0:
            print(f"Data for {url} already exists. Skipping...")
            existing_urls += 1
        else:
            st.info(f"Scraping page {idx + 1}/{len(links)}: {url}")
            new_urls += 1
            scraped_page = scrape_page(url)
            if scraped_page:
                scraped_data.append(scraped_page)
        progress_bar.progress((idx + 1) / len(links))

    progress_bar.empty()
    
    st.success(f"✅ Scraped {new_urls} new URLs. Skipped {existing_urls} existing URLs.")

    return scraped_data


def sematic_search(query, source):
    """
    Perform semantic search on OpenSearch.
    """

    query_vector = get_embeddings(query)  

    query_body = {
        "size": 5,
        "query": {
            "bool": {
                "must": [
                    {"match_phrase": {"source": source}}
                ],
                "filter": {
                    "knn": {
                        "embedding": {  
                            "vector": query_vector,
                            "k": 5
                        }
                    }
                }
            }
        }
    }

    response = config.client.search(index='scraped-data-test', body=query_body)
    
    return response['hits']['hits']

def create_prompt(prompt: str):
    content = []
    
    # Add the initial prompt
    content.append({
        "text": prompt
    })

    return {
            "inferenceConfig": {
            "max_new_tokens": 1000
            },
            "messages": [
            {
                "role": "user",
                "content": content
            }
            ]
        }

def generate_response(retrieved_context, user_question) -> str:
    try:
        # Prepare the multimodal request

        prompt = f"""You are an AI assistant designed to provide accurate and context-aware responses based on the given information. Below is a set of relevant documents retrieved from a website:

                    {retrieved_context}

                    Based on this information, answer the following user query:

                    Query: {user_question}

                    If the provided context does not contain enough information to answer the question, respond with 'The available information does not contain an answer to this question.' Do not make up an answer beyond the given context."""
        # print(prompt)
        request_body = create_prompt(prompt)
        
        # Invoke Claude model
        response = config.bedrock_runtime.invoke_model(
            modelId="amazon.nova-lite-v1:0",
            contentType="application/json",
            accept="application/json",       
            body=json.dumps(request_body)
        )
        
        response_body = json.loads(response['body'].read())

        answer = response_body['output']['message']['content'][0]['text']
        
        usage = response_body['usage']

        return answer, usage
        
    except Exception as e:
        return f"Error during classification: {str(e)}"


# UI for user input
website_url = st.text_input("Enter Website URL:", key="website_url")
submit_button = st.button("Enter")

if "scraped" not in st.session_state:
    st.session_state.scraped = False

if website_url and submit_button:
    if not validators.url(website_url):
        st.error("Invalid URL. Please enter a valid website URL.")
    else:
        st.info("Fetching robots.txt for sitemap information...")
        sitemap_urls = fetch_sitemap_from_robots(website_url)

        st.info("Sitemap urls ..."+ str(", ".join(sitemap_urls)))
        
        st.info("Fetching links from sitemap(s)...")
        links = fetch_sitemap_links(sitemap_urls)

        if links:
            scraped_data = scrape_pages(links)
            
            # table_name = get_table_name(website_url)
            # save_to_postgres(scraped_data, table_name)
            
            st.info("Saving data to OpenSearch...")
            
            insert_data(scraped_data)

            st.success("✅ Data scraping completed and saved to database!")
            st.session_state.scraped = True

# Question input with dummy response after scraping
if st.session_state.scraped:
    question = st.text_input("Ask a question:", key="question_input")
    if question:
        source = website_url.lower().replace('https://', '').replace('www.', '').split('/')[0].lstrip().rstrip()
        print("source ", source)
        response = sematic_search(question, source)
        
        # print("response ", response)
        context = ""
        sources = []
        for res in response[:3]:
            # st.write(f"🤖 Answer: {res['_score']} - {res['_source']['title']} - {res['_source']['chunk_text']}")
            context += res['_source']['chunk_text'] + " \n"
            sources.append(res['_source']['url'])
        
        answer, usage = generate_response(context, question)
        st.write(f"🤖 Answer: {answer}")
        st.write(f"📚 Sources: {', '.join(sources)}")
        st.write(f"🔍 Tokens Usage: {usage}")
