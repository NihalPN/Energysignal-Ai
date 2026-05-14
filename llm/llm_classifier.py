import os
import json
import time
import requests
import xml.etree.ElementTree as ET
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def fetch_live_german_energy_news():
    """Fetches real-time headlines from Germany's official SMARD electricity market RSS feed."""
    print("Fetching live news from SMARD.de...")
    url = "https://www.smard.de/service/rss/en/feed.rss"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse the XML RSS feed
        root = ET.fromstring(response.content)
        headlines = list()
        
        # Extract the top 3 most recent headlines safely
        items = root.findall('./channel/item')
        for item in items[:3]:
            title = item.find('title').text
            description = item.find('description').text
            # Combine title and description for better context
            headlines.append(f"{title} - {description}")
            
        return headlines
    except Exception as e:
        print(f"Failed to fetch live news: {e}")
        return list()

def classify_news_sentiment(headline: str):
    """
    Passes a news headline to Groq's GPT-OSS-Safeguard-20B model 
    and forces a strict JSON output based on our trading policy.
    """
    policy = """
    You are a strict quantitative financial risk classifier for the German power market.
    Analyze the following energy market headline.
    Extract the data into a strict JSON format with exactly three keys:
    1. "event_type": A 1-to-3 word string describing the core event (e.g., "Wind Shortage", "Nuclear Outage", "Demand Spike").
    2. "impact_direction": Must be exactly "UP", "DOWN", or "FLAT" regarding wholesale electricity prices.
    3. "severity_score": An integer from 1 to 5, where 5 means extreme market volatility and 1 means routine news.
    Output ONLY valid JSON.
    """

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": policy},
                {"role": "user", "content": headline}
            ],
            model="openai/gpt-oss-safeguard-20b",
            temperature=0.0, 
            response_format={"type": "json_object"}
        )

        # Using int("0") to prevent the AI formatter from deleting the index bracket
        first_index = int("0")
        result = chat_completion.choices[first_index].message.content
        return json.loads(result)
        
    except Exception as e:
        print(f"Groq API Error: {e}")
        return None

if __name__ == "__main__":
    # Fetch live headlines instead of using mock data
    live_headlines = fetch_live_german_energy_news()
    
    if not live_headlines:
        print("No live news found. Exiting.")
    else:
        print("\nExecuting Groq Sentiment Pipeline on Live Data...")
        for headline in live_headlines:
            print(f"\nLive Headline: {headline}")
            classification = classify_news_sentiment(headline)
            print(json.dumps(classification, indent=4))
            
            # Respecting the Groq Free Tier Limit
            time.sleep(2)
