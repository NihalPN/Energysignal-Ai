import lancedb
import pandas as pd
import sqlite3
import os
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Setup Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, 'database', 'energy_market.db')
LANCE_DB_PATH = os.path.join(BASE_DIR, 'database', 'rag_store')

# Load a highly efficient, CPU-friendly embedding model
print("Loading Embedding Model (This may take a moment on the first run)...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def build_rag_database():
    """Extracts extreme market events from SQLite and embeds them into LanceDB."""
    print("Extracting historical market extremes from SQLite...")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM master_features", conn, parse_dates=['timestamp'])
    conn.close()

    # Filter for severe market anomalies (Price > 150 EUR or Negative Prices)
    anomalies = df[(df['price_eur_mwh'] > 150) | (df['price_eur_mwh'] < 0)].copy()
    print(f"Found {len(anomalies)} severe market events to index.")

    records = list()
    for _, row in anomalies.iterrows():
        # THE FIX: Calculate the load on the fly since we have the residual and renewable data
        actual_load = row['total_renewable'] + row['residual_load']
        
        # Create a clean text representation of the math for the LLM
        context = (f"On {row['timestamp']}, the German electricity price hit {row['price_eur_mwh']:.2f} EUR/MWh. "
                   f"Total renewable generation was {row['total_renewable']:.2f} MW against a grid load of {actual_load:.2f} MW. "
                   f"The overall renewable penetration was {row['renewable_penetration']*100:.1f}%.")     
        vector = embedder.encode(context).tolist()
        
        records.append({
            "vector": vector,
            "timestamp": str(row['timestamp']),
            "price": float(row['price_eur_mwh']),
            "text_context": context
        })

    print("Building LanceDB Vector Store on disk...")
    db = lancedb.connect(LANCE_DB_PATH)
    
    # Reset table if it already exists during testing
    if "market_events" in db.table_names():
        db.drop_table("market_events")
        
    db.create_table("market_events", data=records)
    print("RAG Database successfully built!")

def analyze_market_condition(current_condition: str):
    """Retrieves similar past events from LanceDB and prompts Groq for an analysis."""
    db = lancedb.connect(LANCE_DB_PATH)
    table = db.open_table("market_events")
    
    print("\nSearching vector database for similar historical conditions...")
    query_vector = embedder.encode(current_condition).tolist()
    
    # Retrieve the top 3 most mathematically similar historical events
    results = table.search(query_vector).limit(3).to_pandas()
    historical_evidence = "\n".join(results['text_context'].tolist())
    
    prompt = f"""
    You are a Senior Quantitative Energy Trader. 
    A junior trader has presented you with the following current market condition:
    "{current_condition}"
    
    Our vector database retrieved these mathematically similar historical events for reference:
    {historical_evidence}
    
    Based ONLY on this historical evidence, explain what is likely to happen to the electricity price today and why. Do not use pleasantries.
    """
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="openai/gpt-oss-120b",
            temperature=0.2
        )
        
        # Using int("0") to prevent formatting bugs
        first_index = int("0")
        return chat_completion.choices[first_index].message.content
    except Exception as e:
        return f"Groq API Error: {e}"

if __name__ == "__main__":
    # 1. Build the database (This will embed the historical data into LanceDB)
    build_rag_database()
    
    # 2. Test the RAG system with a hypothetical market shock
    test_scenario = "Meteorological models indicate a sudden, massive drop in offshore wind generation for tomorrow morning, while industrial grid load remains extremely high."
    
    print(f"\nSimulated Market Event: {test_scenario}")
    analysis = analyze_market_condition(test_scenario)
    
    print("\n--- AI TRADING DESK ANALYSIS ---")
    print(analysis)
    print("--------------------------------")
