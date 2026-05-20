import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

# 1. 100% Free Cloud Embeddings (Math done on Hugging Face servers)
hf_token = os.environ.get("HF_TOKEN")
embeddings_model = HuggingFaceInferenceAPIEmbeddings(
    api_key=hf_token, model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. 100% Free Cloud Generation (Logic done on Groq servers)
groq_api_key = os.environ.get("GROQ_API_KEY")
llm = ChatGroq(temperature=0.2, model_name="openai/gpt-oss-120b", api_key=groq_api_key)


def analyze_market_condition(scenario_text, vector_db=None):
    """
    Cloud-native analyst using free serverless APIs.
    """
    try:
        context = ""
        if vector_db:
            # Vector search happens in the cloud
            docs = vector_db.similarity_search(scenario_text, k=3)
            context = "\n".join([doc.page_content for doc in docs])

        prompt = f"""
        You are an expert European energy market quantitative analyst.
        Based on the current grid physics and historical context, provide a concise trading signal.
        
        Historical Context: {context}
        Current Scenario: {scenario_text}
        """

        response = llm.invoke(prompt)
        return response.content

    except Exception as e:
        return f"⚠️ Cloud AI Analyst Error: {str(e)}"
