import os
import requests
from dotenv import load_dotenv
from flask import Flask, request, Response, jsonify
from flask_restx import Api, Resource, fields
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from typing import List, Optional, Any
from pydantic import Field
from langchain.llms.base import BaseLLM
from langchain.schema import LLMResult, Generation
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes
from langchain_ibm import WatsonxEmbeddings

# Load environment variables from the .env file
load_dotenv()

# Configuration of environment variables
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENV")
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")
WATSONX_ACCESS_TOKEN = os.environ.get("WATSONX_ACCESS_TOKEN")
WATSONX_PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID")
WATSONX_API_KEY = os.environ.get("WATSONX_API_KEY")
WATSONX_API_URL = os.environ.get("WATSONX_API_URL")

# Initialize Pinecone
pc = Pinecone(
    api_key=PINECONE_API_KEY
)

# Connect to the Pinecone index
index = pc.Index(INDEX_NAME)

# Initialize WatsonX embeddings
embeddings = WatsonxEmbeddings(
    model_id=EmbeddingTypes.IBM_SLATE_30M_ENG.value,
    url=WATSONX_API_URL,
    apikey=WATSONX_API_KEY,
    project_id=WATSONX_PROJECT_ID,
)

# Initialize the PineconeVectorStore
vectorstore = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings
)

# Initialize Flask and Flask-RESTx
app = Flask(__name__)
api = Api(app, version='1.0', title='Duda RAG API',
          description='API of RAG using Pinecone and WatsonX, orchestrated by LangChain',
          doc='/docs')

ns = api.namespace('api', description='Chatbot operations')

# Input model for Flask-RESTx
chat_model = api.model('ChatQuery', {
    'query': fields.String(required=True, description='Prompt from user')
})

# Function to call the WatsonX LLM
def call_watsonx_llm(query, context):
    """Call the WatsonX API to generate a response."""
    url = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"
    
    body = {
        "input": f"<|user|>{query}<|context|>{context}",
        "parameters": {"decoding_method": "greedy", "max_new_tokens": 900},
        "model_id": "meta-llama/llama-3-70b-instruct",
        "project_id": WATSONX_PROJECT_ID
    }
    
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {WATSONX_ACCESS_TOKEN}"
    }
    
    response = requests.post(url, headers=headers, json=body)
    
    data = response.json()
    return data['results'][0].get('generated_text', '') if 'results' in data else ''

# Configuration of the LLM model for LangChain
class WatsonXLLM(BaseLLM):
    api_key: str = Field(default_factory=lambda: WATSONX_API_KEY)
    project_id: str = Field(default_factory=lambda: WATSONX_PROJECT_ID)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        return call_watsonx_llm(prompt, "")

    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
        generations = [Generation(text=self._call(prompt)) for prompt in prompts]
        return LLMResult(generations=[generations])

    @property
    def _llm_type(self):
        return "watsonx"

# Function to configure the RetrievalQA
def create_qa_chain():
    # Create the retriever from Pinecone
    retriever_vector = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # Configure WatsonX as LLM
    llm = WatsonXLLM()

    # Create the RetrievalQA chain
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever_vector,
        return_source_documents=True
    )
    
# API route for the chatbot
@ns.route('/chat')
class Chatbot(Resource):
    @ns.expect(chat_model)
    def post(self):
        """Receive a query and return a response using RAG."""
        try:
            data = request.json
            if not data:
                print("No data received")
                return {"error": "No JSON data received"}, 400

            query = data.get("query")
            if not query:
                print("No query found in the data")
                return {"error": "Query is required"}, 400

            # Create the QA chain
            print("Creating the QA chain...")
            qa_chain = create_qa_chain()

            # Generate response using the RetrievalQA chain
            print(f"Generating response for query: {query}")
            result = qa_chain.invoke(query)

            # Extracting the response
            if "result" not in result:
                print("No result found in response")
                return {"error": "No result found"}, 500

            response_text = result.get("result", "No result found")
            print("Response:", response_text)
            return jsonify({"response": response_text})

        except Exception as e:
            print("Exception occurred:", str(e))
            return jsonify({"error": str(e)}), 500
        
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)