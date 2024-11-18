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
from collections import deque

# Load environment variables from the .env file
load_dotenv()

# Configuration of environment variables
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENV")
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")
WATSONX_PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID")
WATSONX_API_KEY = os.environ.get("WATSONX_API_KEY")
WATSONX_API_URL = os.environ.get("WATSONX_API_URL")
WATSONX_ACCESS_TOKEN = ""

def get_iam_token(api_key):
    """Obtém um token de acesso usando a API Key da IBM Cloud."""
    url = "https://iam.cloud.ibm.com/identity/token"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
    }
    data = {
        "apikey": api_key,
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey"
    }
    response = requests.post(url, headers=headers, data=data)
    if response.status_code != 200:
        raise Exception(f"Failed to obtain IAM token: {response.status_code} - {response.text}")
    
    return response.json()["access_token"]

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
    url = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2024-03-14"

    prompt = """
    You are a professional Human Resources Specialist and will respond to user inquiries with clarity, politeness, and helpfulness based on the provided documents. You will avoid mentioning the documents or the company name in your responses.

    General Guidelines:
    Always respond in a professional, respectful, and unbiased manner.
    Ensure your answers are free from harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Responses should always promote a positive and inclusive tone.
    If a question is unclear, nonsensical, or factually incoherent, provide an explanation instead of an inaccurate or incorrect response.
    If you do not know the answer, explicitly acknowledge it without sharing false or misleading information.

    Vacation Requests:
    Always ask for the user's name and save it as an entity.
    If the user expresses interest in vacation, ask for the desired period. Vacations cannot exceed 30 days, so ensure the user specifies the start and end dates.
    Additionally, request the user's email address to finalize the vacation request. Without the name, period, and email, the vacation request cannot be processed. Politely insist on obtaining this information.
    Once all required information is provided:
    Confirm the details with the user.
    Format the gathered data into JSON with the following keys: date_start, date_end, name, and email.
    Upon confirmation, thank the user and assure them of your availability for future assistance.

    Meeting, Event, or Conference Scheduling:
    If the user requests to schedule a meeting, event, or conference, collect the following mandatory information:
    Date
    Hour
    Number of attendees
    Additionally, ask for the email address of the user and offer the option to specify a location (optional).
    Ensure the user provides all required details to proceed. Without the mandatory data (date, hour, email, name, and number of attendees), the request cannot be processed. Politely insist on completing the information.
    Once the necessary information is collected:
    Confirm the details with the user.
    Format the gathered data into JSON with the following keys: date, hour, email, name, and location.
    Upon confirmation, thank the user and assure them of your availability for future assistance.

    Data Formatting:
    Always present gathered information in JSON format with proper naming conventions:
    Vacation Requests: date_start, date_end, name, email
    Meeting Scheduling: date, hour, email, name, location
    """

    body = {
        "input": f"<;|system|>;{prompt}<;|user|>;{query}<;|context|>;{context}",
        "parameters": {"decoding_method": "greedy", "max_new_tokens": 900},
        "model_id": "meta-llama/llama-3-70b-instruct",
        "project_id": WATSONX_PROJECT_ID
    }
    
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": "Bearer " + os.environ.get('WATSONX_ACCESS_TOKEN')
    }
    
    response = requests.post(url, headers=headers, json=body)
    
    data = response.json()
    print("data: ", data)
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
        return_source_documents=True,
        context=context
    )
    
# API route for the chatbot
@ns.route('/chat')
class Chatbot(Resource):
    context = deque(maxlen=5)  # Store up to 5 previous interactions
    @ns.expect(chat_model)
    def post(self):
        """Receive a query and return a response using RAG."""
        try:
            # Obter o token de acesso
            access_token = get_iam_token(WATSONX_API_KEY)
            os.environ['WATSONX_ACCESS_TOKEN'] = access_token

            data = request.json
            if not data:
                print("No data received")
                return {"error": "No JSON data received"}, 400

            query = data.get("query")
            if not query:
                print("No query found in the data")
                return {"error": "Query is required"}, 400

            # Add the current query to the context
            self.context.append(query)

            # Create the QA chain
            print("Creating the QA chain...")
            qa_chain = create_qa_chain()

            # Generate response using the RetrievalQA chain
            print(f"Generating response for query: {query}")
            result = qa_chain.invoke(query, context=self.context)

            # Extracting the response
            if "result" not in result:
                print("No result found in response")
                return {"error": "No result found"}, 500
  
            response_text = result.get("result", "No result found")
            return jsonify({"response": response_text})

        except Exception as e:
            print("Exception occurred:", str(e))
            return jsonify({"error": str(e)}), 500
        
if __name__ == "__main__":  
    app.run(host='0.0.0.0', port=8000)
