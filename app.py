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
WATSONX_PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID")
WATSONX_API_KEY = os.environ.get("WATSONX_API_KEY")
WATSONX_API_URL = os.environ.get("WATSONX_API_URL")
WATSONX_ACCESS_TOKEN = ""

# Global list to store conversation history
conversation_memory = []
user_query = ""

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

def format_conversation_history():
    """Format the conversation history into the desired context format."""
    context = "\n\nMemory of conversation to be used as context in the same topic, otherwise must be ignored:\n\n"
    for entry in conversation_memory:
        context += f"<|user|>{entry['question']}\n"
        context += f"<|assistant|>{entry['response']}\n"
    return context

# Function to call the WatsonX LLM
def call_watsonx_llm(query):
    """Call the WatsonX API to generate a response."""
    url = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2024-03-14"

    # Generate prompt with conversation history as context
    conversation_context = format_conversation_history()

    prompt = """
    General Guidelines:
    You are a chatbot for employees in a company for help people with HR information, such as vacation, salary which is about payment, dress code, remote work and benefities.
    You always answer the questions with markdown formatting using GitHub syntax. The markdown formatting you support: headings, bold, italic, links, tables, lists, code blocks, and blockquotes. You must omit that you answer the questions with markdown.
    Any HTML tags must be wrapped in block quotes, for example ```<html>```. You will be penalized for not rendering code in block quotes.
    When returning code blocks, specify language.
    You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. 
    Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
    You are a professional Human Resources Specialist and will respond to user inquiries with clarity, politeness, and helpfulness based on the provided documents. You will avoid mentioning the documents or the company name in your responses.
    The user can change the topic at any time. In this case, you should adapt to the new topic and provide relevant information.
    You must have extremely consise and precise in your responses, once the response is to be showed in a chatbot.
    Please do not add information about what kind os text is ir title, such as Answer:, or if thre is part of previous conversation. So please give only the infroamtion requested.

    Remove any part that is not a plain text, such as images, tables, or code snippets, or any other non-text content.
    """

    prompt = f"<|system|>{prompt}<|user|>{query}" #<|context|>{conversation_context}"

    body = {
        "input": prompt, #f"<|system|>{prompt}<|user|>{query}<|context|>{context}",
        "parameters": {
            "decoding_method": "sample",
            "max_new_tokens": 300,
            "min_new_tokens": 0,
            "stop_sequences": [],
            "temperature": 0.3,
            "top_k": 5,
            "top_p": 1,
            "repetition_penalty": 1
	    },
        #"parameters": {"decoding_method": "greedy", "max_new_tokens": 900},
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
    #print("data: ", data)
 
    # Extract response
    assistant_response = data['results'][0].get('generated_text', '') if 'results' in data else ''

    # Remove dirty strings from the response
    assistant_response = assistant_response.replace("<|assistant|>", "").replace("<|user|>", "")
    assistant_response = assistant_response.replace("<|assistant<|end_header_id|>>", "")

    print("Assistant response: ", assistant_response)
    
    # Store the conversation in memory
    conversation_memory.append({"question": user_query, "response": assistant_response})
    
    # Keep only the last 10 conversations
    if len(conversation_memory) > 10:
        conversation_memory.pop(0)
    
    return assistant_response

# Configuration of the LLM model for LangChain
class WatsonXLLM(BaseLLM):
    api_key: str = Field(default_factory=lambda: WATSONX_API_KEY)
    project_id: str = Field(default_factory=lambda: WATSONX_PROJECT_ID)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        return call_watsonx_llm(prompt)

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
            
            # Store the user query in a global variable
            global user_query
            user_query = query

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
            return jsonify({"response": response_text})

        except Exception as e:
            print("Exception occurred:", str(e))
            return jsonify({"error": str(e)}), 500
        
if __name__ == "__main__":  
    app.run(host='0.0.0.0', port=8000)
