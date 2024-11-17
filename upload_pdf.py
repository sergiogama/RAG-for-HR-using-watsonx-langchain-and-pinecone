import os
from dotenv import load_dotenv
import fitz  # PyMuPDF for reading PDFs
from langchain_ibm import WatsonxEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes
from langchain.docstore.document import Document

load_dotenv()
# Function to initialize Pinecone
def initialize_pinecone():
    pc = Pinecone(
        api_key=os.environ.get("PINECONE_API_KEY")
    )
    return pc

# Function to load and extract text from PDF files in the current directory
def fetch_content_from_pdfs():
    documents = []
    data = os.getcwd() + "/data"

    # List all files in the current directory with .pdf extension
    for filename in os.listdir(data):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(data, filename)
            print(f"Carregando arquivo: {filename}")

            # Extract text from PDF using PyMuPDF (fitz)
            with fitz.open(file_path) as pdf:
                text = ""
                for page in pdf:
                    text += page.get_text()

                if text:
                    # Create a Document object for each PDF file
                    document = Document(
                        page_content=text,
                        metadata={"source": filename}
                    )
                    documents.append(document)

    return documents

# Function to load data into Pinecone using Langchain and WatsonX embeddings
def upload_data_to_pinecone(index_name):
    # Initialize Pinecone
    pc = initialize_pinecone()

    # Check if index exists, otherwise create it
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region=os.environ.get('PINECONE_ENV')
            )
        )

    # Connect to index
    index = pc.Index(index_name)

    # Initialize the WatsonX embeddings model
    embeddings = WatsonxEmbeddings(
        model_id=EmbeddingTypes.IBM_SLATE_30M_ENG.value,
        url=os.environ.get("WATSONX_API_URL"),
        apikey=os.environ.get("WATSONX_API_KEY"),
        project_id=os.environ.get("WATSONX_PROJECT_ID"),
    )

    # Load documents from directory
    documents = fetch_content_from_pdfs()

    if not documents:
        print("Nenhum documento encontrado no diretório.")
        return

    # Process and format documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # Send embeddings and documents to Pinecone
    for i, doc in enumerate(docs):
        print(f"Enviando documento {i+1} para o índice {index_name}...")
        embedding = embeddings.embed_query(doc.page_content)
        index.upsert(vectors=[(
            str(i),
            embedding,
            {
                "source": doc.metadata["source"],
                "text": doc.page_content  # Add the content to the metadata
            }
        )])

    print(f"Enviados {len(docs)} documentos para o índice {index_name}.")

# Main function
if __name__ == "__main__":
    upload_data_to_pinecone(os.environ.get("PINECONE_INDEX_NAME"))
