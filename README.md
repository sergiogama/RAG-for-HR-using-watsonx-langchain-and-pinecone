# 📚 RAG Solution for IBM Hackathon: Unlocking Knowledge with WatsonX.ai & Pinecone

## 🏆 Objective
This project was developed for a global hackathon organized by IBM, aimed at promoting the adoption of **WatsonX.ai** and **WatsonX Assistant**. The solution addresses a common challenge within companies: providing employees with clear information on vacation policies and regulations. 

The goal is to leverage a **Retrieval-Augmented Generation (RAG)** model to efficiently answer questions related to vacation rules using company documents. The solution uses **Pinecone** as a vector database, **WatsonX.ai** for the Large Language Model (LLM) using LLama, and **LangChain** as the orchestrator. 

---

## 🛠️ Proposed Solution
1. **Data Ingestion**: The solution starts by uploading company policy PDFs into Pinecone using Python. The PDFs are split into chunks, embedded using **WatsonX Embeddings**, and stored in the Pinecone vector database.
2. **RAG API**: A Python API built with **Flask** and **Flask-RESTx** handles incoming queries, retrieves relevant documents from Pinecone, and uses WatsonX.ai's LLM to generate contextually accurate responses.
3. **Chatbot Interface**: The API integrates with **WatsonX Assistant**, which provides an interactive web interface for users to ask questions and receive answers in real time.

---

## 🖥️ Technologies Used
- [Python](https://www.python.org/)
- [Flask](https://flask.palletsprojects.com/)
- [Flask-RESTx](https://flask-restx.readthedocs.io/)
- [Pinecone](https://www.pinecone.io/)
- [WatsonX.ai](https://www.ibm.com/watsonx)
- [LangChain](https://python.langchain.com/)
- [Typing](https://docs.python.org/3/library/typing.html)
- [IBM Watson AI](https://cloud.ibm.com/docs/watsonx)

---

## ⚙️ Prerequisites
Before running the project, ensure you have the following:

1. **Python 3.10+** installed.
2. The following Python packages:
   - `requests`
   - `flask`
   - `flask-restx`
   - `python-dotenv`
   - `pydantic`
   - `fitz` (PyMuPDF for PDF processing)
   - `langchain`
   - `langchain_pinecone`
   - `langchain_community`
   - `ibm_watsonx_ai`
3. A **Pinecone** account with an API key.
4. Access to **WatsonX.ai** API.

---

## 🛠️ Installation and Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/ibm-rag-solution.git
cd ibm-rag-solution
```

### 2. Create a `.env` File
There is a sample file provided named `sample.env`. Copy it and adjust it with your credentials:
```bash
cp sample.env .env
```

Edit the `.env` file with your Pinecone and WatsonX API keys:
```
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENV=us-east1-gcp
INDEX_NAME=vacation
WATSONX_ACCESS_TOKEN=your_watsonx_access_token
WATSONX_PROJECT_ID=your_watsonx_project_id
WATSONX_API_KEY=your_watsonx_api_key
WATSONX_API_URL=https://us-south.ml.cloud.ibm.com
```

### 3. Install Dependencies
Ensure all required packages are installed:
```bash
pip install -r requirements.txt
```

### 4. Load PDFs into Pinecone
Before running the API, you need to upload your PDFs into Pinecone:

1. Copy your PDF files into the `./data` folder.
2. Run the following script to load the PDFs into Pinecone:
   ```bash
   python upload_pdf.py
   ```

This script will extract text from the PDFs, split them into chunks, generate embeddings using WatsonX, and upload them to Pinecone.

---

## 🚀 Running the API
To start the Flask API, run:

```bash
python app.py
```

The API will be available at `http://localhost:8000`.

### Access the Swagger Documentation
Visit `http://localhost:8000/docs` to see the automatically generated API documentation.

To download the OpenAPI JSON file:
```bash
curl http://localhost:8000/swagger.json | jq . > openapi.json
```

---

## 📄 Usage

### Endpoint: `/api/chat`
- **Method**: `POST`
- **Payload**:
  ```json
  {
    "query": "What are the main parts of watsonx.ai?"
  }
  ```
- **Response**:
  ```json
  {
    "response": "WatsonX.ai consists of three main components: data management, AI model training, and deployment."
  }
  ```

### Example Request
Use `curl` to test the endpoint:
```bash
curl -X POST http://localhost:8000/api/chat -H "Content-Type: application/json" -d '{"query": "How do I apply for vacation?"}'
```

---

## 🗂️ Project Structure
```
ibm-rag-solution/
├── app.py                  # Main Flask API
├── upload_pdf.py           # Script to load PDFs into Pinecone
├── requirements.txt        # Python dependencies
├── sample.env              # Sample environment variables file
├── .env                    # Environment variables
└── README.md               # Project documentation
```

---

## 🤖 WatsonX Assistant Integration
This API is designed to be integrated with **WatsonX Assistant**. Simply configure your WatsonX Assistant to make requests to the `/api/chat` endpoint to provide an interactive user experience.

---

## 🛠️ Troubleshooting
If you encounter any issues:

- Ensure that all API keys and environment variables are correctly set.
- Check if Pinecone and WatsonX services are accessible from your network.
- Use tools like `curl` and `Postman` to test the API endpoints.

---

## 🔗 Useful Links
- [Pinecone Documentation](https://docs.pinecone.io/)
- [WatsonX.ai Documentation](https://cloud.ibm.com/docs/watsonx)
- [LangChain Documentation](https://python.langchain.com/)
- [Flask-RESTx Documentation](https://flask-restx.readthedocs.io/)

---

## 📢 Contributing
We welcome contributions! Feel free to open issues or submit pull requests with improvements.

---

## 📄 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## ✨ Acknowledgments
Special thanks to IBM for organizing this hackathon and promoting the adoption of WatsonX.ai.

---

Good luck with the hackathon, and may your solution stand out! 🚀