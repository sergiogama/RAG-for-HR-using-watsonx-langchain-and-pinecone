# 📚 RAG Solution for IBM Hackathon: Unlocking Knowledge with WatsonX.ai & Pinecone

## 🏆 Objective
This project was developed for a global hackathon organized by IBM, aimed at promoting the adoption of **WatsonX.ai** and **WatsonX Assistant**. The solution addresses a common challenge within companies: providing employees with clear information on vacation policies and regulations.

The goal is to leverage a **Retrieval-Augmented Generation (RAG)** model to efficiently answer questions related to vacation rules using company documents. The solution uses **Pinecone** as a vector database, **WatsonX.ai** for the Large Language Model (LLM) using LLama, and **LangChain** as the orchestrator.

---

## 🛠️ Proposed Solution
1. **Data Ingestion**: The solution starts by uploading company policy PDFs into Pinecone using Python. The PDFs are split into chunks, embedded using **WatsonX Embeddings**, and stored in the Pinecone vector database.
2. **RAG API**: A Python API built with **Flask** and **Flask-RESTx** handles incoming queries, retrieves relevant documents from Pinecone, and uses WatsonX.ai's LLM to generate contextually accurate responses.
3. **Chatbot Interface**: The API integrates with **WatsonX Assistant V2** using Actions, providing an interactive web interface for users to ask questions and receive answers in real time.

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
5. A **WatsonX Assistant V2** instance.

---

## 🛠️ Installation and Setup

### 1. Clone the Repository
```bash
git clone https://github.com/sergiogama/RAG-for-HR-using-watsonx-langchain-and-pinecone.git
cd ibm-rag-solution
```

### 2. Create a `.env` File
Copy the sample environment file and adjust it with your credentials:
```bash
cp sample.env .env
```

Edit the `.env` file:
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
```bash
pip install -r requirements.txt
```

### 4. Load PDFs into Pinecone
```bash
python upload_pdf.py
```

---

## 🚀 Running the API
```bash
python app.py
```

The API will be available at `http://localhost:8000`.

---

## 🤖 Integrating with WatsonX Assistant V2

### Step 1: Generate API Key and Project ID
1. Log in to [WatsonX.ai](https://watsonx.ai) and create an API key.
2. Find your **Project ID** under `Projects -> Manage -> General -> Details`.

### Step 2: Download the OpenAPI Specification
- Ensure the `openapi.json` file is up to date:
  ```bash
  curl http://localhost:8000/swagger.json -o watsonx-openapi.json
  ```
  Obs: You can use and test the file part of this repository

### Step 3: Create a WatsonX Assistant
1. Log in to **WatsonX Assistant**.
2. Create a new assistant.

### Step 4: Add a Custom Extension
1. Go to the **Integrations** tab of your assistant.
2. Click on **Build custom extension**.
3. Use the downloaded `openapi.json` file to create a custom extension named `RAG HR`.

### Step 5: Upload WatsonX Assistant sample 
1. Go to **Assistant settings -> Download/upload files**.
2. Upload the `watsonx-actions.zip` file (included in this repository).

### Step 6: Configure action RAG HR search to use the extension 
1. Go to **Step 3 -> Edit extension**.
2. Configure the extesnsion and set the parameter, query to query_text.   

### Step 7: Test the Assistant
- Use the **Preview chat** feature to test the assistant.
- If the actions do not work initially, refresh the chat and re-upload the actions.

---

## 📄 Usage

### Endpoint: `/api/chat`
- **Method**: `POST`
- **Payload**:
  ```json
  {
    "query": "How do I apply for vacation?"
  }
  ```
- **Response**:
  ```json
  {
    "response": "You can apply for vacation by filling out the online request form available on the HR portal."
  }
  ```

### Example Request
```bash
curl -X POST http://localhost:8000/api/chat -H "Content-Type: application/json" -d '{"query": "What is WatsonX?"}'
```

---

## 🗂️ Project Structure
```
ibm-rag-solution/
├── app.py                  # Main Flask API
├── upload_pdf.py           # Script to load PDFs into Pinecone
├── watsonx-openapi.json    # OpenAPI specification for WatsonX Assistant
├── watsonx-actions.json    # Actions configuration for WatsonX Assistant V2
├── requirements.txt        # Python dependencies
├── sample.env              # Sample environment variables file
├── .env                    # Environment variables
└── data/                   # Dataset in PDF files to be uploaded to Pinecone
```

---

## 🛠️ Troubleshooting
- Ensure all API keys and environment variables are set correctly.
- Verify Pinecone and WatsonX services are accessible.
- Use `curl` and `Postman` to test the API endpoints.

---

## 🔗 Useful Links
- [Pinecone Documentation](https://docs.pinecone.io/)
- [WatsonX.ai Documentation](https://cloud.ibm.com/docs/watsonx)
- [WatsonX Assistant Documentation](https://cloud.ibm.com/docs/watson-assistant)
- [LangChain Documentation](https://python.langchain.com/)

---

## 📢 Contributing
We welcome contributions! Open issues or submit pull requests.

---

## 📄 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## ✨ Acknowledgments
Special thanks to IBM for organizing this hackathon.

---

Good luck with the hackathon, and may your solution stand out! 🚀
```

### Explanation of Changes
1. **Added a detailed section** for integrating with **WatsonX Assistant V2** using Actions and custom extensions.
2. **Updated the project structure** to include the necessary files (`watsonx-openapi.json` and `watsonx-actions.json`).
3. **Included configuration steps** for authentication and setting up session variables.

Let me know if you need any further customization or adjustments! 😊
