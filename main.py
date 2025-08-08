import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header, Depends, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import logging

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.cohere import CohereEmbedding

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Check if the API keys are set
if os.getenv("GROQ_API_KEY") is None:
    raise ValueError("GROQ_API_KEY is not set in the environment.")

if os.getenv("COHERE_API_KEY") is None:
    raise ValueError("COHERE_API_KEY is not set in the environment.")

# Initialize the FastAPI app
app = FastAPI(
    title="RAG Document Q&A API",
    description="A 100% FREE RAG-based API using Groq + Cohere embeddings",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define API Request/Response Models
class HackathonRequest(BaseModel):
    documents: str  # URL to the PDF document
    questions: List[str]

class HackathonResponse(BaseModel):
    answers: List[str]

# Configure global settings for LlamaIndex
def configure_llama_index():
    """Configure LlamaIndex with 100% FREE models"""
    try:
        # Initialize Groq LLM with Llama 3.1 8B (FREE)
        llm = Groq(
            model="llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.1,  # Lower temperature for more consistent answers
            max_tokens=1024
        )
        
        # Initialize Cohere Embedding model (FREE TIER)
        embed_model = CohereEmbedding(
            api_key=os.getenv("COHERE_API_KEY"),
            model_name="embed-english-light-v3.0",  # Free tier model
            input_type="search_document"  # Optimized for RAG
        )
        
        # Set global settings
        Settings.llm = llm
        Settings.embed_model = embed_model
        Settings.chunk_size = 1024
        Settings.chunk_overlap = 200
        Settings.embed_batch_size = 50
        
        logger.info("LlamaIndex configured successfully with 100% FREE models")
        return llm
    except Exception as e:
        logger.error(f"Failed to configure LlamaIndex: {e}")
        raise

# Configure on startup
llm = configure_llama_index()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "RAG Document Q&A API is running 100% FREE with Groq + Cohere", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "groq_api_key_configured": bool(os.getenv("GROQ_API_KEY")),
        "cohere_api_key_configured": bool(os.getenv("COHERE_API_KEY")),
        "llm_model": "llama-3.1-8b-instant",
        "embedding_model": "embed-english-light-v3.0",
        "provider": "Groq + Cohere",
        "cost": "100% FREE"
    }

def validate_token(authorization: str = Header(...)):
    expected_token = f"Bearer {os.getenv('API_AUTH_TOKEN')}"
    print(expected_token, authorization)
    if authorization != expected_token:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
@app.post("/api/v1/hackrx/run", response_model=HackathonResponse)
async def run_submission(
    request: Request,
    token: None = Depends(validate_token)
):
    """
    Main RAG endpoint that processes documents and answers questions.
    100% FREE using Groq + Cohere embeddings.
    Supports both JSON and form data formats.
    """
    try:
        content_type = request.headers.get("content-type", "")
        
        # Handle JSON request (new format)
        if "application/json" in content_type:
            data = await request.json()
            questions = data.get("questions", [])
            # Map 'documents' field to document_url for compatibility
            document_url = data.get("documents") or data.get("document_url")
            file = None
            
        # Handle form data (existing format)
        elif "multipart/form-data" in content_type:
            form = await request.form()
            
            # Handle questions from form
            if "questions" in form:
                questions_raw = form["questions"]
                try:
                    import json
                    questions = json.loads(questions_raw) if isinstance(questions_raw, str) else [questions_raw]
                except:
                    questions = [questions_raw]
            else:
                questions = []
                for key, value in form.items():
                    if key.startswith("question"):
                        questions.append(value)
            
            document_url = form.get("document_url")
            file = form.get("file")
            
        else:
            raise HTTPException(status_code=400, detail="Content-Type must be application/json or multipart/form-data")

        # Validate input
        if not questions or not isinstance(questions, list):
            raise HTTPException(status_code=400, detail="Questions must be provided as a non-empty list")

        logger.info(f"Number of questions received: {len(questions)}")
        documents = []

        # 1. Load from uploaded file (for blob-based or local PDF uploads)
        if file is not None:
            logger.info(f"Processing uploaded file: {file.filename}")
            try:
                import tempfile

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(await file.read())
                    tmp_file_path = tmp_file.name

                reader = SimpleDirectoryReader(input_files=[tmp_file_path])
                documents = reader.load_data()

                os.unlink(tmp_file_path)
                logger.info(f"Successfully loaded {len(documents)} document chunks from uploaded file")

            except Exception as e:
                logger.error(f"Failed to read uploaded file: {e}")
                raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {str(e)}")

        # 2. Load from URL if no file was uploaded
        elif document_url and document_url.startswith(("http://", "https://")):
            logger.info(f"Processing document from URL: {document_url}")
            try:
                import requests
                import tempfile

                response = requests.get(document_url)
                response.raise_for_status()

                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(response.content)
                    tmp_file_path = tmp_file.name

                reader = SimpleDirectoryReader(input_files=[tmp_file_path])
                documents = reader.load_data()

                os.unlink(tmp_file_path)
                logger.info(f"Successfully loaded {len(documents)} document chunks from URL")

            except Exception as e:
                logger.error(f"Failed to load document from URL: {e}")
                raise HTTPException(status_code=400, detail=f"Failed to load document from URL: {str(e)}")
        else:
            raise HTTPException(
                status_code=400,
                detail="No valid document provided. Provide either a file upload or a valid document URL."
            )

        # 3. Create the Vector Index
        try:
            index = VectorStoreIndex.from_documents(documents)
            logger.info("Successfully created vector index")
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to create document index: {str(e)}")

        # 4. Create the Query Engine
        query_engine = index.as_query_engine(similarity_top_k=3, response_mode="compact")
        answers_with_sources = []

        # 5. Process each question
        for i, question in enumerate(questions):
            try:
                logger.info(f"Processing question {i+1}/{len(questions)}: {question[:100]}...")
                response = query_engine.query(question)
                answer_text = str(response)
                answers_with_sources.append(answer_text)
                logger.info(f"Successfully processed question {i+1}")

            except Exception as e:
                logger.error(f"Failed to process question {i+1}: {e}")
                answers_with_sources.append(f"Error processing question: {str(e)}")

        logger.info(f"Successfully processed all {len(questions)} questions")
        return HackathonResponse(answers=answers_with_sources)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in run_submission: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
@app.post("/api/v1/test")
async def test_endpoint(request: dict):
    """Test endpoint for debugging"""
    return {
        "received_request": request,
        "groq_api_key_present": bool(os.getenv("GROQ_API_KEY")),
        "cohere_api_key_present": bool(os.getenv("COHERE_API_KEY")),
        "llm_model": "llama-3.1-8b-instant",
        "embedding_model": "embed-english-light-v3.0",
        "status": "test_successful",
        "cost": "100% FREE"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
