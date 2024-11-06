from fastapi import FastAPI, File, UploadFile, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
import asyncio
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
from prompts import create_prompt
from vector_db import create_vector_db, load_local_db
from load_data import load_split_pdf_file, load_split_html_file, initialize_splitter
from utils import read_file

app = FastAPI(
    title="FastAPI Server for RAG Systems",
    description="Retrieval Augmented Generation APP which lets users upload a file and get answers to questions using LLMs",
    version="1.0.0",
)

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="RAG Fast-API Server",
        version="1.0.0",
        description="Retrieval Augmented Generation APP which lets users upload a file and get answers to questions using LLMs. "
                    "This API allows you to upload documents (PDF, DOC, DOCX, TXT), query them, and get AI-generated answers based on the document content.",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Initialize the Inference Client
client = InferenceClient(api_key="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")

# Initialize SentenceTransformer for embeddings
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

text_splitter = initialize_splitter(chunk_size=1000, chunk_overlap=100)

@app.post("/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    collection_name: str = "test_collection"
):
    contents = await file.read()
    file_path = f'../data/{file.filename}'
    
    with open(file_path, 'wb') as f:
        f.write(contents)
    
    background_tasks.add_task(process_file, file_path, file.filename, collection_name)
    
    return {"message": f"File {file.filename} uploaded. Processing in the background."}

async def process_file(file_path: str, filename: str, collection_name: str):
    if filename.endswith('.pdf'):
        data = await asyncio.to_thread(load_split_pdf_file, file_path, text_splitter)
    elif filename.endswith('.html'):
        data = await asyncio.to_thread(load_split_html_file, file_path, text_splitter)
    else:
        return {"message": "Only pdf and html files are supported"}
    
    await asyncio.to_thread(create_vector_db, data, embedding_model, collection_name)

@app.get("/query")
async def query(
    question: str, 
    n_results: int = Query(default=2, description="Number of results to return"),
    collection_name: str = Query(default="test_collection", description="Name of the document collection to search")
):
    try:
        collection_list = read_file('COLLECTIONS.txt')
        collection_list = collection_list.split("\n")[:-1]
    except Exception:
        return {"message": "No collections found. Upload some documents first"}

    if collection_name not in collection_list:
        return {"message": f"There is no collection with name {collection_name}",
                "available_collections": collection_list}
    
    collection = await asyncio.to_thread(load_local_db, collection_name)
    query_embedding = embedding_model.encode([question])[0].tolist()
    results = await asyncio.to_thread(collection.query, query_embeddings=[query_embedding], n_results=n_results)
    
    prompt = create_prompt(question, results)
    
    # Use the Inference API to generate the answer
    response = client.text_generation(
        model="mistralai/Mistral-7B-Instruct-v0.1",
        prompt=prompt,
        max_new_tokens=300,
        temperature=0.7
    )
    
    return {
        "question": question,
        "answer": response,
        "context": "\n".join(results['documents'][0])
    }

@app.get("/collections")
async def list_collections():
    try:
        collection_list = read_file('COLLECTIONS.txt')
        collection_list = collection_list.split("\n")[:-1]
        return {"collections": collection_list}
    except Exception:
        return {"message": "No collections found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)