import chromadb
from fastapi import HTTPException

class ChromaDBClient:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="models/chroma_db")
        self.collection = self.client.get_or_create_collection("documents")

    def store_document(self, doc_name, embeddings, doc_text):
        self.collection.add(
            embeddings=[embeddings],
            documents=[doc_text],
            metadatas=[{"name": doc_name}],
            ids=[doc_name]
        )

    def query_similar_documents(self, query_embedding, n_results):
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            return results
        except Exception as e:
            if "Collection is empty" in str(e):
                raise HTTPException(status_code=404, detail="No documents have been ingested yet")
            raise
