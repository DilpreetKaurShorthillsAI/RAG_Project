import os
import pdfplumber
from transformers import AutoTokenizer, AutoModel
import torch
from typing import List
import google.generativeai as genai
from weaviate.auth import AuthApiKey
from weaviate.client import WeaviateClient
from dotenv import load_dotenv
load_dotenv()

class PDFLoader:
    def __init__(self, directory: str):
        self.directory = directory

    def load_pdfs(self) -> List[str]:
        """Loads and extracts text from all PDFs in the specified directory."""
        documents = []
        for file in os.listdir(self.directory):
            if file.endswith(".pdf"):
                pdf_path = os.path.join(self.directory, file)
                with pdfplumber.open(pdf_path) as pdf:
                    text = "".join(page.extract_text() or "" for page in pdf.pages)
                    documents.append(text)
        return documents

class TextSplitter:
    def __init__(self, chunk_size: int = 1000, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split_text(self, text: str) -> List[str]:
        """Splits text into overlapping chunks."""
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            start += self.chunk_size - self.overlap
        return chunks

class EmbeddingGenerator:
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def generate_embeddings(self, texts: List[str]) -> List[torch.Tensor]:
        """Generates embeddings for a list of texts."""
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
        return embeddings

class WeaviateStore:
    def __init__(self, weaviate_url: str, class_name: str, api_key: str):
        self.client = WeaviateClient(
            url=weaviate_url,
            additional_headers={"X-API-KEY": api_key}

        )
        self.class_name = class_name
        self._ensure_class_schema()

    def _ensure_class_schema(self):
        """Ensures the class schema exists in the Weaviate instance."""
        if not self.client.schema.contains_class(self.class_name):
            schema = {
                "class": self.class_name,
                "vectorizer": "none",
                "properties": [
                    {"name": "content", "dataType": ["text"]}
                ]
            }
            self.client.schema.create_class(schema)

    def add_documents(self, documents: List[str], embeddings: List[torch.Tensor]):
        """Adds documents and their embeddings to the Weaviate database."""
        for doc, embedding in zip(documents, embeddings):
            self.client.data_object.create(
                data_object={"content": doc},
                class_name=self.class_name,
                vector=embedding.tolist()
            )

    def query(self, query_embedding: List[float], top_k: int = 5) -> List[str]:
        """Performs a vector similarity search and returns the top-k results."""
        results = self.client.query.get(
            self.class_name, ["content"]
        ).with_near_vector({"vector": query_embedding}).with_limit(top_k).do()

        return [item["content"] for item in results["data"]["Get"][self.class_name]]

class RAGModel:
    def __init__(self, directory: str, weaviate_url: str, class_name: str, api_key: str, gemini_api_key: str):
        self.loader = PDFLoader(directory)
        self.splitter = TextSplitter()
        self.embedder = EmbeddingGenerator()
        self.vector_store = WeaviateStore(weaviate_url, class_name, api_key)
        genai.configure(api_key=gemini_api_key)
        self.llm_model = genai.GenerativeModel("gemini-1.5-flash")

    def build_knowledge_base(self):
        """Loads, splits, embeds, and stores documents in the vector store."""
        documents = self.loader.load_pdfs()
        all_chunks = []
        for doc in documents:
            chunks = self.splitter.split_text(doc)
            all_chunks.extend(chunks)

        embeddings = self.embedder.generate_embeddings(all_chunks)
        self.vector_store.add_documents(all_chunks, embeddings)
        print("Knowledge base built successfully!")

    def query(self, user_query: str, top_k: int = 5) -> str:
        """Queries the vector store and generates a response using LLM."""
        query_embedding = self.embedder.generate_embeddings([user_query])[0]
        relevant_chunks = self.vector_store.query(query_embedding.tolist(), top_k)
        context = " ".join(relevant_chunks)

        response = self._generate_response(user_query, context)
        return response

    def _generate_response(self, query: str, context: str) -> str:
        """Generates a response using the Gemini LLM."""
        prompt = (
            f"Context: {context}\n"
            f"Question: {query}\n"
            f"Answer:"
        )
        response = self.llm_model.generate_content(prompt)
        return response.text

if __name__ == "__main__":
    # Configuration
    directory = "data"
    weaviate_url = os.getenv("WCD_URL")
    class_name = "PDFDocuments"
    weaviate_api_key = os.getenv("WCD_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    # Initialize and build knowledge base
    rag_model = RAGModel(directory, weaviate_url, class_name, weaviate_api_key, gemini_api_key)
    rag_model.build_knowledge_base()

    # Query the knowledge base
    query = "What is Engineering Mechanics"
    response = rag_model.query(query)
    print("Response:", response)
