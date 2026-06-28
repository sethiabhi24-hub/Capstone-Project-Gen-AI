import os
import glob
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from google import genai
from google.genai import types
from pypdf import PdfReader
import docx

EXTRACT_DIR = "./novatech_dataset"
DB_PATH = "./novatech_qdrantdb"

def extract_text_from_file(file_path):
    """Extracts raw text content from PDF, DOCX, or TXT files."""
    text = ""
    try:
        if file_path.endswith('.pdf'):
            reader = PdfReader(file_path)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        elif file_path.endswith('.docx'):
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                text += para.text + "\n"
        elif file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return text.strip()

def build_vector_database(client: genai.Client):
    """Scans files, generates embeddings, and inserts them into Qdrant."""
    print("Initializing Qdrant client and scanning documents...")
    
    qdrant_client = QdrantClient(path=DB_PATH)
    collection_name = "novatech_policies"
    
    if qdrant_client.collection_exists(collection_name):
        qdrant_client.delete_collection(collection_name)
        
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )
    
    doc_files = []
    for ext in ('*.pdf', '*.docx', '*.txt'):
        doc_files.extend(glob.glob(f"{EXTRACT_DIR}/**/{ext}", recursive=True))
    
    document_chunks = []
    document_metadata = []
    
    for file_path in doc_files:
        raw_text = extract_text_from_file(file_path)
        file_name = os.path.basename(file_path)
        chunks = [c.strip() for c in raw_text.split('\n\n') if len(c.strip()) > 50]
        for i, chunk in enumerate(chunks):
            document_chunks.append(chunk)
            document_metadata.append({"source": file_name, "chunk_id": i})
            
    batch_size = 50
    for i in range(0, len(document_chunks), batch_size):
        batch_texts = document_chunks[i:i+batch_size]
        batch_meta = document_metadata[i:i+batch_size]
        
        response = client.models.embed_content(
            model='gemini-embedding-001',
            contents=batch_texts,
            config=types.EmbedContentConfig(output_dimensionality=768)
        )
        batch_embeddings = [emb.values for emb in response.embeddings]
        
        points = []
        for j, (text, meta, vector) in enumerate(zip(batch_texts, batch_meta, batch_embeddings)):
            point_id = i + j
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={"text": text, "source": meta["source"], "chunk_id": meta["chunk_id"]}
                )
            )
            
        qdrant_client.upsert(collection_name=collection_name, points=points)
        
    print(f"RAG db in Chunks embedded and stored.")

def query_knowledge_base(query: str, client: genai.Client) -> str:
    """Queries the db"""
    print(f"Querying db for -> '{query}'")
    
    qdrant_client = QdrantClient(path=DB_PATH)
    collection_name = "novatech_policies"
    
    query_emb_response = client.models.embed_content(
        model='gemini-embedding-001',
        contents=query,
        config=types.EmbedContentConfig(output_dimensionality=768)
    )
    query_vector = query_emb_response.embeddings[0].values
    
    search_results = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=3
    ).points
    
    if not search_results:
        return "No relevant policy information found."
        
    formatted_context = "--- RETRIEVED POLICIES ---\n"
    for point in search_results:
        payload = point.payload
        formatted_context += f"[Source: {payload['source']}]\n{payload['text']}\n\n"
        
    return formatted_context
