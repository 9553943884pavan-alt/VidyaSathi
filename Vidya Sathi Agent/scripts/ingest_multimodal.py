import os
import argparse
from pathlib import Path
import chromadb
import numpy as np

# Note: You will need to pip install yt-dlp openai-whisper PyMuPDF sentence-transformers
try:
    import yt_dlp
    import whisper
    import fitz  # PyMuPDF
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Please install required packages: pip install yt-dlp openai-whisper PyMuPDF sentence-transformers")
    exit(1)

from config.settings import get_settings

def ingest_youtube(url: str, collection: chromadb.Collection, embedding_model: SentenceTransformer):
    print(f"Downloading audio from {url}...")
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': 'temp_audio.%(ext)s',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    
    print("Transcribing with Whisper...")
    model = whisper.load_model("base")
    result = model.transcribe("temp_audio.mp3")
    
    print("Ingesting into ChromaDB...")
    documents = []
    metadatas = []
    ids = []
    embeddings = []
    
    for i, segment in enumerate(result['segments']):
        text = segment['text'].strip()
        if not text:
            continue
            
        timestamp = segment['start']
        
        documents.append(text)
        metadatas.append({
            "source": url,
            "source_type": "video",
            "timestamp": timestamp
        })
        ids.append(f"video_{url}_{i}")
        
    if documents:
        print("Generating embeddings...")
        embeddings = embedding_model.encode(documents).tolist()
        
        collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Added {len(documents)} video chunks to ChromaDB.")
        
    if os.path.exists("temp_audio.mp3"):
        os.remove("temp_audio.mp3")

def ingest_slides(pdf_path: str, collection: chromadb.Collection, embedding_model: SentenceTransformer):
    print(f"Extracting text from {pdf_path}...")
    documents = []
    metadatas = []
    ids = []
    
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text").strip()
        
        if not text:
            continue
            
        documents.append(text)
        metadatas.append({
            "source": os.path.basename(pdf_path),
            "source_type": "slide",
            "slide_number": page_num + 1
        })
        ids.append(f"slide_{os.path.basename(pdf_path)}_{page_num}")
        
    if documents:
        print("Generating embeddings...")
        embeddings = embedding_model.encode(documents).tolist()
        
        collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Added {len(documents)} slide chunks to ChromaDB.")

def main():
    parser = argparse.ArgumentParser(description="Ingest multimodal content into ChromaDB")
    parser.add_argument("--youtube", type=str, help="YouTube URL to ingest")
    parser.add_argument("--slides", type=str, help="Path to PDF slide deck to ingest")
    args = parser.parse_args()

    if not args.youtube and not args.slides:
        print("Please provide either --youtube or --slides argument.")
        return

    settings = get_settings()
    persist_dir = Path(settings.chroma_persist_directory)
    
    if not persist_dir.exists():
        print(f"Creating Chroma directory at {persist_dir}")
        persist_dir.mkdir(parents=True, exist_ok=True)
        
    client = chromadb.PersistentClient(path=str(persist_dir))
    
    # We use get_or_create to add to the existing collection alongside NCERT text
    collection = client.get_or_create_collection(name=settings.collection_name)
    
    print("Loading embedding model (BAAI/bge-small-en-v1.5)...")
    embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    
    if args.youtube:
        ingest_youtube(args.youtube, collection, embedding_model)
        
    if args.slides:
        ingest_slides(args.slides, collection, embedding_model)

if __name__ == "__main__":
    main()
