import os
import sys
import argparse
from pathlib import Path
import chromadb
import numpy as np

# Add the parent directory to sys.path so 'config' can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Note: You will need to pip install yt-dlp openai-whisper PyMuPDF sentence-transformers
try:
    import yt_dlp
    import whisper
    import pymupdf as fitz  # Updated from import fitz
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

def ingest_local_audio(audio_path, collection, embedding_model):
    print(f"Transcribing local audio with Whisper: {audio_path}")
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    
    print("Ingesting into ChromaDB...")
    chunks = []
    metadata = []
    ids = []
    
    for i, segment in enumerate(result["segments"]):
        text = segment["text"].strip()
        if not text: continue
        chunks.append(text)
        metadata.append({
            "source": audio_path,
            "source_type": "audio",
            "start": segment["start"],
            "end": segment["end"]
        })
        ids.append(f"audio_{Path(audio_path).stem}_{i}")
        
    if chunks:
        print("Generating embeddings...")
        embeddings = embedding_model.encode(chunks).tolist()
        
        collection.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadata,
            ids=ids
        )
        print(f"Added {len(chunks)} audio chunks to ChromaDB.")
    else:
        print("No audio transcribed.")

def main():
    parser = argparse.ArgumentParser(description="Ingest multimodal content into ChromaDB")
    parser.add_argument("--youtube", type=str, help="YouTube URL to ingest")
    parser.add_argument("--slides", type=str, help="Path to PDF slide deck to ingest")
    parser.add_argument("--audio", type=str, help="Path to local audio file to ingest (bypasses YouTube blocks)")
    args = parser.parse_args()

    if not args.youtube and not args.slides and not args.audio:
        print("Please provide --youtube, --slides, or --audio argument.")
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
        
    if args.audio:
        ingest_local_audio(args.audio, collection, embedding_model)

if __name__ == "__main__":
    main()
