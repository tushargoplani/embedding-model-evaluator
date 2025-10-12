import os
import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import cohere
import voyageai
import asyncio
import logging
import numpy as np
from dotenv import load_dotenv

from app.utils.utils import prepare_data_for_chroma

load_dotenv()
client = chromadb.PersistentClient(path="chroma_db")
cohere_client = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
voyage_client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
logging.basicConfig(level=logging.INFO)

class Kb_Import:
    async def add_file_import(self, file_path):
        logging.info(f"Adding file import: {file_path}")
        file_name = os.path.basename(file_path)
        chunks = self.load_pdf_chunks(file_path, file_name)
        await self.create_embed_and_store(chunks)
        os.remove(file_path)
        return { 
            "chunks": chunks,
            "file_name": file_name,
            "file_id": os.path.splitext(file_name)[0].split("_")[1],
        }

    def load_pdf_chunks(self, file_path, file_name):
        doc = fitz.open(file_path)
        all_text = ""

        for page in doc:
            all_text += page.get_text()
        doc.close()

        chunks = self.create_chunks(all_text, file_name)
        return chunks

    def create_chunks(self, text, file_name="", chunk_size=2000, chunk_overlap=200):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", " ", ""]
        )

        split_texts = text_splitter.split_text(text)
        logging.info(f"Total chunks: {len(split_texts)}")

        file_id = os.path.splitext(file_name)[0].split("_")[1]
        chunk_dicts = []

        for i, chunk in enumerate(split_texts):
            chunk_dicts.append({
                "text": chunk,
                "metadata": {
                    "chunk_id": str(i),
                    "source_file": file_name,
                    "file_id": file_id
                }
            })

        return chunk_dicts

    async def create_embed_and_store(self, chunks, batch_size=10):
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            logging.info(f"Starting batch {i // batch_size + 1} with {len(batch)} chunks")
            data_for_chroma = prepare_data_for_chroma(batch)
            # Run all three embedding + store functions concurrently in threads
            await asyncio.gather(
                asyncio.to_thread(self.create_embed_with_sentence_transformer_and_store, data_for_chroma),
                asyncio.to_thread(self.create_embed_with_cohere_and_store, data_for_chroma),
                asyncio.to_thread(self.create_embed_with_voyage_and_store, data_for_chroma)
            )
            logging.info(f"Batch {i // batch_size + 1} completed")

            # Wait 1 minute before next batch if there are more batches
            if i + batch_size < len(chunks):
                logging.info("Waiting 60 seconds to respect rate limit...")
                await asyncio.sleep(60)

        return True

    def create_embed_with_sentence_transformer_and_store(self, data_for_chroma):
        model = SentenceTransformer("all-MiniLM-L6-v2")
        hf_db = client.get_or_create_collection("huggingface_embeddings", 
            configuration={
                "hnsw": {
                    "space": "cosine",
                    "ef_construction": 200
                }
            }
        )
        embeddings = model.encode(data_for_chroma["documents"])
        # L2-normalize each embedding vector
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Convert to list to store in Chroma
        embeddings = embeddings.tolist()
        hf_db.add(
            ids=data_for_chroma["ids"],
            documents=data_for_chroma["documents"],
            embeddings=embeddings,
            metadatas=data_for_chroma["metadatas"]
        )
        logging.info("Sentence Transformer embeddings stored in chroma successfully")

    def create_embed_with_cohere_and_store(self, data_for_chroma):
        cohere_db = client.get_or_create_collection("cohere_embeddings", 
            configuration={
                "hnsw": {
                    "space": "cosine",
                    "ef_construction": 200
                }
            }
        )

        response = cohere_client.embed(
            model="embed-v4.0",
            texts=data_for_chroma["documents"],
            input_type="classification",
            output_dimension=1536,
            embedding_types=["float"],
        )
        embeddings = response.embeddings.float
        cohere_db.add(
            ids=data_for_chroma["ids"],
            documents=data_for_chroma["documents"],
            embeddings=embeddings,
            metadatas=data_for_chroma["metadatas"]
        )
        logging.info("Cohere embeddings stored in chroma successfully")

    def create_embed_with_voyage_and_store(self, data_for_chroma):
        voyage_db = client.get_or_create_collection("voyage_embeddings", 
            configuration={
                "hnsw": {
                    "space": "cosine",
                    "ef_construction": 200
                }
            }
        )
        result = voyage_client.embed(
            data_for_chroma["documents"], 
            model="voyage-3.5", 
            input_type="document"
        )
        voyage_db.add(
            ids=data_for_chroma["ids"],
            documents=data_for_chroma["documents"],
            embeddings=result.embeddings,
            metadatas=data_for_chroma["metadatas"]
        )
        logging.info("Voyage embeddings stored in chroma successfully")