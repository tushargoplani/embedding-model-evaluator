import os
from sentence_transformers import SentenceTransformer
from ranx import Qrels, Run, compare
import chromadb
import cohere
import voyageai
import asyncio
import logging
import numpy as np
import math
from dotenv import load_dotenv
from app.utils.utils import format_chroma_results

load_dotenv()
client = chromadb.PersistentClient(path="chroma_db")
cohere_client = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
voyage_client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
logging.basicConfig(level=logging.INFO)

class Query:
    def __init__(self):
        self.request_counter = 0

    async def evaluate_embedding_model(self, request_data):
        queries = request_data["queries"]
        top_k = request_data.get("top_k", 5)
        file_ids = request_data.get("file_ids", None)
        qrels_dict = request_data.get("qrels", {})
        qrels = Qrels(qrels_dict) if qrels_dict else None
        logging.info(f"Querying: {queries}")

        final_results = {}
        st_results = {}
        co_results = {}
        voy_results = {}

        for q in queries:
            logging.info(f"Processing query: {q}")

            # Check rate limit
            if self.request_counter >= 3:
                logging.info("Hit rate limit, sleeping for 60s...")
                await asyncio.sleep(60)
                self.request_counter = 0  # reset counter

            self.request_counter += 1  # count this batch of requests

            # Run all 3 models in parallel
            st_ctx, co_ctx, voy_ctx = await asyncio.gather(
                asyncio.to_thread(self.get_context_from_sentence_transformer, q, top_k, file_ids),
                asyncio.to_thread(self.get_context_from_cohere, q, top_k, file_ids),
                asyncio.to_thread(self.get_context_from_voyage, q, top_k, file_ids),
            )
            print("Length: ", len(st_ctx), len(co_ctx), len(voy_ctx))
            # Store raw results
            final_results[q] = {
                "sentence_transformer": st_ctx,
                "cohere": co_ctx,
                "voyage": voy_ctx,
            }

            # Flatten results for Ranx
            st_run = {item["metadata"]["chunk_id"]: float(item["score"]) for item in st_ctx}
            co_run = {item["metadata"]["chunk_id"]: float(item["score"]) for item in co_ctx}
            voy_run = {item["metadata"]["chunk_id"]: float(item["score"]) for item in voy_ctx}

            # Save for overall metrics calculation
            st_results[q] = st_run
            co_results[q] = co_run
            voy_results[q] = voy_run

            # Per-query metrics
            if qrels:
                per_query_qrels = Qrels({q: qrels_dict.get(q, {})})
                per_query_metrics = compare(
                    per_query_qrels,
                    [
                        Run({q: st_run}, name="sentence_transformer"),
                        Run({q: co_run}, name="cohere"),
                        Run({q: voy_run}, name="voyage"),
                    ],
                    metrics=["map", "ndcg", "recall@5", "recall@10"]
                ).to_dict()
                # Clean float values and add to results
                per_query_metrics_clean = self._filter_ranx_metrics(per_query_metrics)
                final_results[q]["metrics"] = per_query_metrics_clean

        # Overall metrics across all queries
        overall_metrics = {}
        best_model = ""
        if qrels:
            runs_dict = {
                "sentence_transformer": Run(st_results, name="sentence_transformer"),
                "cohere": Run(co_results, name="cohere"),
                "voyage": Run(voy_results, name="voyage"),
            }
            overall_metrics_raw = compare(
                qrels,
                list(runs_dict.values()),
                metrics=["map", "ndcg", "recall@5", "recall@10"]
            ).to_dict()
            overall_metrics = self._filter_ranx_metrics(overall_metrics_raw)
            best_model = self._find_best_model(overall_metrics)

        return {
            "results": final_results,
            "overall_metrics": overall_metrics,
            "best_model": best_model
        }


    def get_context_from_sentence_transformer(self, query, top_k, file_ids):
        model = SentenceTransformer("all-MiniLM-L6-v2")
        hf_db = client.get_or_create_collection("huggingface_embeddings", 
            metadata={"hnsw:space": "cosine"}
        )
        print("hf_db ", hf_db)
        query_emb = model.encode(query)
        # L2-normalize the query embedding
        query_emb = query_emb / np.linalg.norm(query_emb)
        # Retrieve top-k
        results = hf_db.query(
            query_embeddings=query_emb,
            n_results=top_k,
            where={"file_id": {"$in": file_ids}} if file_ids else None,
        )

        return format_chroma_results(results)

    def get_context_from_cohere(self, query, top_k, file_ids):
        cohere_db = client.get_or_create_collection("cohere_embeddings", 
            metadata={"hnsw:space": "cosine"}
        )
        response = cohere_client.embed(
            model="embed-v4.0",
            texts=[query],
            input_type="classification",
            output_dimension=1536,
            embedding_types=["float"],
        )
        query_emb = response.embeddings.float[0]
        # Retrieve top-k
        results = cohere_db.query(
            query_embeddings=query_emb,
            n_results=top_k,
            where={"file_id": {"$in": file_ids}} if file_ids else None,
        )

        return format_chroma_results(results)

    def get_context_from_voyage(self, query, top_k, file_ids):
        voyage_db = client.get_or_create_collection("voyage_embeddings", 
            metadata={"hnsw:space": "cosine"}
        )
        response = voyage_client.embed([query], model="voyage-3.5", input_type="document")
        query_emb = response.embeddings[0]
        # Retrieve top-k
        results = voyage_db.query(
            query_embeddings=query_emb,
            n_results=top_k,
            where={"file_id": {"$in": file_ids}} if file_ids else None,
        )

        return format_chroma_results(results)
        
    def _filter_ranx_metrics(self, full_metrics):
        """Extracts only the average scores from the Ranx comparison object."""
        # This structure must match the model_names list in the Ranx output
        model_names = full_metrics.get("model_names", []) 
        
        # Initialize the simplified metrics dictionary
        simplified_metrics = {}
        
        for model_name in model_names:
            # Check if the model's scores exist in the full output
            model_data = full_metrics.get(model_name)
            if model_data and "scores" in model_data:
                # Extract the scores and round the floating point numbers
                scores = model_data["scores"]
                simplified_metrics[model_name] = {
                    metric: round(score, 4) if isinstance(score, float) else score
                    for metric, score in scores.items()
                }
                
        return simplified_metrics
    
    def _find_best_model(self, metrics):
        """
        Pick the best model based on MAP (or NDCG as fallback).
        """
        if not metrics:
            return None
        
        best_model = None
        best_score = -1

        for model, scores in metrics.items():
            map_score = scores.get("map", 0)
            ndcg_score = scores.get("ndcg", 0)

            # Prefer MAP, fallback to NDCG if MAPs are equal
            if map_score > best_score or (map_score == best_score and ndcg_score > scores.get("ndcg", 0)):
                best_score = map_score
                best_model = model

        return best_model