def prepare_data_for_chroma(chunks):
    # Prepare chunk data for storing in Chroma or any vector DB.
    ids = [chunk["metadata"]["chunk_id"] for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    
    return {"ids": ids, "documents": documents, "metadatas": metadatas}

def format_chroma_results(results):
    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    combined = []
    for i in range(len(ids)):
        combined.append({
            "text": documents[i],
            "metadata": metadatas[i],
            "score": distances[i]
        })
    combined = sorted(combined, key=lambda x: x["score"], reverse=True)
    
    return combined