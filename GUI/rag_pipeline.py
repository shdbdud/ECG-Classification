import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from app_config import RAG_DIRS

try:
    import faiss
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False
    from sklearn.neighbors import NearestNeighbors


def build_retrieval_index(embeddings: np.ndarray, save_dir: Path):
    """
    离线使用：重建索引时才需要。
    GUI 主流程一般不会调用这个函数。
    """
    embeddings = embeddings.astype("float32")

    if HAS_FAISS:
        emb_norm = embeddings.copy()
        faiss.normalize_L2(emb_norm)

        index = faiss.IndexFlatIP(emb_norm.shape[1])
        index.add(emb_norm)

        index_path = save_dir / "rag_index.faiss"
        faiss.write_index(index, str(index_path))
        return "faiss", index_path

    else:
        nn = NearestNeighbors(metric="cosine")
        nn.fit(embeddings)

        index_path = save_dir / "rag_index_sklearn.joblib"
        joblib.dump(nn, index_path)
        return "sklearn", index_path


def load_retrieval_index(index_type: str, index_path: Path):
    if index_type == "faiss":
        import faiss
        return faiss.read_index(str(index_path))
    else:
        return joblib.load(index_path)


def retrieve_topk_unique(query_embedding, meta_df, index_obj, index_type="faiss", k=5, overfetch=20):
    query_embedding = np.asarray(query_embedding, dtype="float32").reshape(1, -1)
    overfetch = max(overfetch, k)

    if index_type == "faiss":
        import faiss
        q = query_embedding.copy()
        faiss.normalize_L2(q)
        scores, ids = index_obj.search(q, min(overfetch, len(meta_df)))
        ids = ids[0]
        scores = scores[0]
    else:
        n_neighbors = min(overfetch, len(meta_df))
        dist, ids = index_obj.kneighbors(query_embedding, n_neighbors=n_neighbors)
        ids = ids[0]
        scores = 1.0 - dist[0]

    seen_paths = set()
    keep_rows = []

    for idx, score in zip(ids, scores):
        row = meta_df.iloc[int(idx)].to_dict()
        src = row["source_path"]

        if src in seen_paths:
            continue

        seen_paths.add(src)
        row["similarity"] = float(score)
        keep_rows.append(row)

        if len(keep_rows) >= k:
            break

    out = pd.DataFrame(keep_rows).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def load_rag_assets(model_name: str):
    """
    读取某个模型已经离线生成好的 RAG 资产：
    - rag_meta.csv
    - rag_index.faiss 或 rag_index_sklearn.joblib
    """
    rag_dir = RAG_DIRS[model_name]
    meta_path = rag_dir / "rag_meta.csv"
    faiss_path = rag_dir / "rag_index.faiss"
    sklearn_path = rag_dir / "rag_index_sklearn.joblib"

    if not meta_path.exists():
        raise FileNotFoundError(f"Missing rag_meta.csv: {meta_path}")

    meta_df = pd.read_csv(meta_path)

    if faiss_path.exists():
        index_type = "faiss"
        index_path = faiss_path
    elif sklearn_path.exists():
        index_type = "sklearn"
        index_path = sklearn_path
    else:
        raise FileNotFoundError(
            f"No retrieval index found under {rag_dir}. "
            f"Expected rag_index.faiss or rag_index_sklearn.joblib"
        )

    index_obj = load_retrieval_index(index_type, index_path)
    return {
        "rag_dir": rag_dir,
        "meta_df": meta_df,
        "index_type": index_type,
        "index_path": index_path,
        "index_obj": index_obj,
    }


def retrieve_similar_cases(model_name: str, query_embedding, k=5):
    assets = load_rag_assets(model_name)
    retrieved_df = retrieve_topk_unique(
        query_embedding=query_embedding,
        meta_df=assets["meta_df"],
        index_obj=assets["index_obj"],
        index_type=assets["index_type"],
        k=k,
        overfetch=max(20, k)
    )
    return retrieved_df