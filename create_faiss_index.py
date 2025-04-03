import faiss
import torch
from sentence_transformers import SentenceTransformer
import pickle

def main():
    # Embeddingモデルの準備
    embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
    embedding_model = SentenceTransformer(embedding_model_name)

    # ドキュメント（サンプル）: 実際には自分のコーパスを読み込む
    documents = [
        "東京は日本の首都です。",
        "パリはフランスの首都です。",
        "猫はかわいい動物です。",
        "人工知能は急速に発展しています。"
    ]

    # ベクトル化
    doc_embeddings = embedding_model.encode(documents, convert_to_tensor=True)

    # FAISSインデックスの作成
    dim = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(doc_embeddings.cpu().detach().numpy())

    # pickleで保存
    with open("faiss_index.pkl", "wb") as f:
        pickle.dump((index, documents), f)

    print("FAISSインデックスを作成して faiss_index.pkl に保存しました。")

if __name__ == "__main__":
    main()
