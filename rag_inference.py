import os
import pickle
import torch
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. 事前に作成したFAISSインデックスとドキュメントを読み込み
with open("faiss_index.pkl", "rb") as f:
    index, documents = pickle.load(f)

# 2. Embeddingモデルの読み込み
embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
embedding_model = SentenceTransformer(embedding_model_name)

# 3. ファインチューニング済みGPT-2をロード
#    (例) チェックポイントを指定する場合: "./results/checkpoint-500"
model_checkpoint = "./results/final_model"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)  # OK: tokenizerファイルあり
model = AutoModelForCausalLM.from_pretrained(model_checkpoint)

# 4. 推論用の関数を定義
def generate_answer_with_rag(query, top_k=2, max_length=100):
    """
    入力query(文字列)に対し、
    1) ベクトル検索により関連ドキュメントを取得
    2) 取得ドキュメントをプロンプトに組み込み
    3) GPT-2で文章生成
    """

    # (A) クエリをEmbeddingして検索
    query_emb = embedding_model.encode([query], convert_to_tensor=False)
    # FAISSのsearchは numpy array で行う
    D, I = index.search(query_emb, top_k)  # 上位top_k件

    # (B) 検索結果を取得して連結(簡易)
    retrieved_docs = [documents[i] for i in I[0]]
    context_text = "\n".join(retrieved_docs)

    # (C) LLMへの入力プロンプトを作成
    #     - 実際の運用ではフォーマットを工夫して、以下のようにすることが多い
    #       「以下の文章を参考に質問に答えてください: ... 質問: ...」
    prompt = f"以下の文章を参考に回答してください。\n\n---\n{context_text}\n---\n質問: {query}\n回答:"

    # (D) テンソル化して生成
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        do_sample=True,
        top_p=0.9,
        top_k=40,
        temperature=0.8
    )
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text

if __name__ == "__main__":
    # GPUがあれば利用
    if torch.cuda.is_available():
        model.cuda()

    # テストクエリ
    user_query = "日本の首都はどこですか？"
    answer = generate_answer_with_rag(user_query)
    print("-----")
    print(f"Query: {user_query}")
    print(f"Answer:\n{answer}")
