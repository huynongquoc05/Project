# Load from disk
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

db=FAISS.load_local("vector_db2chunk_nltk", embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large-instruct", )
,allow_dangerous_deserialization=True)
# # Example query
query= """Nhập dữ liệu từ bàn phím trong Java
# """
#
#
results = db.similarity_search_with_score(query, k=10)
print("query:", query)
for r, score in results:
    print(len(r.page_content))
    print(f"Score: {score:.4f}")
    print(r.page_content[:])
    print("-" * 80)
