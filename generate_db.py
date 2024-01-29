from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader

raw_doc = TextLoader(f'C:\\Users\\icecr\\Desktop\\ugo\\chatgpt\\shop_details.txt', encoding='utf-8').load()
# チャンクに分割
text_splitter = CharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=0,
)
chunks = text_splitter.split_documents(raw_doc)

# Embedding、FAISSに保存
db = FAISS.from_documents(chunks,HuggingFaceEmbeddings(model_name="oshizo/sbert-jsnli-luke-japanese-base-lite"))
db.save_local("shop_index")

#ロード
#db = FAISS.load_local(index_dir, embeddings)



#import requests
# ベクトルデータベースを生成してAWSにアップロードする関数
#def generate_and_upload_db():
    # ベクトルデータベースの生成

    # AWSにアップロード
    #data = {'db_content': db_content}
    #response