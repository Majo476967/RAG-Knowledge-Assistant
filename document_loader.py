     #读取PDF文件
from langchain_community.document_loaders import PyPDFLoader
    #从 langchain-community 库中导入专门处理 PDF 加载的工具类 PyPDFLoader；
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

    #文本切分
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20
    )
    
    chunks = text_splitter.split_documents(documents)
    #两个split_documents同名不同属
    return chunks