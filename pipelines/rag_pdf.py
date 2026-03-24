from document_loader import load_pdf, split_documents

def run():
    file_path = "documents/test.pdf"
    
    documents = load_pdf(file_path)
    chunks = split_documents(documents)
    
    print(f"原始文档页数: {len(documents)}")
    print(f"切分后chunk数量: {len(chunks)}")
    
    print("\n示例chunk内容：")
    print(chunks[0].page_content)

