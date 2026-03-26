from document_loader import load_pdf, split_documents
from embeddings.embedding_model import get_embedding_model
from vectorstore.faiss_store import build_vector_store
from LLM.doubao_LLM import generate_answer

def search_docs(vectorstore, query, k=5):
    docs = vectorstore.similarity_search(query, k=k)
    return docs

def run():
    file_path = "documents/test.pdf"

    # 加载 + 切分
    documents = load_pdf(file_path)
    chunks = split_documents(documents)

    print(f"\n📊 chunk数量：{len(chunks)}")

    # 构建向量库
    embedding_model = get_embedding_model()
    vectorstore = build_vector_store(chunks, embedding_model)

    print("✅ 向量库构建完成，可以开始提问")

    # 进入问答循环
    while True:
        query = input("\n请输入你的问题（输入q退出）：")

        if query.lower() == "q":
            print("退出问答")
            break

        # 检索
        docs = search_docs(vectorstore, query, k=3)

        # 关键词过滤（优化）
        filtered_docs = docs

        if "优点" in query:
           temp = []
        for doc in docs:
            if "优点" in doc.page_content:
                    temp.append(doc)
            if temp:
                filtered_docs = temp

        if "缺点" in query:
             temp = []
        for doc in docs:
            if "缺点" in doc.page_content:
                    temp.append(doc)
            if temp:
                filtered_docs = temp

        docs = filtered_docs

        # 排序优化（内容多的优先）
        docs = sorted(docs, key=lambda x: len(x.page_content), reverse=True)

        # 去重优化（保留排序顺序）
        unique_contents = []
        seen = set()       #集合不允许重复
        for doc in docs:
            if doc.page_content not in seen:
                unique_contents.append(doc.page_content)
                seen.add(doc.page_content)

        # 拼接上下文，使用去重后的内容
        context = "\n".join(unique_contents)

        # 打印检索结果（调试用）
        print("\n🔍 检索结果：")
        for i, doc in enumerate(docs):
            print(f"\n--- 结果 {i} ---")
            print(doc.page_content[:200])
            print("来源：", doc.metadata)

        print("\n📚 拼接后的上下文：")
        print(context[:500])

        # 拼接 Prompt
        prompt = f"""
       你是一个专业的AI助手，请严格根据提供的文档内容回答问题。

       要求：
       1. 只使用文档中的信息回答
       2. 如果有多个要点，请分点说明
       3. 语言简洁清晰
       4. 不要编造内容

       文档内容：
       {context}

       问题：
       {query}

       回答：
       """
        print("\n🧠 Prompt：")
        print(prompt[:500])

        # 调用大模型
        answer = generate_answer(prompt)

        print("\n🤖 最终答案：")
        print(answer)

        # 引用来源
        print("\n📚 引用来源：")
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "未知来源")
            preview = doc.page_content[:50].replace("\n", "")
            print(f"[{i+1}] {preview}...")
