from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import gradio as gr
import os
import time
import json

os.environ["OPENAI_API_KEY"] = 'EMPTY'
os.environ["OPENAI_API_BASE"] = "http://127.0.0.1:8000/v1"

QA_JSONL_PATH = "./train_qa.jsonl"


def load_llmmodel():
    model_name = "Qwen/Qwen3.5-35B-A3B"
    llm = ChatOpenAI(model_name=model_name)
    return llm


def load_qa_docs(jsonl_path):
    """从train_qa.jsonl载入文档：question→page_content, answer→metadata"""
    documents = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            # 用 question 作为文档正文
            # answer 以及其他字段放入 metadata
            doc = {
                "page_content": record["question"],
                "metadata": {
                    "answer": record["answer"],
                    "intent": record.get("intent", ""),
                    "source": record.get("source", ""),
                },
            }
            documents.append(doc)
    print(f"    共加载 {len(documents)} 条问答对")
    return documents


def create_vectorstore(documents):
    embeddings = SentenceTransformerEmbeddings(
        model_name="./models/shibing624/text2vec-base-chinese"
    )
    # 将 dict 列表转为 langchain Document 对象
    from langchain_core.documents import Document as LCDocument
    lc_docs = [LCDocument(page_content=d["page_content"], metadata=d["metadata"]) for d in documents]
    vectorstore = Chroma.from_documents(lc_docs, embeddings)
    return vectorstore


def search_docs(vectorstore, query):
    """相似度检索，返回匹配的 Document 列表（question+metadata）"""
    matching_docs = vectorstore.similarity_search(query)
    return matching_docs


def format_context(docs):
    """将检索到的文档拼成 LLM 上下文，附带 answer"""
    parts = []
    for i, doc in enumerate(docs):
        answer = doc.metadata.get("answer", "")
        parts.append(f"相关问题{i+1}：{doc.page_content}\n对应回答：{answer}")
    return "\n\n".join(parts)


def answer_fromchain(llm, matching_docs, query):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个客服助手。根据以下检索到的问答对，回答用户的问题。\n\n{context}"),
        ("human", "{question}"),
    ])
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "context": format_context(matching_docs),
        "question": query,
    })
    return answer


def chat_webui(llm):
    with gr.Blocks() as blocks:
        gr.HTML("""<h1 align="center">RAG演示（JSONL问答对）</h1>""")
        chatbot = gr.Chatbot()
        msg = gr.Textbox(show_label=False,
                         placeholder="请输入问题...", container=False)
        clear = gr.Button("清除问题")

        def messages(user_message, history):
            return "", history + [[user_message, None]]

        def bot(history):
            query = history[-1][0]
            matching_docs = search_docs(vectorstore, query)

            history[-1][1] = ""
            # 先展示检索到的匹配结果
            bot_message = "检索到的最相似问题：\n"
            for i, doc in enumerate(matching_docs[:3]):
                bot_message += (
                    f"  [{i+1}] {doc.page_content}\n"
                    f"      → 回答：{doc.metadata['answer']}\n"
                )
            for character in bot_message:
                history[-1][1] += character
                time.sleep(0.03)
            yield history

            # LLM 生成回答
            answer = answer_fromchain(llm, matching_docs, query)
            bot_message = "\nLLM生成回答：\n" + answer
            for character in bot_message:
                history[-1][1] += character
                time.sleep(0.03)
            yield history

        msg.submit(messages, [msg, chatbot],
                   [msg, chatbot], queue=False).then(
            bot, chatbot, chatbot
        )
        clear.click(lambda: None, None, chatbot, queue=False)
    blocks.queue()
    blocks.launch()


if __name__ == "__main__":
    print("装载模型")
    llm = load_llmmodel()
    print("装载问答数据")
    documents = load_qa_docs(QA_JSONL_PATH)
    print("创建向量库（question → embedding）")
    vectorstore = create_vectorstore(documents)
    print("启动WebUI")
    chat_webui(llm)
