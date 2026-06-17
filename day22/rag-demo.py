from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import gradio as gr
import os
import time
os.environ["OPENAI_API_KEY"] = 'EMPTY'
os.environ["OPENAI_API_BASE"] = "http://127.0.0.1:8000/v1"


def load_llmmodel():
    model_name = "Qwen/Qwen3.5-35B-A3B"
    llm = ChatOpenAI(model_name=model_name)
    return llm


def load_docs(directory):
    loader = DirectoryLoader(directory, glob='**/*.*',
                             show_progress=True)
    documents = loader.load()
    return documents


def split_docs(documents):
    text_splitter = CharacterTextSplitter(chunk_size=150,
                                          chunk_overlap=20)
    split_docs = text_splitter.split_documents(documents)
    return split_docs


def create_vectorstore(split_docs):
    embeddings = SentenceTransformerEmbeddings(
        model_name="./models/shibing624/text2vec-base-chinese"
    )
    vectorstore = Chroma.from_documents(split_docs, embeddings)
    return vectorstore


def search_docs(vectorstore, query):
    matching_docs = vectorstore.similarity_search(query)
    return matching_docs


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def answer_fromchain(llm, matching_docs, query):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Use the following pieces of context to answer the user's question.\n\n{context}"),
        ("human", "{question}"),
    ])
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": format_docs(matching_docs), "question": query})
    return answer


def chat_webui(llm):
    with gr.Blocks() as blocks:
        gr.HTML("""<h1 align="center">RAG演示</h1>""")
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
            bot_message = "知识库检索结果：\n" + \
                matching_docs[0].page_content + "[" + \
                matching_docs[0].metadata["source"] + "]"
            for character in bot_message:
                history[-1][1] += character
                time.sleep(0.05)
                yield history
            answer = answer_fromchain(llm, matching_docs, query)
            bot_message = "\nLLM生成结果：\n" + answer
            for character in bot_message:
                history[-1][1] += character
                time.sleep(0.05)
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
    print("装载文档")
    documents = load_docs("./documents")
    print("分割文档")
    split_docs = split_docs(documents)
    print("创建向量库")
    vectorstore = create_vectorstore(split_docs)
    print("启动WebUI")
    chat_webui(llm)
