import streamlit as st
import asyncio
from openai import AsyncOpenAI
import httpx

# 页面设置
st.set_page_config(page_title="Chat Bot", page_icon="⚡")
st.title("Chat应用")


client = AsyncOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="EMPTY",
    http_client=httpx.AsyncClient(timeout=60.0)
)

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


async def chat_bot(prompt):
    """获取异步流式响应"""
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)

    # AI回复区域
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        try:
            # 异步流式调用
            stream = await client.chat.completions.create(
                model="qwen3",
                messages=[{"role": m["role"], "content": m["content"]}
                          for m in st.session_state.messages],
                stream=True,
                temperature=0.7
            )

            # 处理流式响应
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    text_chunk = chunk.choices[0].delta.content
                    full_response += text_chunk
                    placeholder.markdown(full_response + "▌")

            # 完成输出
            placeholder.markdown(full_response)

        except Exception as e:
            error_msg = f"错误: {str(e)}"
            placeholder.markdown(error_msg)
            full_response = error_msg

    # 添加AI回复
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response})

# 处理用户输入
if prompt := st.chat_input("请输入问题..."):
    asyncio.run(chat_bot(prompt))
