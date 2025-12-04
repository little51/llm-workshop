import gradio as gr
from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url="http://127.0.0.1:8000/v1/",
    api_key="EMPTY",
)

history = []


async def predict(message, history):
    history.append({"role": "user", "content": message})
    stream = await client.chat.completions.create(
        model="deepseek",
        messages=history,
        stream=True
    )
    chunks = []
    async for chunk in stream:
        chunks.append(chunk.choices[0].delta.content or "")
        print("".join(chunks), end="", flush=True)
        yield "".join(chunks)


demo = gr.ChatInterface(fn=predict)

demo.launch()
