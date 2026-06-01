import gradio as gr
from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="0000",
)


class StreamChatBot:
    def __init__(self, model: str = "qwen3"):
        self.model = model

    async def stream_response(self, message: str, history: list):
        messages = []

        # 系统提示
        system_prompt = "你是一个专业的客服助手。请用中文回答用户的问题，回答要简洁、礼貌、准确。"
        messages.append({"role": "system", "content": system_prompt})

        # 添加历史对话
        for msg in history:
            if msg["role"] == "user":
                messages.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "assistant":
                messages.append({"role": "assistant", "content": msg["content"]})

        # 添加当前用户消息
        messages.append({"role": "user", "content": message})

        try:
            stream = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                max_tokens=2048,
                temperature=0.7,
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    yield content

        except Exception as e:
            yield f"抱歉，发生了错误: {str(e)}"


chat_bot = StreamChatBot()


async def predict(message, history):
    full_response = ""
    async for content in chat_bot.stream_response(message, history):
        full_response += content
        yield full_response


# 创建聊天界面
def create_chat_interface():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🛒 智能客服助手")

        chatbot = gr.Chatbot(
            label="客服对话",
            height=500,
            show_copy_button=True,
            type="messages",
        )

        with gr.Row():
            msg = gr.Textbox(
                label="输入您的问题",
                placeholder="请输入您的问题，例如：我的订单什么时候发货？",
                scale=4,
                container=False,
            )
            submit_btn = gr.Button("发送", variant="primary", scale=1)

        with gr.Row():
            clear_btn = gr.Button("清空对话", variant="secondary")

        # 处理消息提交
        async def respond(message, chat_history):
            if not message.strip():
                yield chat_history, ""
                return

            chat_history.append({"role": "user", "content": message})
            full_response = ""
            async for content in predict(message, chat_history[:-1]):
                full_response = content
                if len(chat_history) > 0 and chat_history[-1]["role"] == "user":
                    chat_history.append({"role": "assistant", "content": full_response})
                else:
                    chat_history[-1] = {"role": "assistant", "content": full_response}
                yield chat_history, ""

        submit_btn.click(respond, [msg, chatbot], [chatbot, msg])
        msg.submit(respond, [msg, chatbot], [chatbot, msg])
        clear_btn.click(lambda: [], None, chatbot)

    return demo


if __name__ == "__main__":
    demo = create_chat_interface()
    demo.launch(share=False)
