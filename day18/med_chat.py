import gradio as gr
from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="0000",
)


class StreamChatBot:
    def __init__(self, model: str = "qwen3"):
        self.model = model

    async def stream_response(self, message: str, history: list, medical_info: dict):
        messages = []

        # 系统提示
        system_prompt = "你是一个专业的中医助手。请用中文回答用户的问题。"
        messages.append({
            "role": "system",
            "content": system_prompt
        })

        # 添加历史对话
        for msg in history:
            if msg["role"] == "user":
                messages.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "assistant":
                messages.append(
                    {"role": "assistant", "content": msg["content"]})

        # 构建包含医疗信息的用户消息
        user_message = "患者医疗信息：\n"
        
        if medical_info["present_illness"]:
            user_message += f"- 现病史：{medical_info['present_illness']}\n"
        if medical_info["past_history"]:
            user_message += f"- 既往史：{medical_info['past_history']}\n"
        if medical_info["current_symptoms"]:
            user_message += f"- 刻下症：{medical_info['current_symptoms']}\n"
        if medical_info["allergy_history"]:
            user_message += f"- 过敏史：{medical_info['allergy_history']}\n"
        if medical_info["tcm_diagnosis"]:
            user_message += f"- 中医四诊：{medical_info['tcm_diagnosis']}\n"
        if medical_info["physical_exam"]:
            user_message += f"- 体格检查：{medical_info['physical_exam']}\n"
        if medical_info["diagnosis_name"]:
            user_message += f"- 诊断名称：{medical_info['diagnosis_name']}\n"
        if medical_info["tcm_syndrome"]:
            user_message += f"- 中医症候：{medical_info['tcm_syndrome']}\n"

        user_message += f"\n用户问题：{message}"
        user_message += "\n\n请基于以上患者信息提供专业的中医诊疗建议。"
        print(user_message)
        messages.append({"role": "user", "content": user_message})
        try:
            stream = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                max_tokens=4096,
                temperature=0.7,
            )
            async for chunk in stream:
                if (chunk.choices and
                        chunk.choices[0].delta.content is not None):
                    content = chunk.choices[0].delta.content
                    yield content

        except Exception as e:
            yield f"抱歉，发生了错误: {str(e)}"


chat_bot = StreamChatBot()


async def predict(message, history, medical_info):
    full_response = ""
    async for content in chat_bot.stream_response(message, history, medical_info):
        full_response += content.replace("<think>",
                                         "思考...").replace("</think>", "思考完成")
        yield full_response


# 创建医疗信息输入组件
def create_medical_inputs():
    with gr.Accordion("患者医疗信息", open=True):
        present_illness = gr.Textbox(
            label="现病史",
            value="患者主诉咳嗽、咳痰3天，伴有发热",
            lines=2,
            placeholder="请输入患者现病史..."
        )
        past_history = gr.Textbox(
            label="既往史",
            value="无重大疾病史，无手术史",
            lines=2,
            placeholder="请输入患者既往史..."
        )
        current_symptoms = gr.Textbox(
            label="刻下症",
            value="咳嗽频作，痰黄粘稠，发热38.5℃，口渴，咽痛",
            lines=2,
            placeholder="请输入患者刻下症..."
        )
        allergy_history = gr.Textbox(
            label="过敏史",
            value="无药物及食物过敏史",
            lines=2,
            placeholder="请输入患者过敏史..."
        )
        tcm_diagnosis = gr.Textbox(
            label="中医四诊",
            value="舌红苔黄腻，脉浮数",
            lines=2,
            placeholder="请输入中医四诊信息..."
        )
        physical_exam = gr.Textbox(
            label="体格检查",
            value="咽部充血，扁桃体I度肿大，双肺呼吸音粗",
            lines=2,
            placeholder="请输入体格检查结果..."
        )
        diagnosis_name = gr.Textbox(
            label="诊断名称",
            value="急性支气管炎",
            lines=2,
            placeholder="请输入诊断名称..."
        )
        tcm_syndrome = gr.Textbox(
            label="中医症候",
            value="风热犯肺证",
            lines=2,
            placeholder="请输入中医症候..."
        )

    return [
        present_illness, past_history, current_symptoms, allergy_history,
        tcm_diagnosis, physical_exam, diagnosis_name, tcm_syndrome
    ]


# 创建自定义聊天界面
def create_chat_interface():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🩺 中医智能诊疗助手")

        with gr.Row(equal_height=False):
            # 左侧：医疗信息输入
            with gr.Column(scale=1, min_width=400):
                medical_inputs = create_medical_inputs()
                gr.Markdown("---")
                gr.Markdown("### 使用说明")
                gr.Markdown("""
                1. 填写或修改左侧患者医疗信息
                2. 在右侧输入您的问题
                3. AI助手将基于患者信息提供专业建议
                """)

            # 右侧：聊天界面
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="中医诊疗对话",
                    height=500,
                    show_copy_button=True,
                    type="messages"
                )

                with gr.Row():
                    msg = gr.Textbox(
                        label="输入您的问题",
                        placeholder="请输入关于患者诊疗的问题...",
                        scale=4,
                        container=False,
                    )
                    submit_btn = gr.Button("发送", variant="primary", scale=1)

                with gr.Row():
                    clear_btn = gr.Button("清空对话", variant="secondary")

        # 处理函数
        def get_medical_info(*args):
            return {
                "present_illness": args[0],
                "past_history": args[1],
                "current_symptoms": args[2],
                "allergy_history": args[3],
                "tcm_diagnosis": args[4],
                "physical_exam": args[5],
                "diagnosis_name": args[6],
                "tcm_syndrome": args[7]
            }

        # 处理消息提交
        async def respond(message, chat_history, *medical_args):
            if not message.strip():
                yield chat_history, ""
                return

            medical_info = get_medical_info(*medical_args)

            # 添加用户消息到历史
            chat_history.append({"role": "user", "content": message})

            full_response = ""
            async for content in predict(message, chat_history[:-1], medical_info):
                full_response = content
                # 更新助手的最新回复
                if len(chat_history) > 0 and chat_history[-1]["role"] == "user":
                    chat_history.append(
                        {"role": "assistant", "content": full_response})
                else:
                    chat_history[-1] = {"role": "assistant",
                                        "content": full_response}
                yield chat_history, ""

        # 绑定事件
        submit_btn.click(
            respond,
            [msg, chatbot] + medical_inputs,
            [chatbot, msg]
        )

        msg.submit(
            respond,
            [msg, chatbot] + medical_inputs,
            [chatbot, msg]
        )

        clear_btn.click(
            lambda: [],
            None,
            chatbot
        )

    return demo


if __name__ == "__main__":
    demo = create_chat_interface()
    demo.launch(share=False)