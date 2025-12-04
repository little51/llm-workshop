import torch
import gradio as gr
import soundfile as sf
import tempfile
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from cached_path import cached_path
import json
import numpy as np
from utils_infer_noffmpeg import (
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text
)
from f5_tts.model import DiT

DEFAULT_TTS_MODEL = "F5-TTS_v1"
DEFAULT_TTS_MODEL_CFG = [
    "models/SWivid/F5-TTS_Emilia-ZH-EN/model_1250000.safetensors",
    "models/SWivid/F5-TTS_Emilia-ZH-EN/vocab.txt",
    json.dumps(dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)),
]

# 初始化模型
def load_whisper_model():
    """加载Whisper语音识别模型"""
    model_id = "models/openai/whisper-large-v3-turbo"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, dtype=torch.float16, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor

def load_f5tts():
    """加载F5-TTS语音合成模型"""
    ckpt_path = str(cached_path(DEFAULT_TTS_MODEL_CFG[0]))
    F5TTS_model_cfg = json.loads(DEFAULT_TTS_MODEL_CFG[2])
    return load_model(DiT, F5TTS_model_cfg, ckpt_path)

# 全局变量
whisper_model = None
whisper_processor = None
f5tts_model = None
vocoder = None

def init_models():
    """初始化所有模型"""
    global whisper_model, whisper_processor, f5tts_model, vocoder
    print("加载Whisper语音识别模型...")
    whisper_model, whisper_processor = load_whisper_model()
    print("加载F5-TTS语音合成模型...")
    f5tts_model = load_f5tts()
    print("加载Vocoder...")
    vocoder = load_vocoder(is_local=True, local_path="models/charactr/vocos-mel-24khz")
    print("所有模型加载完成！")

def transcribe_audio(audio_file):
    """使用Whisper转录音频"""
    if not audio_file:
        return ""
    try:
        import librosa
        audio_data, _ = librosa.load(audio_file, sr=16000, mono=True)
        audio_data = audio_data.astype(np.float32) 
        inputs = whisper_processor(
            audio_data,
            sampling_rate=16000,
            return_tensors="pt",
            language="zh",
            task="transcribe"
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_dtype = next(whisper_model.parameters()).dtype
        processed_inputs = {}
        for key, value in inputs.items():
            if value.dtype == torch.float:  
                processed_inputs[key] = value.to(device=device, dtype=model_dtype)
            else: 
                processed_inputs[key] = value.to(device)
        with torch.no_grad():
            generated_ids = whisper_model.generate(**processed_inputs)
        return whisper_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    except Exception as e:
        print(f"语音识别失败: {e}")
        return ""

def synthesize_speech(ref_audio_file, ref_text, gen_text, seed=0, speed=1.0):
    """合成语音"""
    if not ref_audio_file or not ref_text.strip() or not gen_text.strip():
        return None, None, "请提供所有必要输入"
    try:
        # 设置随机种子
        torch.manual_seed(seed)
        # 预处理参考音频和文本
        ref_audio, ref_text_processed = preprocess_ref_audio_text(ref_audio_file, ref_text)
        # 合成语音
        final_wave, final_sample_rate, _ = infer_process(
            ref_audio,
            ref_text_processed,
            gen_text,
            f5tts_model,
            vocoder,
            cross_fade_duration=0.15,
            nfe_step=32,
            speed=speed,
            show_info=print,
            progress=None,
        )
        # 保存到临时文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
            sf.write(temp_path, final_wave, final_sample_rate)
        return temp_path, final_sample_rate, "合成成功"
    except Exception as e:
        print(f"语音合成失败: {e}")
        return None, None, f"合成失败: {str(e)}"

def process_full_pipeline(audio_file, gen_text, use_auto_transcribe=True, seed=0, speed=1.0):
    """完整的处理流程：语音识别 + 语音合成"""
    # 步骤1: 语音识别
    if use_auto_transcribe and audio_file:
        ref_text = transcribe_audio(audio_file)
    else:
        ref_text = ""
    # 步骤2: 语音合成
    if audio_file and gen_text.strip():
        audio_path, sample_rate, message = synthesize_speech(audio_file, ref_text, gen_text, seed, speed)
        return ref_text, audio_path, message
    else:
        return ref_text, None, "请提供参考音频和要生成的文本"

# 创建Gradio界面
def create_interface():
    with gr.Blocks(title="语音识别与合成系统") as app:
        gr.Markdown("# 🎤 语音识别与合成系统")
        gr.Markdown("上传参考音频，系统将自动识别语音内容，并根据输入的文本生成新的语音。")
        with gr.Row():
            with gr.Column():
                # 输入部分
                audio_input = gr.Audio(
                    label="上传参考音频",
                    type="filepath",
                    sources=["upload", "microphone"],
                )
                
                auto_transcribe = gr.Checkbox(
                    label="自动语音识别",
                    value=True,
                    info="勾选后自动识别参考音频的内容"
                )
                
                gen_text_input = gr.Textbox(
                    label="输入要生成的文本",
                    placeholder="请输入要转换为语音的文本...",
                    lines=5,
                )
                
                with gr.Accordion("高级设置", open=False):
                    seed_input = gr.Number(
                        label="随机种子",
                        value=0,
                        minimum=0,
                        maximum=2147483647,
                        step=1,
                        info="设置随机种子以确保可重复性"
                    )
                    
                    speed_slider = gr.Slider(
                        label="语速",
                        minimum=0.5,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        info="调整语音的播放速度"
                    )
                
                process_btn = gr.Button("🚀 开始处理", variant="primary")
                
            with gr.Column():
                # 输出部分
                transcribe_result = gr.Textbox(
                    label="语音识别结果",
                    interactive=False,
                    lines=3,
                )
                
                audio_output = gr.Audio(
                    label="合成的语音",
                    autoplay=True,
                )
                
                status_output = gr.Textbox(
                    label="处理状态",
                    interactive=False,
                )
        
        # 按钮点击事件
        process_btn.click(
            fn=process_full_pipeline,
            inputs=[
                audio_input,
                gen_text_input,
                auto_transcribe,
                seed_input,
                speed_slider,
            ],
            outputs=[
                transcribe_result,
                audio_output,
                status_output,
            ]
        )
        
        # 清除输入事件
        audio_input.clear(
            lambda: ["", None, ""],
            outputs=[transcribe_result, audio_output, status_output]
        )
    
    return app

# 主函数
def main():
    print("初始化模型...")
    init_models()
    print("创建Gradio界面...")
    app = create_interface()
    # 启动应用
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )

if __name__ == "__main__":
    main()