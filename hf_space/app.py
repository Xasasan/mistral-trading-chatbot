import gradio as gr
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("Xasan01/mistral-trading-chatbot")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.chat_template = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{% if (message['role'] == 'system') %}{{ '[INST] ' + message['content'] + '\n\n' }}{% endif %}"
    "{% if (message['role'] == 'user') %}{{ message['content'] + ' [/INST] ' }}{% endif %}"
    "{% if (message['role'] == 'assistant') %}{{ message['content'] + eos_token }}{% endif %}"
    "{% endfor %}"
)

model = AutoModelForCausalLM.from_pretrained(
    "Xasan01/mistral-trading-chatbot",
    torch_dtype=torch.float32,  # CPU needs float32
    use_fast=False,
    device_map="cpu",
)

def chat(message, history):
    messages = [
        {"role": "system", "content": "You are an expert trading assistant specializing in US stock market analysis."},
        {"role": "user", "content": message}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=400,
        do_sample=True,
        temperature=0.2,
        top_p=0.9,
        repetition_penalty=1.2,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("[/INST]")[-1].strip()
    return response

gr.ChatInterface(
    fn=chat,
    title="📈 Trading Assistant",
    description="Ask me anything about stocks, technical analysis, RSI, MACD, and more.",
    examples=[
        "What is RSI and how do I use it?",
        "Explain FIBONACHI with an example",
        "What is the difference between support and resistance?",
    ],
    theme=gr.themes.Soft()
).launch()
