# Trading Q&A chatbot: Mistral-7B QLoRA fine-tuning

Fine-tunes `mistralai/Mistral-7B-Instruct-v0.3` to answer US stock-market and trading questions,
then serves it in a Gradio app on Hugging Face Spaces.

- Model: [Xasan01/mistral-trading-chatbot](https://huggingface.co/Xasan01/mistral-trading-chatbot)
- Gradio app code: `hf_space/app.py` (the free CPU Space is offline: a 7B model needs a GPU, see below)

## Data
| Source | Pairs |
|---|---|
| Public dataset [`yymYYM/stock_trading_QA`](https://huggingface.co/datasets/yymYYM/stock_trading_QA) | 7,165 |
| My own Q&A pairs (`data/own_qa_pairs.json`, deduplicated) | 362 |
| **Total after removing duplicates** | **7,516** (90/10 train/validation split) |

## Training
QLoRA: 4-bit NF4 base model, LoRA r=32, α=64, dropout 0.05 on all linear layers;
lr 1e-4, effective batch 8, 2,000 steps (≈2.4 epochs), Mistral `[INST]` chat template, TRL `SFTTrainer`.

| Step | 200 | 400 | 800 | 1200 | **1600** | 1800 | 2000 |
|---|---|---|---|---|---|---|---|
| Train loss | 0.77 | 1.04 | 0.94 | 0.65 | 0.68 | 0.50 | 0.46 |
| Validation loss | 1.03 | 0.96 | 0.91 | 0.90 | **0.87** | 0.95 | 0.95 |

**Reading the curve:** validation loss was lowest at step 1600 and rose afterwards while training loss kept falling.
That's overfitting, so the step-1600 checkpoint is the better model. Next time I'd use `load_best_model_at_end=True` + early stopping.

## Limitations
- Loss is not answer quality. There's no held-out factual-accuracy eval yet.
- Not financial advice.

## Run it (needs a GPU with ~6 GB VRAM)
The HF repo holds a **LoRA adapter**, so load the base model first, then the adapter on top.
The tokenizer comes from the base model (the saved one needs transformers ≥ 5).
```python
# pip install transformers peft bitsandbytes accelerate
# Base model is gated: accept the license at huggingface.co/mistralai/Mistral-7B-Instruct-v0.3, then `huggingface-cli login`
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

base_id, adapter_id = "mistralai/Mistral-7B-Instruct-v0.3", "Xasan01/mistral-trading-chatbot"
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(base_id)
model = AutoModelForCausalLM.from_pretrained(base_id, quantization_config=bnb, device_map="auto")
model = PeftModel.from_pretrained(model, adapter_id)   # adds the fine-tuned LoRA weights

prompt = "[INST] What is the wash-sale rule? [/INST]"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=200, temperature=0.2, do_sample=True)
print(tokenizer.decode(out[0], skip_special_tokens=True).split("[/INST]")[-1].strip())
```
Works on a free Colab/Kaggle T4. Free CPU Spaces (16 GB RAM) can't hold the 7B model.
