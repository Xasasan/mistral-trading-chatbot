# Trading Q&A chatbot: Mistral-7B QLoRA fine-tuning

Fine-tunes `mistralai/Mistral-7B-Instruct-v0.3` to answer US stock-market and trading questions,
then serves it in a Gradio app on Hugging Face Spaces.

- Model: [Xasan01/mistral-trading-chatbot](https://huggingface.co/Xasan01/mistral-trading-chatbot)
- Demo: [Space: Xasan01/Trading_chatbot](https://huggingface.co/spaces/Xasan01/Trading_chatbot) (`hf_space/app.py`)

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
