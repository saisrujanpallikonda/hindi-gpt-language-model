# Hindi GPT Language Model 🇮🇳

A transformer-based GPT language model built **from scratch** for Hindi text generation, trained on 163,000+ Hindi Wikipedia articles. This project compares a Bigram baseline against a full GPT architecture with multi-head self-attention.

---

## 📊 Results

| Model | Perplexity |
|-------|-----------|
| Bigram (baseline) | 13.35 |
| **GPT with Attention** | **4.97** ✅ |

---

## 🏗️ Architecture

| Hyperparameter | Value |
|----------------|-------|
| Layers | 4 |
| Attention Heads | 4 |
| Embedding Dimension | 128 |
| Context Length | 64 |
| Total Parameters | 0.83M |
| Dropout | 0.1 |

---

## 📉 Training Progress

| Step | Perplexity |
|------|-----------|
| 0 | 105.30 |
| 2000 | 8.04 |
| 4000 | 6.55 |
| 6000 | 5.93 |
| 8000 | 5.59 |
| 10000 | 5.39 |
| 20000 | **4.97** ✅ |

---

## 🚀 Quick Start

**1. Clone the repo**
```bash
git clone https://github.com/saisrujanpallikonda/hindi-gpt-language-model.git
cd hindi-gpt-language-model
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download dataset**
```python
from datasets import load_dataset
dataset = load_dataset("wikimedia/wikipedia", "20231101.hi", split="train")
with open("hindi_corpus.txt", "w", encoding="utf-8") as f:
    for item in dataset:
        f.write(item["text"] + "\n")
```

**4. Run the notebook**

Open `notebooks/03_gpt_model.ipynb` — recommended to run on Kaggle free T4 GPU (~35–40 mins)

---

## 🛠️ Tech Stack

- Python 3.12
- PyTorch 2.10
- HuggingFace Datasets
- Kaggle GPU (T4)


---

## 📄 License

MIT License
