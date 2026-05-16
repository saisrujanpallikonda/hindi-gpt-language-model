# ============================================================
# HINDI GPT LANGUAGE MODEL — COMPLETE CODE
# Run on Kaggle with T4 GPU
# ============================================================

import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import re
import json
import os
from datasets import load_dataset

# ============================================================
# STEP 1 — DOWNLOAD DATASET
# ============================================================

print("📥 Downloading Hindi Wikipedia...")
dataset = load_dataset("wikimedia/wikipedia", "20231101.hi", split="train")

with open("hindi_corpus.txt", "w", encoding="utf-8") as f:
    for item in dataset:
        f.write(item["text"] + "\n")

print(f"✅ Downloaded {len(dataset)} articles")

# ============================================================
# STEP 2 — CLEAN CORPUS
# ============================================================

print("\n🧹 Cleaning corpus...")
devanagari_pattern = re.compile(r'^[\u0900-\u097F]+$')

def is_hindi_word(word):
    return devanagari_pattern.match(word) is not None

cleaned_lines = []
with open("hindi_corpus.txt", "r", encoding="utf-8") as f:
    for line in f:
        words       = line.strip().split()
        hindi_words = [w for w in words if is_hindi_word(w)]
        if hindi_words:
            cleaned_lines.append(" ".join(hindi_words))

with open("oscar_hi_cleaned.txt", "w", encoding="utf-8") as f:
    for line in cleaned_lines:
        f.write(line + "\n")

print(f"✅ Cleaned {len(cleaned_lines)} lines")

# ============================================================
# STEP 3 — BUILD VOCABULARY
# ============================================================

print("\n📚 Building vocabulary...")
with open("oscar_hi_cleaned.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars      = sorted(list(set(text)))
vocab_size = len(chars)
stoi       = {ch:i for i,ch in enumerate(chars)}
itos       = {i:ch for i,ch in enumerate(chars)}
encode     = lambda s: [stoi[c] for c in s if c in stoi]
decode     = lambda l: ''.join([itos[i] for i in l])

print(f"✅ Vocab size: {vocab_size}")

# ============================================================
# STEP 4 — ENCODE AND SPLIT
# ============================================================

print("\n✂️ Splitting data...")
data       = torch.tensor(encode(text), dtype=torch.long)
n          = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]

print(f"✅ Train: {len(train_data):,} | Val: {len(val_data):,}")

# ============================================================
# STEP 5 — BATCH LOADER
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n✅ Using device: {device}")

def get_batch(split, batch_size, block_size):
    d  = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x  = torch.stack([d[i:i+block_size] for i in ix])
    y  = torch.stack([d[i+1:i+block_size+1] for i in ix])
    return x, y

@torch.no_grad()
def evaluate_perplexity(model, block_size, batch_size):
    model.eval()
    losses = []
    for _ in range(200):
        xb, yb  = get_batch('val', batch_size, block_size)
        xb, yb  = xb.to(device), yb.to(device)
        _, loss = model(xb, yb)
        losses.append(loss.item())
    avg_loss   = sum(losses) / len(losses)
    perplexity = math.exp(avg_loss)
    model.train()
    return avg_loss, perplexity

# ============================================================
# STEP 6 — BIGRAM MODEL (BASELINE)
# ============================================================

print("\n" + "="*60)
print("BIGRAM MODEL — BASELINE")
print("="*60)

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits  = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss    = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _   = self(idx)
            logits      = logits[:, -1, :]
            probs       = F.softmax(logits, dim=-1)
            idx_next    = torch.multinomial(probs, num_samples=1)
            idx         = torch.cat((idx, idx_next), dim=1)
        return idx

# Bigram hyperparameters
BIGRAM_BATCH_SIZE  = 32
BIGRAM_BLOCK_SIZE  = 8
BIGRAM_STEPS       = 10000
BIGRAM_LR          = 1e-3

bigram_model     = BigramLanguageModel(vocab_size).to(device)
bigram_optimizer = torch.optim.AdamW(bigram_model.parameters(), lr=BIGRAM_LR)

print(f"Parameters: {sum(p.numel() for p in bigram_model.parameters())/1e6:.2f}M")
print(f"Training for {BIGRAM_STEPS} steps...\n")

bigram_ppl_log = []

for step in range(BIGRAM_STEPS):
    xb, yb         = get_batch('train', BIGRAM_BATCH_SIZE, BIGRAM_BLOCK_SIZE)
    xb, yb         = xb.to(device), yb.to(device)
    logits, loss   = bigram_model(xb, yb)
    bigram_optimizer.zero_grad()
    loss.backward()
    bigram_optimizer.step()

    if step % 1000 == 0:
        val_loss, ppl = evaluate_perplexity(bigram_model, BIGRAM_BLOCK_SIZE, BIGRAM_BATCH_SIZE)
        bigram_ppl_log.append({'step': step, 'ppl': ppl})
        print(f"Step {step:5d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Perplexity: {ppl:.2f}")

bigram_final_ppl = bigram_ppl_log[-1]['ppl']
print(f"\n✅ Bigram Final Perplexity: {bigram_final_ppl:.4f}")

# ============================================================
# STEP 7 — GPT MODEL (TRANSFORMER WITH ATTENTION)
# ============================================================

print("\n" + "="*60)
print("GPT MODEL — TRANSFORMER WITH MULTI-HEAD ATTENTION")
print("="*60)

# GPT hyperparameters
GPT_BATCH_SIZE  = 32
GPT_BLOCK_SIZE  = 64
GPT_STEPS       = 20000
GPT_LR          = 3e-4
N_EMBD          = 128
N_HEAD          = 4
N_LAYER         = 4
DROPOUT         = 0.1

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key     = nn.Linear(N_EMBD, head_size, bias=False)
        self.query   = nn.Linear(N_EMBD, head_size, bias=False)
        self.value   = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(GPT_BLOCK_SIZE, GPT_BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        k       = self.key(x)
        q       = self.query(x)
        wei     = q @ k.transpose(-2,-1) * C**-0.5
        wei     = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei     = F.softmax(wei, dim=-1)
        wei     = self.dropout(wei)
        return wei @ self.value(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads   = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj    = nn.Linear(N_EMBD, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa   = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding    = nn.Embedding(vocab_size, N_EMBD)
        self.position_embedding = nn.Embedding(GPT_BLOCK_SIZE, N_EMBD)
        self.blocks             = nn.Sequential(*[Block(N_EMBD, N_HEAD) for _ in range(N_LAYER)])
        self.ln_f               = nn.LayerNorm(N_EMBD)
        self.lm_head            = nn.Linear(N_EMBD, vocab_size)

    def forward(self, idx, targets=None):
        B, T    = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=device))
        x       = tok_emb + pos_emb
        x       = self.blocks(x)
        x       = self.ln_f(x)
        logits  = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits  = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss    = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond  = idx[:, -GPT_BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits    = logits[:, -1, :]
            probs     = F.softmax(logits, dim=-1)
            idx_next  = torch.multinomial(probs, num_samples=1)
            idx       = torch.cat((idx, idx_next), dim=1)
        return idx

gpt_model     = GPTLanguageModel().to(device)
gpt_optimizer = torch.optim.AdamW(gpt_model.parameters(), lr=GPT_LR)

print(f"Parameters: {sum(p.numel() for p in gpt_model.parameters())/1e6:.2f}M")
print(f"Training for {GPT_STEPS} steps...\n")

gpt_ppl_log = []

for step in range(GPT_STEPS):
    xb, yb       = get_batch('train', GPT_BATCH_SIZE, GPT_BLOCK_SIZE)
    xb, yb       = xb.to(device), yb.to(device)
    logits, loss = gpt_model(xb, yb)
    gpt_optimizer.zero_grad()
    loss.backward()
    gpt_optimizer.step()

    if step % 2000 == 0:
        val_loss, ppl = evaluate_perplexity(gpt_model, GPT_BLOCK_SIZE, GPT_BATCH_SIZE)
        gpt_ppl_log.append({'step': step, 'ppl': ppl})
        print(f"Step {step:5d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Perplexity: {ppl:.2f}")

gpt_final_ppl = gpt_ppl_log[-1]['ppl']
print(f"\n✅ GPT Final Perplexity: {gpt_final_ppl:.4f}")

# ============================================================
# STEP 8 — FINAL COMPARISON
# ============================================================

print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
print(f"Bigram Model  → Perplexity: {bigram_final_ppl:.2f}")
print(f"GPT Model     → Perplexity: {gpt_final_ppl:.2f}")
print(f"Improvement   → {((bigram_final_ppl - gpt_final_ppl) / bigram_final_ppl * 100):.1f}%")

# ============================================================
# STEP 9 — SAVE EVERYTHING
# ============================================================

print("\n💾 Saving all files...")
os.makedirs("/kaggle/working/hindi_gpt", exist_ok=True)

# Save GPT model
torch.save({
    'model_state_dict': gpt_model.state_dict(),
    'vocab_size': vocab_size,
    'chars': chars,
    'perplexity': gpt_final_ppl,
}, "/kaggle/working/hindi_gpt/hindi_gpt_final.pt")
