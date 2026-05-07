# MiniGPT — Decoder-Only Transformer Built From Scratch

A compact GPT-style language model implemented entirely in PyTorch from first principles.

This project demonstrates the complete pipeline for building a decoder-only transformer:
- dataset preparation
- tokenization
- causal masking
- multi-head self-attention
- transformer decoder blocks
- autoregressive training
- text generation

The implementation is intentionally educational and transparent, avoiding high-level abstractions wherever possible.

---

# Architecture Overview

```text
Input Text
   │
   ▼
Tokenizer
   │
   ▼
Token IDs
   │
   ▼
Token Embeddings + Positional Embeddings
   │
   ▼
Stack of Decoder Blocks
   │
   ├── Multi-Head Self Attention
   │       ├── Query Projection
   │       ├── Key Projection
   │       ├── Value Projection
   │       └── Causal Attention Mask
   │
   ├── Residual Connection
   ├── LayerNorm
   │
   ├── Feed Forward Network
   ├── Residual Connection
   └── LayerNorm
   │
   ▼
Final LayerNorm
   │
   ▼
Linear Vocabulary Projection
   │
   ▼
Next Token Probabilities
```

---

# Core Components

## 1. Dataset Pipeline

The model trains on Shakespeare text data.

### Dataset Flow

```text
Raw Shakespeare Text
        │
        ▼
Sliding Window Chunking
        │
        ▼
Tokenizer Encoding
        │
        ▼
Input IDs + Attention Masks
        │
        ▼
PyTorch Dataset / DataLoader
```

### Key Classes

#### `CharacterDataset`
Creates fixed-length text windows from raw text.

#### `PreTokenizedDataset`
Returns:
- input sequence
- causal attention mask
- target sequence

Targets are shifted by one token for autoregressive training.

---

# 2. Self Attention

The project implements scaled dot-product self-attention manually.

## Attention Formula

```math
Attention(Q,K,V) = softmax(QKᵀ / √d)V
```

### Internal Flow

```text
Input Embeddings
      │
      ├── Linear → Query
      ├── Linear → Key
      └── Linear → Value
              │
              ▼
        Attention Scores
              │
              ▼
        Causal Mask Applied
              │
              ▼
           Softmax
              │
              ▼
      Weighted Value Sum
```

---

# 3. Causal Masking

A lower triangular mask prevents tokens from attending to future tokens.

```text
Token 1 → can attend only Token 1
Token 2 → can attend Token 1,2
Token 3 → can attend Token 1,2,3
```

This preserves autoregressive generation behavior.

---

# 4. Multi-Head Attention

Multiple attention heads learn different relationships simultaneously.

## Head Flow

```text
Input
  │
  ├── Head 1 Attention
  ├── Head 2 Attention
  ├── Head 3 Attention
  └── ...
  │
  ▼
Concatenate Heads
  │
  ▼
Final Linear Projection
```

### Benefits
- parallel contextual understanding
- richer token relationships
- improved sequence modeling

---

# 5. Decoder Block

Each transformer block contains:

```text
Multi-Head Attention
        │
Residual Connection
        │
LayerNorm
        │
Feed Forward Network
        │
Residual Connection
        │
LayerNorm
```

### Feed Forward Network

```math
FFN(x) = Linear(ReLU(Linear(x)))
```

---

# 6. MiniGPT Model

The full model stacks multiple decoder blocks.

## Model Structure

```text
Embedding Layer
      │
Positional Embedding
      │
Dropout
      │
N × Decoder Blocks
      │
Final LayerNorm
      │
Linear Vocabulary Head
      │
Token Logits
```

### Current Configuration

| Component | Value |
|---|---|
| Embedding Dimension | 256 |
| Attention Heads | 8 |
| Decoder Layers | 6 |
| Vocabulary Size | 1000 |
| Architecture | Decoder-only Transformer |

---

# 7. Training Pipeline

## Training Steps

```text
Input Tokens
      │
      ▼
Forward Pass
      │
      ▼
Cross Entropy Loss
      │
      ▼
Backpropagation
      │
      ▼
Optimizer Step
      │
      ▼
Parameter Update
```

### Objective

The model learns:

```text
Predict next token given previous tokens
```

---

# 8. Text Generation

Generation is autoregressive.

## Generation Loop

```text
Prompt
   │
   ▼
Tokenizer Encode
   │
   ▼
Model Prediction
   │
   ▼
Sample Next Token
   │
   ▼
Append Token
   │
   ▼
Repeat
```

### Sampling Features
- temperature scaling
- top-k sampling
- causal context windowing

---

# Technical Highlights

## Implemented From Scratch
- scaled dot-product attention
- multi-head attention
- decoder blocks
- causal masking
- positional embeddings
- autoregressive generation

## Libraries Used
- PyTorch
- tokenizers
- matplotlib
- scikit-learn

---

# Why This Project Matters

This implementation helps understand:
- how GPT architectures function internally
- how attention mechanisms work mathematically
- why causal masking is necessary
- how autoregressive generation operates
- how transformer blocks are constructed

Rather than using pretrained abstractions, this project exposes the actual mechanics behind modern LLMs.

---

# Future Improvements

## Potential Upgrades
- rotary positional embeddings (RoPE)
- Flash Attention
- grouped query attention
- KV caching
- mixed precision training
- beam search decoding
- larger datasets
- distributed training
- tokenizer improvements
- inference optimization

---

# Example Use Cases

- educational transformer implementation
- experimentation platform
- research sandbox
- attention visualization
- architecture debugging
- small-scale language modeling

---

# Running The Project

## Install Dependencies

```bash
pip install torch tokenizers matplotlib scikit-learn tqdm
```

## Generate Text

```python
generate(
    model,
    tokenizer,
    prompt="To be or not to be",
    max_new_tokens=100
)
```

---

# Learning Outcomes

By building this project manually, you gain understanding of:
- tensor transformations
- attention mechanics
- transformer internals
- decoder-only architectures
- sequence modeling
- autoregressive learning
- language model inference

---

# Project Philosophy

The focus of MiniGPT is clarity over abstraction.

Every major transformer component is implemented explicitly so the training dynamics and architectural behavior remain understandable and modifiable.

This makes the project suitable for:
- students
- researchers
- ML engineers
- transformer experimentation
- architecture prototyping

---

# License

MIT License
