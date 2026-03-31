---
type: paper
title: "Attention Is All You Need"
year: 2017
authors:
  - Ashish Vaswani
  - Noam Shazeer
  - Niki Parmar
  - Jakob Uszkoreit
  - Llion Jones
  - Aidan N. Gomez
  - Lukasz Kaiser
  - Illia Polosukhin
conference: NeurIPS 2017
citations: 98542
arxiv: "1706.03762"
github: "https://github.com/tensorflow/tensor2tensor"
impact: high
tags:
  - Transformer
  - Attention Mechanism
  - Neural Machine Translation
  - Sequence Modeling
  - NLP
---

## Paper Information

| Field | Value |
|-------|-------|
| Title | Attention Is All You Need |
| Year | 2017 |
| Authors | Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin |
| Conference | NeurIPS 2017 |
| Citations | 98,542+ |
| ArXiv | arXiv:1706.03762 |
| GitHub | tensorflow/tensor2tensor |

## One-Sentence Summary

The Transformer architecture relies entirely on attention mechanisms, dispensing with recurrence and convolutions, to achieve state-of-the-art results in neural machine translation with dramatically improved parallelization and training efficiency.

## Core Innovations

1. **Pure Attention Mechanism**: Completely removes recurrence (RNN) and convolutions, using only self-attention layers
2. **Multi-Head Attention**: Parallel attention operations on different subspaces to capture various relationships
3. **Positional Encoding**: Sinusoidal encoding to incorporate sequence order information without recurrence
4. **Parallelizable Training**: Dramatically faster training due to reduced sequential computation
5. **Scaled Dot-Product Attention**: Attention function with scaling factor to prevent vanishing gradients

## Key Technical Details

### Architecture
- **Encoder-Decoder Structure**: 6-layer encoder and 6-layer decoder stacks
- **Model Dimensions**: d_model=512, d_ff=2048 (feed-forward)
- **Attention Heads**: h=8 parallel attention heads, each with d_k=d_v=64
- **Dropout**: 0.1 dropout rate during training

### Attention Formula
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

### Positional Encoding
```python
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### Training
- **Optimizer**: Adam with β_1=0.9, β_2=0.98, ε=10^-9
- **Learning Rate Schedule**: Warm-up with 4000 steps, then inverse square root decay
- **Batch Size**: 25,000 tokens per batch
- **Training Steps**: 100,000 steps on 8 P100 GPUs

## Experimental Results

| Task | Model | Score |
|------|-------|-------|
| WMT EN-DE | Transformer (big) | 28.4 BLEU |
| WMT EN-FR | Transformer (big) | 41.8 BLEU |
| WMT EN-DE | Transformer (base) | 25.8 BLEU |

### Key Findings
- Transformer (big) achieves 41.8 BLEU on WMT English-to-French translation
- Training time reduced by factor of 4 compared to Google's previous GNMT model
- Attention patterns are interpretable and show different head behaviors
- English-to-German task shows 2 BLEU improvement over previous best

## Related Papers

- [[BERT]] - Bidirectional Encoder Representations from Transformers
- [[GPT论文]] - GPT: Improving Language Understanding by Generative Pre-Training
- [[GPT-2]] - Language Models are Unsupervised Multitask Learners
- [[GPT-3]] - Language Models are Few-Shot Learners
- [[Transformer-XL]] - Attentive Language Models Beyond a Fixed-Length Context
- [[RoBERTa]] - A Robustly Optimized BERT Pretraining Approach
- [[XLNet]] - Generalized Autoregressive Pretraining for Language Understanding

## BibTeX Citation

```bibtex
@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, Lukasz and Polosukhin, Illia},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
```
