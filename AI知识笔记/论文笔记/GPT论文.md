---
type: paper
title: "GPT Series: GPT-1, GPT-2, and GPT-3 - Language Model Pretraining"
year: 2018-2020
authors:
  - GPT-1: Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever
  - GPT-2: Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever
  - GPT-3: Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, et al.
conference: GPT-1: 2018, GPT-2: 2019, GPT-3: NeurIPS 2020
citations:
  GPT-1: 8921
  GPT-2: 7652
  GPT-3: 18432
arxiv:
  GPT-1: "1801.10146"
  GPT-2: "1909.08053"
  GPT-3: "2005.14165"
github: "https://github.com/openai/gpt-2"
impact: high
tags:
  - GPT
  - Language Model
  - Pre-training
  - Transformer Decoder
  - In-context Learning
  - NLP
  - Few-shot Learning
---

## Paper Information

| Field | Value |
|-------|-------|
| Title | GPT Series Overview: GPT-1 (2018), GPT-2 (2019), GPT-3 (2020) |
| Year | 2018-2020 |
| Authors | OpenAI Research Team |
| Conference | Various |
| Citations | GPT-1: 8,921+ / GPT-2: 7,652+ / GPT-3: 18,432+ |

## One-Sentence Summary

The GPT series demonstrates that large-scale language model pre-training on diverse text, followed by task-specific fine-tuning (GPT-1) or in-context learning (GPT-2/GPT-3), achieves strong performance across a broad range of natural language tasks.

## Core Innovations

### GPT-1: Foundation Model Approach
1. **Generative Pre-training + Discriminative Fine-tuning**: First to show transfer learning works at scale
2. **Unidirectional Transformer Decoder**: Left-to-right attention architecture
3. **Task-agnostic fine-tuning**: Same architecture fine-tuned for diverse downstream tasks

### GPT-2: Zero-shot Transfer
1. **Zero-shot Task Transfer**: Model performs tasks without gradient updates
2. **WebText Dataset**: 40GB high-quality web scraped text
3. **Larger Scale**: 1.5B parameters vs 117M in GPT-1
4. **In-context Learning**: Uses language model prefix to condition behavior

### GPT-3: Few-shot Learning
1. **Few-shot In-context Learning**: Learns from demonstrations in context
2. **Massive Scale**: 175B parameters (100x larger than GPT-2)
3. **Diverse Benchmarks**: 45+ tasks tested
4. **CoCoA**: Consistent Attentional Contexts approach

## Key Technical Details

### Architecture Comparison

| Model | Layers | Hidden Size | Heads | Parameters |
|-------|--------|-------------|-------|------------|
| GPT-1 | 12 | 768 | 12 | 117M |
| GPT-2 | 48 | 1600 | 25 | 1.5B |
| GPT-3 | 96 | 12288 | 96 | 175B |

### GPT-1 Pre-training
- BooksCorpus dataset (7,000+ books)
- Max sequence length: 512
- Learning rate: 2.5e-4 with linear decay
- 100K steps, batch size 64

### GPT-2 Pre-training
- WebText: 40GB text from Reddit outbound links
- 8M web pages
- Longer context: 1024 tokens
- No fine-tuning required for downstream tasks

### GPT-3 Pre-training
- High-quality datasets: CommonCrawl, WebText, Books, Wikipedia
- 300B tokens total training text
- Context length: 2048 tokens
- Few-shot, one-shot, and zero-shot evaluation

### Transformer Decoder
```
GPT uses masked self-attention (causal attention):
Attention(Q,K,V) = softmax(QK^T / √d_k + mask)V
```

## Experimental Results

### GPT-1 Fine-tuning Performance

| Task | GPT-1 | Best Previous |
|------|-------|---------------|
| SNLI | 89.9% | 88.4% |
| MultiNLI | 82.9% | 74.0% |
| SST-2 | 91.2% | 93.2% |
| RTE | 59.5% | 70.1% |

### GPT-2 Zero-shot Performance

| Task | GPT-2 (117M) | GPT-2 (1.5B) | Previous SOTA |
|------|--------------|--------------|---------------|
| Lambada | 19.7% | **52.7%** | 45.9% |
| Winograd | 49.2% | **70.7%** | 63.7% |
| CoQA | 35.5 | **55.0** | 47.5 |

### GPT-3 Few-shot Performance

| Task | SOTA (supervised) | GPT-3 Few-shot |
|------|-------------------|----------------|
| SuperGLUE | 89.9 | **71.8** (10-shot) |
| TriviaQA | 68.9 | **64.3** (k=64) |
| LAMBADA | 68.0 | **76.2** (k=10) |

## Related Papers

- [[AttentionIsAllYouNeed]] - Transformer architecture
- [[BERT]] - Bidirectional pre-training approach
- [[T5]] - Text-to-Text Transfer Transformer
- [[PaLM]] - Pathway Language Model
- [[InstructGPT]] - Training Language Models to Follow Instructions
- [[ChatGPT]] - Optimizing Language Models for Dialogue
- [[GPT-4]] - GPT-4 Technical Report

## BibTeX Citation

```bibtex
@article{radford2018gpt1,
  title={Improving language understanding by generative pre-training},
  author={Radford, Alec and Narasimhan, Karthik and Salimans, Tim and Sutskever, Ilya},
  journal={OpenAI Blog},
  year={2018}
}

@article{radford2019gpt2,
  title={Language models are unsupervised multitask learners},
  author={Radford, Alec and Wu, Jeffrey and Child, Rewon and Luan, David and Amodei, Dario and Sutskever, Ilya},
  journal={OpenAI Blog},
  year={2019}
}

@article{brown2020gpt3,
  title={Language models are few-shot learners},
  author={Brown, Tom B and Mann, Benjamin and Ryder, Nick and Subbiah, Melanie and others},
  journal={Advances in neural information processing systems},
  volume={33},
  pages={1877--1901},
  year={2020}
}
```
