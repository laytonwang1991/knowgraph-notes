---
type: paper
title: "PaLM: Pathway Language Model"
year: 2023
authors:
  - Aakanksha Chowdhery
  - Sharan Narang
  - Jacob Devlin
  - Maarten Bosma
  - Ankur Goyal
  - Warren Lee
  - Jiwei Liu
  - Wei Li
  - Katie Millican
  - Armin W. Thomas
  - Sasha T. R. Phillip
  - Emily S. Houlsby
  - Qian
  - Saurav
  - The
  - Bradley
  - Google
conference: ICML 2023
citations: 3421
arxiv: "2204.02311"
github: "https://github.com/google-research/totx"
impact: high
tags:
  - PaLM
  - Large Language Model
  - Pathway
  - 540B Parameters
  - Chain-of-Thought
  - Few-shot
  - NLP
---

## Paper Information

| Field | Value |
|-------|-------|
| Title | PaLM: Scaling Language Modeling with Pathways |
| Year | 2023 |
| Authors | Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, et al. (Google Research) |
| Conference | ICML 2023 |
| Citations | 3,421+ |
| ArXiv | arXiv:2204.02311 |

## One-Sentence Summary

PaLM is a 540-billion parameter Transformer language model trained on 780 billion tokens using Google's Pathways system, demonstrating unprecedented few-shot performance through scaling and showcasing emergent capabilities like chain-of-thought reasoning and multi-step arithmetic.

## Core Innovations

1. **Pathways Distributed Training**: First large-scale model trained with Google's Pathways system across thousands of TPU pods
2. **Scaling to 540B Parameters**: 15x larger than previous Google models (T5-xxl was 11B)
3. **Efficient Attention**: Uses "query-key normalization" and multi-query attention
4. **Emergent Abilities**: Chain-of-thought reasoning emerges at scale (62B+)
5. **Bilingual Pre-training**: Mix of web documents (50% English, 50% multilingual)
6. **Joke Explanation**: Novel emergent capability demonstrated at scale

## Key Technical Details

### Model Architecture

| Parameter | Value |
|-----------|-------|
| Parameters | 540B |
| Layers | 118 |
| Hidden Size | 18,432 |
| Attention Heads | 48 |
| Head Dimension | 256 |
| FFN Hidden | 65,536 |
| Attention | Multi-query (1 key-value head) |
| Vocab Size | 256k tokenizers |
| Context Length | 2,048 tokens |

### Pathways System
```
Traditional:
- Data parallel: replicate model, split data
- Pipeline parallel: split layers across devices
- Tensor parallel: split tensors within layers

Pathways:
- 6,144 TPU v4 chips for training
- All-reduce operations across 2D mesh
- Asynchronous dispatch of computation
- Virtual device abstraction
```

### Training Configuration
- **Tokens**: 780B tokens (one epoch of training data)
- **Batch Size**: 4,096 sequences (8,192 tokens each)
- **Training Time**: 60 days on 6,144 TPU v4 chips
- **FLOPs**: 2.55e24 total training compute

### Pre-training Data
| Dataset | Tokens | Proportion |
|---------|--------|------------|
| Web Documents | 390B | 50% |
| Multilingual | 390B | 50% |
| Books | 39B | 5% |
| GitHub | 13B | 1.6% |
| Wikipedia | 4B | 0.5% |

### Novel Techniques

#### Multi-Query Attention
```
Standard: Each head has separate K, V matrices
Multi-query: Share K, V across all heads
  → Reduces memory for keys/values
  → Only query matrices per head
  → Trade-off: slight quality reduction for efficiency
```

#### Query-Key Normalization
```
- Apply layer norm to Q and K before attention
- Prevents training instability at scale
- Improves convergence
```

## Experimental Results

### Benchmark Performance

| Benchmark | PaLM 8B | PaLM 62B | PaLM 540B | SOTA |
|-----------|---------|----------|-----------|------|
| MMLU (57 tasks) | 38.2 | 62.0 | **76.6** | 71.8 |
| Big-Bench Hard | 34.6 | 55.6 | **68.5** | 55.0 |
| GSM8K (math) | 7.2 | 40.1 | **56.5** | 41.8 |
| HumanEval (code) | 3.6 | 21.7 | **37.6** | 30.8 |

### Chain-of-Thought Emergence
```
Scale Analysis (PaLM models):
- 8B: Chain-of-thought does NOT help
- 62B: Chain-of-thought marginally helps
- 540B: Chain-of-thought SIGNIFICANTLY helps

Example: Word in Context task
8B: 53.4% (no CoT improvement)
62B: 66.7% (modest improvement)
540B: 86.4% (major improvement)
```

### Reasoning Capabilities
- Multi-step arithmetic: 57% on 5-digit addition
- Chronological reasoning: 83% accuracy
- Joke explanation: Novel emergent capability
- Code generation: Competitive with specialized models

## Related Papers

- [[GPT-3]] - Language Models are Few-Shot Learners (175B)
- [[Chinchilla]] - Training Compute-Optimal Large Language Models
- [[Gopher]] - Language Models for Research
- [[Megatron-Turing NLG]] - 530B Parameter Model
- [[Pathways]] - Google's Distributed Training System
- [[Flan-PaLM]] - Instruction-tuned PaLM
- [[PaLM-E]] - PaLM with Embodied Multimodal Reasoning
- [[T5]] - Text-to-Text Transfer Transformer

## BibTeX Citation

```bibtex
@article{chowdhery2023palm,
  title={PaLM: Scaling Language Modeling with Pathways},
  author={Chowdhery, Aakanksha and Narang, Sharan and Devlin, Jacob and Bosma, Maarten and Goyal, Ankur and Lee, Warren and Liu, Jiwei and Li, Wei and Millican, Katie and Thomas, Armin W and others},
  journal={arXiv preprint arXiv:2204.02311},
  year={2023}
}
```
