---
type: paper
title: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
year: 2018
authors:
  - Jacob Devlin
  - Ming-Wei Chang
  - Kenton Lee
  - Kristina Toutanova
conference: NAACL 2019
citations: 87234
arxiv: "1810.04805"
github: "https://github.com/google-research/bert"
impact: high
tags:
  - BERT
  - Pre-training
  - Bidirectional
  - Transformer
  - NLP
  - Language Model
---

## Paper Information

| Field | Value |
|-------|-------|
| Title | BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding |
| Year | 2018 |
| Authors | Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova |
| Conference | NAACL 2019 |
| Citations | 87,234+ |
| ArXiv | arXiv:1810.04805 |
| GitHub | google-research/bert |

## One-Sentence Summary

BERT introduces bidirectional Transformer encoders pre-trained with masked language modeling and next sentence prediction, enabling fine-tuning for a wide range of NLP tasks with minimal task-specific architecture modifications.

## Core Innovations

1. **Bidirectional Pre-training**: First large-scale bidirectional Transformer encoder, capturing context from both directions
2. **Masked Language Model (MLM)**: Randomly masks 15% of tokens and predicts the original vocabulary ID
3. **Next Sentence Prediction (NSP)**: Pre-training task to capture sentence-level relationships for QA and NLI
4. **Task-Agnostic Architecture**: Same pre-trained model fine-tuned with minimal output layers for downstream tasks
5. **Fine-tuning Approach**: No feature engineering required; entire model is fine-tuned end-to-end

## Key Technical Details

### Pre-training Architecture
- **Model**: Transformer Encoder (24 layers for BERT-Large, 12 for BERT-Base)
- **Hidden Size**: 1024 (Large), 768 (Base)
- **Attention Heads**: 16 (Large), 12 (Base)
- **Parameters**: 340M (Large), 110M (Base)

### Pre-training Tasks
```
1. Masked Language Model (MLM):
   - Mask 15% of all tokens [80% -> [MASK], 10% -> random token, 10% -> unchanged]
   - Predict original token

2. Next Sentence Prediction (NSP):
   - Given two sentences A and B, predict if B follows A
   - 50% positive, 50% negative examples
```

### Fine-tuning
- Pre-trained weights used to initialize encoder
- Output layer added for specific task
- Fine-tuned for 3-4 epochs with learning rate 2e-5 to 5e-5

### Pre-training Data
- BooksCorpus (800M words)
- English Wikipedia (2,500M words)
- Training steps: 1,000,000 (BERT-Large)
- Batch size: 256 sequences
- Sequence length: 512 tokens

## Experimental Results

| Task | Dataset | BERT-Base | BERT-Large | Previous Best |
|------|---------|-----------|------------|---------------|
| SQuAD 1.1 | Dev F1 | 88.5 | **90.9** | 85.1 |
| SQuAD 2.0 | Dev F1 | 77.5 | **83.1** | 74.0 |
| MNLI | Dev Acc | 84.6 | **86.7** | 82.9 |
| SST-2 | Dev Acc | 93.5 | **94.9** | 93.2 |
| CoLA | Dev MCC | 60.6 | **63.6** | 45.6 |
| STS-B | Dev PC | 79.9 | **83.1** | 76.0 |

### Key Findings
- BERT achieved state-of-the-art on 11 NLP benchmarks at time of release
- Ablation studies show both MLM and NSP contribute to performance
- Bidirectional encoding crucial for deep contextual understanding
- Fine-tuning outperforms feature-based approaches (ELMo, GPT-1)

## Related Papers

- [[AttentionIsAllYouNeed]] - Transformer original architecture
- [[GPT论文]] - GPT-1: Generative Pre-Training
- [[GPT-2]] - Language Models are Unsupervised Multitask Learners
- [[GPT-3]] - Language Models are Few-Shot Learners
- [[RoBERTa]] - Robustly Optimized BERT Pretraining
- [[ALBERT]] - A Lite BERT for Self-supervised Learning
- [[ELECTRA]] - Pre-training Text Encoders as Discriminators Rather Than Generators
- [[XLNet]] - Generalized Autoregressive Pretraining

## BibTeX Citation

```bibtex
@article{devlin2018bert,
  title={Bert: Pre-training of deep bidirectional transformers for language understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  journal={arXiv preprint arXiv:1810.04805},
  year={2018}
}
```
