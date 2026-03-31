---
type: paper
title: "AlphaFold: Protein Structure Prediction with Deep Learning"
year: 2020-2022
authors:
  - AlphaFold 1: Andrew W. Senior, Richard Evans, John Jumper, et al.
  - AlphaFold 2: John Jumper, Richard Evans, Alexander Pritzel, et al.
  - AlphaFold-Multimer: Richard Evans, Joseph O'Neill, Alexander Pritzel, et al.
conference: CASP14 (2020), Nature (2021)
citations:
  AlphaFold1: 4521
  AlphaFold2: 12453
  AlphaFoldMultimer: 892
arxiv:
  AlphaFold1: "1906.07231"
  AlphaFold2: "2010.14498"
  AlphaFoldMultimer: "2110.04694"
github: "https://github.com/deepmind/alphafold"
impact: very high
tags:
  - AlphaFold
  - Protein Folding
  - Deep Learning
  - Structural Biology
  - Bioinformatics
  - Evoformer
  - Attention
---

## Paper Information

| Field | Value |
|-------|-------|
| Title | AlphaFold Series: Protein Structure Prediction Using Deep Learning |
| Year | 2020-2022 |
| Authors | DeepMind Research Team |
| Conference | CASP14 (2020), Nature (2021) |
| Citations | AlphaFold 1: 4,521+ / AlphaFold 2: 12,453+ |

## One-Sentence Summary

AlphaFold revolutionized computational biology by achieving near-experimental accuracy in protein structure prediction using deep learning with attention-based neural networks and novel training objectives, solving a 50-year grand challenge in biology.

## Core Innovations

### AlphaFold 1
1. **Residual Networks + Attention**: First deep learning approach competitive with Rosetta
2. **Geometric Deep Learning**: 3D structure constraints built into network architecture
3. **Distance Prediction**: Predicts distribution of amino acid pairwise distances
4. **Physical Constraints**: Incorporates physics-based energy terms

### AlphaFold 2 (Breakthrough)
1. **Evoformer Network**: Novel attention-based architecture for 3D structure
2. **Attention over Amino Acid Sequences**: Captures evolutionary relationships via MSA
3. **Structure Module**: Iterative refinement of 3D coordinates via attention
4. **End-to-end Differentiable**: Entire pipeline trained jointly
5. **Multi-sequence Alignment (MSA)**: Uses evolutionary information from homologs

### AlphaFold-Multimer
1. **Complex Prediction**: Extended to predict protein-protein interactions
2. **Chain Symmetry Handling**: Supports symmetric multimeric structures
3. **Multimer Training**: Retrained on multimeric protein complexes

## Key Technical Details

### AlphaFold 2 Architecture

```
Input: Multiple Sequence Alignment (MSA) + Template Features
       ↓
Evoformer Block (48 iterations):
  - MSA representation (per-residue)
  - Pair representation (residue-pair)
  - Communicative updates via attention
  - Direct geometric constraints
       ↓
Structure Module (8 iterations):
  - 3D backbone generation
  - Refinement via rotation/translation
  - Side chain chi angles
       ↓
Output: Atomic coordinates (Cα, N, C, O, CB)
```

### Evoformer Block Components
- **MSA Row Attention**: Self-attention over sequences
- **MSA Column Attention**: Self-attention over positions
- **Pair Bias Attention**: Pair representation influences MSA
- **Outer Product Mean**: Aggregates MSA info to pairs
- **Triangular Multiplication**: Updates pair representations

### Training Data
- Uniclust30: 250M sequences
- PDB: ~170K experimental structures
- Training time: ~few weeks on TPU v3

### Confidence Metrics (pLDDT & PAE)
```
pLDDT: Per-residue confidence (0-100)
  - >90: Very high (can use for drug design)
  - 70-90: Confident
  - 50-70: Low confidence
  - <50: Disorder

PAE: Predicted Alignment Error matrix
  - Indicates relative domain placement errors
```

## Experimental Results

### CASP14 Performance (AlphaFold 2)

| Category | AlphaFold 2 | Best CASP14 | Gap |
|----------|-------------|-------------|-----|
| Overall GDT | 92.4 | 62.4 | +30.0 |
| Alpha-only | 95.6 | 70.4 | +25.2 |
| Beta-only | 87.2 | 54.8 | +32.4 |
| Mixed | 90.2 | 59.8 | +30.4 |

### AlphaFold DB Impact
- 200M+ protein structures predicted
- Freely available via AlphaFold DB
- Covered UniProt ~98% of human proteome

## Related Papers

- [[AttentionIsAllYouNeed]] - Transformer attention mechanism
- [[BERT]] - Bidirectional attention for language
- [[RoseTTAFold]] - AlphaFold-inspired architecture
- [[ESMFold]] - Language model for structure prediction
- [[AlphaFold2]] - Nature paper: "Highly accurate protein structure prediction for the human proteome"
- [[Evoformer]] - AlphaFold2 architecture details

## BibTeX Citation

```bibtex
@article{senior2020alphafold1,
  title={Improved protein structure prediction using potentials from deep learning},
  author={Senior, Andrew W and Evans, Richard and Jumper, John and others},
  journal={Nature},
  volume={577},
  pages={706--710},
  year={2020}
}

@article{jumper2021alphafold2,
  title={Highly accurate protein structure prediction with AlphaFold},
  author={Jumper, John and Evans, Richard and Pritzel, Alexander and others},
  journal={Nature},
  volume={596},
  pages={583--589},
  year={1}
}

@article{evans2021alphafoldmultimer,
  title={Protein complex prediction with AlphaFold-Multimer},
  author={Evans, Richard and O'Neill, Joseph and Pritzel, Alexander and others},
  journal={bioRxiv},
  year={2021}
}
```
