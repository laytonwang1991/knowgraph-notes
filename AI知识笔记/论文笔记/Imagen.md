---
type: paper
title: "Imagen: Photo-Realistic Image Generation with Diffusion Models"
year: 2022
authors:
  - Chitwan Saharia
  - William Chan
  - Saurabh Saxena
  - Lala Li
  - Jay Whang
  - Emily Denton
  - Seyed Kamyar Seyed Ghasemipour
  - Burcu Karagol Ayan
  - S. Sara Mahdavi
  - Rapha Gontijo Lopes
  - Tim Salimans
  - Jonathan Ho
  - David J. Fleet
  - Mohammad Norouzi
conference: arXiv 2022 (ICML 2022)
citations: 4523
arxiv: "2205.11487"
github: "https://github.com/google/lm-definitions"
impact: high
tags:
  - Imagen
  - Diffusion Models
  - Text-to-Image
  - Large Language Models
  - Image Generation
  - AI Art
---

## Paper Information

| Field | Value |
|-------|-------|
| Title | Imagen: Photo-Realistic Image Generation with Diffusion Models |
| Year | 2022 |
| Authors | Chitwan Saharia, William Chan, Saurabh Saxena, et al. (Google Research) |
| Conference | arXiv (ICML 2022) |
| Citations | 4,523+ |
| ArXiv | arXiv:2205.11487 |

## One-Sentence Summary

Imagen combines a large frozen language model (T5) with cascaded diffusion models to achieve unprecedented photorealism and text-image alignment in text-to-image generation, using novel techniques like dynamic thresholding and efficient U-Net architectures.

## Core Innovations

1. **Frozen LLM for Text Encoding**: Uses T5-XXL as text encoder, showing LLM knowledge transfers to image generation
2. **Cascaded Diffusion Architecture**: Progressive super-resolution from 64x64 to 256x256 to 1024x1024
3. **Dynamic Thresholding**: Novel classifier-free guidance technique for high-resolution generation
4. **T5 vs CLIP Comparison**: Demonstrates frozen LLM outperforms CLIP for text understanding
5. **Efficient U-Net**: Memory-efficient attention mechanisms for high-resolution synthesis
6. **Classifier-Free Guidance**: Combined text conditions with unconditional training

## Key Technical Details

### Architecture Overview

```
Text Input → T5-XXL (frozen) → Cross-Attention → Base Diffusion (64x64)
                                                      ↓
                                              Super-Resolution 1 (64→256)
                                                      ↓
                                              Super-Resolution 2 (256→1024)
                                                      ↓
                                              Output Image (1024x1024)
```

### Base Diffusion Model
- **Text Encoder**: T5-XXL (4.6B parameters, frozen)
- **Unet**: 2.5B parameters
- **Resolution**: 64x64
- **Training Steps**: 2.5M steps
- **Batch Size**: 2048 (TPU v4)

### Super-Resolution Models
| Stage | Input→Output | Model Size | Attention |
|-------|--------------|------------|-----------|
| SR1 | 64→256 | 300M | Memory-efficient |
| SR2 | 256→1024 | 300M | Memory-efficient |

### Key Techniques

#### Dynamic Thresholding
```
During sampling, clip x_t to [-s, s] where s is dynamic:
- High guidance weights: s increases to prevent over-saturation
- Prevents saturation artifacts at high CFG values
```

#### Efficient Attention
```
- Uses linear attention instead of full attention in SR models
- Reduces memory from O(N^2) to O(N)
- Enables 1024x1024 generation
```

#### Classifier-Free Guidance
```
w * ε_cond + (1-w) * ε_uncond
where w = guidance weight (typically 1-10)
```

### Training Data
- Internal dataset: 460M image-text pairs
- Images filtered for aesthetic quality
- Text from image alt-text and descriptions

## Experimental Results

### FID Scores on COCO

| Model | FID ↓ | CLIP ↑ |
|-------|-------|--------|
| DALL-E 2 | 10.4 | - |
| Latent Diffusion | 5.9 | - |
| Make-A-Scene | 11.3 | - |
| **Imagen** | **7.3** | **0.84** |

### Human Evaluation: Text Fidelity
- 91.0% of raters prefer Imagen over other methods
- 97.3% prefer Imagen for photorealism
- 88.6% prefer Imagen for image-text alignment

### Ablation Studies
- T5-XXL > CLIP text encoders (contrary to DALL-E 2 choice)
- Dynamic thresholding prevents over-saturation
- Larger text encoders improve text fidelity
- Cascaded SR improves fine details

## Related Papers

- [[DALL-E2]] - DALL-E 2: Hierarchical Image Generation
- [[StableDiffusion]] - High-Resolution Image Synthesis with Latent Diffusion
- [[Parti]] - Google's Pathway Autoregressive Model
- [[GLIDE]] - Text-Weighted Diffusion for Image Generation
- [[CLIP]] - Contrastive Language-Image Pretraining
- [[LatentDiffusion]] - High-Resolution Image Synthesis with Latent Diffusion Models
- [[Imagen3]] - Next generation Imagen

## BibTeX Citation

```bibtex
@article{saharia2022imagen,
  title={Photorealistic text-to-image diffusion models with deep language understanding},
  author={Saharia, Chitwan and Chan, William and Saxena, Saurabh and Li, Lala and Whang, Jay and Denton, Emily and Ghasemipour, Seyed Kamyar Seyed and Ayan, Burcu Karagol and Mahdavi, S Sara and Lopes, Rapha Gontijo and others},
  journal={arXiv preprint arXiv:2205.11487},
  year={2022}
}
```
