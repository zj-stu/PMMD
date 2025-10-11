<div align="center">
## Pseudo Label Guided Multimodal Manipulation Detection (PMMD)

# Code for PMMD: One Framework to Detect and Localize: Towards Unified Multimodal Manipulation Forensics
<br>

<img src='./figs/motivation.png' width='90%'>

</div>

## Introduction
This is the official implementation of *Spatial Intelligence as a Prior: A Two-Stage Framework for Multimodal Forgery Detection and Grounding*. We propose a novel pseudo-label guided paradigm for multimodal manipulation detection, namely **P**seudo **L**abel Guided **M**ultimodal **M**anipulation **D**etection (PMMD).

Different from existing methods that treat detection and localization as loosely coupled tasks, PMMD establishes a unified framework where high-quality manipulation localization pseudo labels act as spatial priors to systematically enhance both visual grounding and cross-modal reasoning.

Our two-stage architecture consists of:
1. Forgery Region Proposal Generation
2. Forgery Region-Aided Manipulation Detection and Grounding

The framework of the proposed HAMMER model:

<div align="center">
<img src='./figs/framework.png' width='90%'>
</div>

This implementation is written by Bingwen Hu and Jun Zhou at Anhui University of Technology.

## Prerequisites
- Python 3.8 or above
- Pytorch 1.12
- CUDA 11.6 or above

## Train & Test
Other modules will be updated after publication.
