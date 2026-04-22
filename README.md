# Yoga Pose Classification and Feedback System

## Table of Contents
- [Quick Start](#quick-start)
- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Datasets](#datasets)
- [Pipeline](#pipeline)
- [Core Modeling Ideas](#core-modeling-ideas)
---

## Quick Start

<img width="921" height="488" alt="system_architecture" src="https://github.com/user-attachments/assets/f363a820-811e-4ca1-90a8-bc777de4cc87" />


### Clone repo
```bash
git clone <your-repo-url>
cd <repo-name>
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run notebooks

**Image baseline**
```
CS7150_YogaImageClassification_ManojDataset.ipynb
```

**Video baseline**
```
cs7150-video2.ipynb
```

**Final model**
```
cs7150-yogavideo-synthpose-extension.ipynb
```

---

## Project Overview

Our Northeastern CS7150 Final Project builds a yoga pose assessment system that:
- classifies yoga poses
- detects incorrect form
- provides joint-level feedback
- supports transfer learning across datasets

---

## Repository Structure
```
.
├── notebooks/
│   ├── CS7150_YogaImageClassification_ManojDataset.ipynb
│   ├── cs7150-video2.ipynb
│   ├── cs7150-yogavideo-synthpose.ipynb
│   └── cs7150-yogavideo-synthpose-extension.ipynb
├── scripts/
│   └── video_downloaderBest.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Datasets
- Manoj Dataset (image baseline)
- YogNet (main evaluation dataset)
- 3DYoga90 (pretraining + transfer learning)

---

## Pipeline
Raw Data → Pose Extraction → Normalization → Windowing → Synthetic Errors → TCN → Outputs

---

## Core Modeling Ideas
- MediaPipe (33 keypoints)
- SynthPose (52 keypoints)
- Temporal Convolutional Networks
- Multi-task learning (pose + quality + feedback)
- Transfer learning

---

## Environment Notes
- Designed for Kaggle GPU
- Update file paths for local use
