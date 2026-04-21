# Yoga Pose Classification and Feedback System

## Table of Contents
- [Quick Start](#quick-start)
- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Datasets](#datasets)
- [Pipeline](#pipeline)
- [Core Modeling Ideas](#core-modeling-ideas)
- [File-by-File Guide](#file-by-file-guide)
- [Reproduction Workflows](#reproduction-workflows)
- [Dependencies](#dependencies)
- [Environment Notes](#environment-notes)
- [Outputs](#outputs)
- [Limitations](#limitations)
- [Conclusion](#conclusion)

---

## Quick Start

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

This project builds a system that:
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

## File-by-File Guide

### CS7150_YogaImageClassification_ManojDataset.ipynb
- Image classification baseline
- MediaPipe + MLP
- Feedback generation

### cs7150-video2.ipynb
- Video pipeline
- TCN model
- Synthetic augmentation

### cs7150-yogavideo-synthpose.ipynb
- Improved pose extraction

### cs7150-yogavideo-synthpose-extension.ipynb
- Full transfer learning pipeline

### video_downloaderBest.py
- Downloads 3DYoga90 videos

---

## Reproduction Workflows
Run notebooks depending on target experiment:
- image baseline
- video baseline
- final model

---

## Dependencies
See requirements.txt

---

## Environment Notes
- Designed for Kaggle GPU
- Update file paths for local use

---

## Outputs
- classification metrics
- confusion matrices
- feedback predictions
- inference videos

---

## Limitations
- notebook-based
- hardcoded paths
- datasets not included

---

## Conclusion
A full pipeline for yoga pose classification and feedback using deep learning.
