# SurveyEval
python GeoAdapterStep2.py --mode ffhq --device cuda --fp16 --model stabilityai/stable-diffusion-xl-base-1.0 --resolution 256 --batch-size 1 --max-train-steps 200 --log-every 10
A comprehensive evaluation framework for face generation and manipulation methods, focusing on geometric and perceptual quality assessment.

## Overview

This project provides a multi-step evaluation pipeline for assessing face generation methods through:
- Geometric analysis (landmarks, symmetry, face geometry)
- Perceptual quality metrics (CLIPScore, HPSv2)
- Survey-based human evaluation
- Correlation analysis between metrics and human judgments

## Project Structure

- `step0.py` - Setup and model verification
- `step1.py` - Feature extraction (landmarks, identity embeddings, CLIP embeddings)
- `step2.py` - Geometric analysis
- `step3.py` - Perceptual quality evaluation
- `step4.py` - Survey data analysis and correlation computation
- `GeoAdapterStep1.py`, `GeoAdapterStep2.py` - Geometric adapter components

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the evaluation pipeline in sequence:

```bash
python step0.py    # Setup and verify models
python step1.py    # Extract features
python step2.py    # Geometric analysis
python step3.py    # Perceptual metrics
python step4.py    # Survey analysis
```

## Requirements

See `requirements.txt` for full dependency list. Key dependencies include:
- PyTorch
- OpenCV
- InsightFace
- dlib
- scikit-learn
- matplotlib

## Data

Large datasets, model files, and checkpoints are excluded from this repository. See `.gitignore` for details.

## License

See individual component licenses in the project.

