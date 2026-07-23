# Project Overview

This repository contains a final-year Legal AI project for multi-label IPC section prediction from legal text and audio.

## Main components

- Legal_AI_System/: backend, models, scripts, and frontend integration
- Dataset/: training and evaluation dataset files
- docs/: project architecture, flowcharts, and report documents
- Legal_AI_System/data/audio_samples/: sample audio files for testing

## How the system works

1. User provides text or audio input.
2. The backend routes the request to the appropriate predictor.
3. The model predicts relevant IPC sections.
4. Results are returned through the API or CLI.

## Key technologies

- Python
- PyTorch
- Transformers
- Flask
- React
- Speech recognition models
