# Project Overview

## Purpose

This repository implements a legal document analysis system that predicts IPC sections from legal text and audio data using a trained transformer-based model.

## Main capabilities

- Text-based legal document analysis
- Audio-based legal document analysis
- Multimodal prediction flow
- REST API for frontend integration
- React-based web interface

## Main folders

- core/: prediction and evaluation logic
- modalities/: text/audio processing modules
- models/: trained model and ASR model assets
- scripts/: training, prediction, and testing utilities
- frontend/: React frontend application
- docs/: project documentation

## Typical workflow

1. User enters text or uploads audio.
2. Backend sends the input to the appropriate predictor.
3. The model returns predicted IPC sections with confidence scores.
4. Results are displayed in the frontend or returned through the API.
