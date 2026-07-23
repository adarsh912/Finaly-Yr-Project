# Setup Guide

## 1. Clone the repository

```bash
git clone <your-repository-url>
cd <repository-folder>
```

## 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

```bash
cd Legal_AI_System
pip install -r requirements.txt
```

## 4. Train the Model (Required for Model Weights Bootstrapping)

Because the custom-trained InLegalBERT model weights (`~418MB`) exceed GitHub's file storage limits, they are excluded from Git. You must train the model locally to initialize the model weights and classes metadata:

```bash
python3 scripts/train_inlegalbert.py
```

This will run training on the local dataset split parts inside the `Dataset/` folder, save the model state, and generate the required class mapping (`classes.json`) in the `models/trained_model` directory.

## 5. Run the CLI

Ensure you are inside the `Legal_AI_System` folder:

```bash
python3 main.py --help
python3 main.py --input "The accused committed theft" --input-type text
```

## 6. Run the backend API

```bash
python3 api_server.py
```

## 7. Run the frontend

```bash
cd frontend/legal-ai-frontend
npm install
npm start
```
