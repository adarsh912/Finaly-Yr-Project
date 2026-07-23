# Setup Guide

## 1. Prerequisites

- Python 3.9+
- Node.js 18+
- npm
- Internet access for initial dependency install if model downloads are required

## 2. Python environment

```bash
cd Legal_AI_System
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 3. Frontend dependencies

```bash
cd frontend/legal-ai-frontend
npm install
```

## 4. Run the backend

From the project root:

```bash
python3 api_server.py
```

The backend will run on http://localhost:5001.

## 5. Run the frontend

```bash
cd frontend/legal-ai-frontend
npm start
```

The frontend will run on http://localhost:3000.

## 6. Run the CLI

```bash
python3 main.py --input "The accused committed theft" --input-type text
```
