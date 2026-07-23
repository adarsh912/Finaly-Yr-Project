# Legal AI System - Running Commands

This document contains all the commands needed to run the Legal AI System.

## Prerequisites

Make sure you have the following installed:
- Python 3.8+
- Node.js 14+
- npm or yarn

## Backend Setup & Running

### 1. Navigate to the Legal AI System directory
```bash
cd Legal_AI_System
```

### 2. Activate Virtual Environment
```bash
source venv/bin/activate
```

### 3. Install Python Dependencies (if not already installed)
```bash
pip install -r requirements.txt
```

### 4. Start the Backend API Server
```bash
python3 api_server.py
```

**Expected Output:**
```
Initializing Legal AI System models...
✅ Model loaded successfully with 100 classes
🤖 Single Model Predictor Initialized
🚀 Unified Legal AI System Initialized
Starting Legal AI System API server...
 * Running on http://127.0.0.1:5001
 * Debugger is active!
```

**Backend will be available at:** http://localhost:5001

## Frontend Setup & Running

### 1. Navigate to Frontend Directory
```bash
cd Legal_AI_System/frontend/legal-ai-frontend
```

### 2. Install Node.js Dependencies (if not already installed)
```bash
npm install
```

### 3. Start the Frontend Development Server
```bash
npm start
```

**Expected Output:**
```
Compiled successfully!
Local:            http://localhost:3000
On Your Network:  http://192.168.x.x:3000
```

**Frontend will be available at:** http://localhost:3000

## Complete Startup Sequence

### Terminal 1 - Backend
```bash
cd Legal_AI_System
source venv/bin/activate
python3 api_server.py
```

### Terminal 2 - Frontend
```bash
cd Legal_AI_System/frontend/legal-ai-frontend
npm start
```

## Auto-Start Both Backend and Frontend

You can use the provided script to start both backend and frontend automatically:

```bash
cd Legal_AI_System
python3 start_system.py
```

- This script will check prerequisites, start the backend and frontend, and display logs for both.
- Press `Ctrl+C` to stop both servers.

## Testing the System

### 1. Test Backend API
```bash
curl -X GET http://localhost:5001/api/status
```

### 2. Test Text Analysis
```bash
curl -X POST http://localhost:5001/api/analyze/text \
  -H "Content-Type: application/json" \
  -d '{"text": "The accused intentionally caused the death of the victim", "mode": "single"}'
```

### 3. Test Audio Analysis
```bash
curl -X POST http://localhost:5001/api/analyze/audio \
  -F "audio=@audio_samples/audio1.mp3" \
  -F "mode=single"
```

## Troubleshooting

### Backend Issues

#### Virtual Environment Not Found
```bash
# If venv doesn't exist, create it
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Port Already in Use
```bash
# Check what's using port 5001
lsof -i :5001

# Kill the process if needed
kill -9 <PID>
```

#### Model Loading Issues
```bash
# Check if models exist
ls -la models/trained_model/

# Reinstall dependencies
pip install -r requirements.txt
```

### Frontend Issues

#### Node Modules Missing
```bash
cd Legal_AI_System/frontend/legal-ai-frontend
rm -rf node_modules package-lock.json
npm install
```

#### Port 3000 Already in Use
```bash
# Check what's using port 3000
lsof -i :3000

# Kill the process if needed
kill -9 <PID>
```

#### Network Error in Frontend
1. Ensure backend is running on port 5001
2. Check browser console for CORS errors
3. Verify API endpoints are accessible

## System Architecture

```
Frontend (React) → Backend (Flask) → AI Models
     ↓                    ↓              ↓
  Port 3000          Port 5001      InLegalBERT
```

## Available Endpoints

- `GET /api/status` - System status
- `POST /api/analyze/text` - Text analysis
- `POST /api/analyze/audio` - Audio analysis
- `GET /api/models` - Available models

## File Structure

```
Legal_AI_System/
├── api_server.py              # Main backend server
├── venv/                      # Python virtual environment
├── models/
│   └── trained_model/         # Trained InLegalBERT model
├── frontend/
│   └── legal-ai-frontend/     # React frontend
├── audio_samples/             # Sample audio files
├── start_system.py            # Auto-start script for backend & frontend
└── requirements.txt           # Python dependencies
```

## Stopping the System

### Stop Backend
Press `Ctrl+C` in the backend terminal

### Stop Frontend
Press `Ctrl+C` in the frontend terminal

### Stop Both (if using auto-start script)
Press `Ctrl+C` in the terminal running `start_system.py`

## Environment Variables

The system uses these default configurations:
- Backend Port: 5001
- Frontend Port: 3000
- Model Path: models/trained_model/
- Confidence Threshold: 0.3

## Support

If you encounter issues:
1. Check the console output for error messages
2. Verify all dependencies are installed
3. Ensure ports are not in use by other applications
4. Check that the trained model files exist in the correct location 