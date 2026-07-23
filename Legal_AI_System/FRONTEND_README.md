# Legal AI System - Frontend Integration
## Final Year BTech Project - Web Interface Component

This document describes the frontend web interface for the Legal AI System, providing a modern, user-friendly way to interact with the IPC classification system. This frontend component is part of a comprehensive **Final Year BTech Project** at NIT Srinagar.

**Academic Context:** Final Year BTech Project (2024-2025)  
**Department:** Information Technology, NIT Srinagar  
**Component:** Web Interface for Legal AI System

## 🎯 Overview

The frontend provides a complete web interface for:
- **Text Analysis**: Analyze legal documents and text using InLegalBERT
- **Audio Analysis**: Upload and analyze audio files with automatic transcription
- **Multi-Modal Support**: Choose between single model and ensemble analysis
- **Real-time Results**: View comprehensive analysis with confidence scores

## 🎓 Academic Context

This frontend component demonstrates several key aspects of modern web development and system integration:

- **Full-Stack Development**: Complete web application with React frontend and Python backend
- **Modern Web Technologies**: TypeScript, Material-UI, and responsive design
- **API Integration**: RESTful API communication with AI backend
- **User Experience Design**: Professional interface for legal professionals
- **Production Deployment**: Ready-to-deploy web application

## 🏗️ Architecture

```
┌─────────────────┐    HTTP/JSON    ┌─────────────────┐
│   React Frontend│ ◄──────────────► │  Flask API      │
│   (Port 3000)   │                 │  (Port 5001)    │
└─────────────────┘                 └─────────────────┘
                                              │
                                              ▼
                                    ┌─────────────────┐
                                    │ Legal AI System │
                                    │ (InLegalBERT +  │
                                    │  ASR Models)    │
                                    └─────────────────┘
```

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

1. **Start the complete system**:
   ```bash
   cd Legal_AI_System
   python3 start_frontend_demo.py
   ```

2. **Follow the on-screen instructions** to start the frontend

3. **Open your browser** to `http://localhost:3000`

### Option 2: Manual Setup

#### Backend Setup
1. **Install Python dependencies**:
   ```bash
   cd Legal_AI_System
   pip install flask flask-cors
   ```

2. **Start the API server**:
   ```bash
   python3 api_server.py
   ```

#### Frontend Setup
1. **Navigate to frontend directory**:
   ```bash
   cd frontend/legal-ai-frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Start the development server**:
   ```bash
   npm start
   ```

4. **Open your browser** to `http://localhost:3000`

## 🎨 Features

### Text Analysis Tab
- **Large text input area** for legal content
- **Analysis mode selection** (Single/Multimodal)
- **Real-time validation** and error handling
- **Supported content types**:
  - Legal documents and case files
  - Police reports and FIRs
  - Court judgments and orders
  - Legal pleadings and petitions

### Audio Analysis Tab
- **Drag & drop file upload** interface
- **File validation** (type and size limits)
- **Progress tracking** during upload and processing
- **Supported formats**: MP3, WAV, M4A, FLAC, OGG (max 50MB)
- **Supported content**:
  - Court proceedings and hearings
  - Legal consultations and advice
  - Police statements and interviews
  - Witness testimonies

### Results Tab
- **Comprehensive results display** with confidence scores
- **Performance metrics** and processing times
- **Model comparison tables** for multimodal analysis
- **Expandable sections** for detailed analysis
- **Visual confidence indicators** with color coding

## 🔧 Technical Details

### Frontend Technology Stack
- **React 18** with TypeScript
- **Material-UI** for professional UI components
- **Axios** for API communication
- **React Dropzone** for file uploads
- **Responsive design** for mobile and desktop

### Backend API Endpoints
- `GET /api/status` - System status check
- `GET /api/models` - Available models information
- `POST /api/analyze/text` - Text analysis
- `POST /api/analyze/audio` - Audio analysis
- `GET /api/health` - Health check

### API Response Format
```json
{
  "success": true,
  "data": {
    "input_type": "text|audio",
    "input_content": "analyzed content",
    "analysis_mode": "single|multimodal",
    "timestamp": "2024-01-01T12:00:00Z",
    "results": {
      "best_model": "model name",
      "best_predictions": [
        {
          "section": "IPC 302",
          "description": "Murder",
          "confidence": 0.85
        }
      ],
      "all_model_results": [...],
      "total_processing_time": 1.2
    },
    "metadata": {
      "audio_duration": 45.2,
      "transcription_models": ["Whisper", "Faster-Whisper"],
      "transcription_confidence": 0.94
    }
  }
}
```

## 📁 Project Structure

```
Legal_AI_System/
├── frontend/
│   └── legal-ai-frontend/
│       ├── src/
│       │   ├── components/
│       │   │   ├── TextAnalysis.tsx
│       │   │   ├── AudioAnalysis.tsx
│       │   │   └── ResultsDisplay.tsx
│       │   ├── services/
│       │   │   └── api.ts
│       │   ├── types.ts
│       │   └── App.tsx
│       ├── package.json
│       └── README.md
├── api_server.py
├── start_frontend_demo.py
└── FRONTEND_README.md
```

## 🎯 Usage Examples

### Text Analysis
1. Navigate to "Text Analysis" tab
2. Enter legal text:
   ```
   The accused intentionally caused the death of the victim by 
   striking them with a blunt object, resulting in severe head 
   injuries that led to death within 24 hours.
   ```
3. Select "Single Model" or "Multimodal Ensemble"
4. Click "Analyze Text"
5. View results in "Results" tab

### Audio Analysis
1. Navigate to "Audio Analysis" tab
2. Drag and drop an audio file or click to select
3. Choose analysis mode
4. Click "Analyze Audio"
5. Monitor progress and view results

## 🔍 Analysis Modes

### Single Model
- **Model**: Trained InLegalBERT
- **Speed**: Fast processing
- **Use Case**: Quick analysis, single predictions
- **Output**: Direct IPC classifications

### Multimodal Ensemble
- **Models**: Multiple ASR + InLegalBERT
- **Speed**: Slower but more accurate
- **Use Case**: Comprehensive analysis, multiple predictions
- **Output**: Best result from ensemble

## 🎓 Academic Documentation

This frontend is part of a comprehensive Final Year BTech Project. For complete documentation:

- **Main Project Report**: [../docs/Final_Project_Report.md](../docs/Final_Project_Report.md)
- **System Architecture**: `docs/COMBINED_ARCHITECTURE.md`
- **User Guide**: `docs/COMBINED_README.md`
- **Academic Report**: `docs/COMBINED_REPORT.md`

## 🔧 Development Guidelines

### Code Standards
- **TypeScript**: Full type safety for all components
- **Material-UI**: Consistent design system
- **Responsive Design**: Mobile-first approach
- **Error Handling**: Comprehensive error management
- **Performance**: Optimized rendering and API calls

### Testing
- **Unit Tests**: Component testing with React Testing Library
- **Integration Tests**: API integration testing
- **User Testing**: Interface usability testing
- **Performance Testing**: Load and stress testing

## 📊 Performance Metrics

### Frontend Performance
- **Load Time**: < 3 seconds initial load
- **Response Time**: < 1 second for user interactions
- **Memory Usage**: < 100MB during operation
- **Bundle Size**: < 2MB optimized build

### User Experience
- **Accessibility**: WCAG 2.1 AA compliance
- **Cross-browser**: Chrome, Firefox, Safari, Edge
- **Mobile Support**: Responsive design for all devices
- **Error Recovery**: Graceful error handling and recovery

## 🚀 Deployment

### Production Build
```bash
cd frontend/legal-ai-frontend
npm run build
```

### Deployment Options
- **Static Hosting**: Netlify, Vercel, GitHub Pages
- **Container Deployment**: Docker with nginx
- **Cloud Platforms**: AWS, Google Cloud, Azure
- **Local Deployment**: Serve with Python or Node.js

## 🎓 Research Contributions

This frontend component contributes to the research objectives by demonstrating:

1. **Modern Web Development**: Contemporary web technologies and practices
2. **User Interface Design**: Professional interface for legal professionals
3. **System Integration**: Seamless integration with AI backend
4. **Production Readiness**: Deployment-ready web application
5. **User Experience**: Intuitive interface for complex AI operations

## 📞 Support

For questions, issues, or contributions:

- **Project Report**: See [../docs/Final_Project_Report.md](../docs/Final_Project_Report.md) for complete documentation
- **Technical Issues**: Check the documentation in `docs/`
- **Frontend Issues**: Review this README and component documentation
- **Academic Context**: Refer to the combined report in `docs/COMBINED_REPORT.md`

---

**Final Year BTech Project - Information Technology Department, NIT Srinagar (2024-2025)** 