# Legal AI System Frontend
## Final Year BTech Project - Web Interface for Legal Document Analysis

A modern React-based web interface for the Legal AI System that provides multi-modal analysis of legal documents and audio content for Indian Penal Code (IPC) classification. This frontend is part of a comprehensive **Final Year BTech Project** at NIT Srinagar.

**Academic Context:** Final Year BTech Project (2024-2025)  
**Department:** Information Technology, NIT Srinagar  
**Component:** Web Interface for Legal AI System

## Features

### 🎯 Core Functionality
- **Text Analysis**: Analyze legal text documents, case files, and legal content
- **Audio Analysis**: Upload and analyze audio files with automatic transcription
- **Multi-Modal Support**: Choose between single model and ensemble analysis modes
- **Real-time Results**: View comprehensive analysis results with confidence scores

### 🎨 User Interface
- **Modern Material-UI Design**: Clean, professional interface with responsive design
- **Tabbed Interface**: Easy navigation between text, audio, and results views
- **Drag & Drop**: Intuitive file upload for audio analysis
- **Progress Indicators**: Visual feedback during analysis processes
- **Comprehensive Results**: Detailed breakdown of predictions and model performance

### 🔧 Technical Features
- **TypeScript**: Full type safety and better development experience
- **Responsive Design**: Works seamlessly on desktop and mobile devices
- **Error Handling**: Comprehensive error handling and user feedback
- **API Integration**: Ready for backend integration with RESTful API

## 🎓 Academic Context

This frontend component is part of a comprehensive Final Year BTech Project that demonstrates:

- **Full-Stack Development**: Complete web application with React frontend and Python backend
- **Modern Web Technologies**: TypeScript, Material-UI, and responsive design
- **API Integration**: RESTful API communication with AI backend
- **User Experience Design**: Professional interface for legal professionals
- **Production Deployment**: Ready-to-deploy web application

## Screenshots

### Main Interface
- Clean tabbed interface with Material-UI components
- Professional legal-themed design with gavel icon
- Responsive layout for different screen sizes

### Text Analysis
- Large text input area for legal content
- Analysis mode selection (Single/Multimodal)
- Real-time validation and error handling

### Audio Analysis
- Drag & drop file upload interface
- File validation and progress tracking
- Support for multiple audio formats

### Results Display
- Comprehensive results with confidence scores
- Performance metrics and processing times
- Detailed model comparison tables
- Expandable sections for detailed analysis

## Installation

### Prerequisites
- Node.js (v14 or higher)
- npm or yarn package manager
- Backend Legal AI System running on port 5001

### Setup Instructions

1. **Clone the repository** (if not already done):
   ```bash
   cd Legal_AI_System/frontend/legal-ai-frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Start the development server**:
   ```bash
   npm start
   ```

4. **Open your browser** and navigate to `http://localhost:3000`

### Automated Setup (Recommended)
```bash
# From the main Legal_AI_System directory
python start_frontend_demo.py
```

## Usage

### Text Analysis
1. Navigate to the "Text Analysis" tab
2. Enter or paste legal text in the input area
3. Select analysis mode (Single Model or Multimodal Ensemble)
4. Click "Analyze Text" to start the analysis
5. View results in the "Results" tab

### Audio Analysis
1. Navigate to the "Audio Analysis" tab
2. Drag and drop an audio file or click to select
3. Choose analysis mode
4. Click "Analyze Audio" to start processing
5. Monitor upload and processing progress
6. View results in the "Results" tab

### Results Interpretation
- **Best Predictions**: Top IPC section classifications with confidence scores
- **Performance Metrics**: Processing time and confidence levels
- **Model Comparison**: Detailed breakdown of different model performances
- **Input Summary**: Overview of analyzed content and metadata

## Supported Formats

### Text Input
- Legal documents and case files
- Police reports and FIRs
- Court judgments and orders
- Legal pleadings and petitions
- General legal text and descriptions

### Audio Input
- **Formats**: MP3, WAV, M4A, FLAC, OGG
- **Size Limit**: Maximum 50MB
- **Content Types**:
  - Court proceedings and hearings
  - Legal consultations and advice
  - Police statements and interviews
  - Witness testimonies
  - Legal discussions and debates

## Analysis Modes

### Single Model
- Uses the trained InLegalBERT model
- Faster processing time
- Suitable for quick analysis

### Multimodal Ensemble
- Combines multiple ASR models for audio
- Enhanced accuracy and confidence
- More comprehensive analysis

## Backend Integration

The frontend is designed to work with the Legal AI System backend API. The current implementation includes:

- **API Service Layer**: Ready for backend integration
- **Error Handling**: Comprehensive error management
- **Progress Tracking**: Upload and processing progress
- **Type Safety**: Full TypeScript integration

### API Endpoints (Expected)
- `POST /api/analyze/text` - Text analysis
- `POST /api/analyze/audio` - Audio analysis
- `GET /api/status` - System status
- `GET /api/models` - Available models

## Development

### Project Structure
```
src/
├── components/          # React components
│   ├── TextAnalysis.tsx
│   ├── AudioAnalysis.tsx
│   └── ResultsDisplay.tsx
├── services/           # API services
│   └── api.ts
├── types.ts           # TypeScript type definitions
└── App.tsx           # Main application component
```

### Available Scripts
- `npm start` - Start development server
- `npm build` - Build for production
- `npm test` - Run tests
- `npm eject` - Eject from Create React App

### Customization
- **Theme**: Modify Material-UI theme in `App.tsx`
- **API Configuration**: Update API settings in `services/api.ts`
- **Components**: Extend or modify components as needed

## Dependencies

### Core Dependencies
- **React 18**: Modern React with hooks
- **TypeScript**: Type safety and better development experience
- **Material-UI**: Professional UI components
- **Axios**: HTTP client for API communication
- **React Dropzone**: File upload functionality

### Development Dependencies
- **Create React App**: Development environment
- **Testing Library**: Component testing utilities

## Browser Support

- Chrome (recommended)
- Firefox
- Safari
- Edge

## 🎓 Academic Documentation

This frontend is part of a comprehensive Final Year BTech Project. For complete documentation:

- **Main Project Report**: [../../../docs/Final_Project_Report.md](../../../docs/Final_Project_Report.md) (or [../docs/COMBINED_REPORT.md](../docs/COMBINED_REPORT.md) in the backend docs)
- **System Architecture**: `../docs/COMBINED_ARCHITECTURE.md`
- **User Guide**: `../docs/COMBINED_README.md`
- **Academic Report**: `../docs/COMBINED_REPORT.md`

## Contributing

This is a Final Year BTech Project component. For academic purposes:

1. Review the complete project documentation
2. Understand the system architecture
3. Follow the established coding standards
4. Test thoroughly before deployment

## License

This project is part of a Final Year BTech Project and follows academic guidelines. All code and documentation are provided for educational and research purposes.

---

**Final Year BTech Project - Information Technology Department, NIT Srinagar (2024-2025)**
