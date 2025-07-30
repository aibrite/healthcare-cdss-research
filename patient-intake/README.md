# Medical Intake Assistant

An AI-powered patient intake system that helps healthcare professionals gather comprehensive medical information through intelligent questioning and conversation management.

![Medical Intake Assistant](https://img.shields.io/badge/AI-Powered%20Medical%20Intake-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Flask](https://img.shields.io/badge/Flask-Web%20Interface-green.svg)
![Ollama](https://img.shields.io/badge/Ollama-Local%20AI-orange.svg)

## Overview

The Medical Intake Assistant is a sophisticated AI-driven application designed to streamline the patient intake process in healthcare settings. It uses advanced natural language processing to conduct intelligent conversations with patients, asking relevant follow-up questions and generating comprehensive intake reports for healthcare professionals.

### Key Features

- **🤖 AI-Powered Questioning**: Intelligent follow-up questions based on patient responses
- **📝 Comprehensive Intake**: Covers all essential medical history areas systematically
- **⚡ Real-time Streaming**: Live conversation with streaming AI responses
- **🔍 Completion Detection**: Automatically determines when intake is complete
- **📊 Report Generation**: Creates detailed medical intake summaries
- **🎯 Thinking Process**: Optional visibility into AI reasoning process
- **🌐 Web Interface**: Modern, responsive chat-based interface
- **🔒 Session Management**: Secure conversation continuity
- **🏥 Healthcare Focused**: Specifically designed for medical intake workflows

## Architecture

The system is built with a modular architecture:

- **Frontend**: Modern web interface with HTML5, CSS3, and JavaScript
- **Backend**: Flask web server with session management
- **AI Engine**: Ollama integration with configurable language models
- **Core Logic**: Python modules for intake management and report generation

## Prerequisites

- **Python**: 3.10 or higher
- **Ollama**: Local AI model server
- **System**: Linux, macOS, or Windows with WSL2

## Installation

### 1. Clone and Setup

```bash
cd patient-intake
```

### 2. Install Dependencies

Using uv:

```bash
uv sync
```

Using pip:
```bash
pip install flask ollama flask-session
```

### 3. Ollama Setup

Install and start Ollama:
```bash
# Install Ollama (visit https://ollama.com for installation instructions)
ollama serve

# Pull the required model (in a new terminal)
ollama pull gemma3:12b
```

### 4. Environment Configuration

Copy the environment template and configure:
```bash
cp default.env .env
```

Edit `.env` with your configuration:
```env
OLLAMA_HOST="localhost:11434"
MODEL_NAME="gemma3:12b"
TEMPERATURE=0.1
SECRET_KEY="your-secure-secret-key"
```

## Usage

### Quick Start

Launch the application with the automated setup:

```bash
python run_web_app.py
```

This will:
- ✅ Check all dependencies
- ✅ Verify Ollama connection
- ✅ Test AI functionality
- 🚀 Start the web server
- 🌐 Open your browser automatically

### Manual Start

For development or custom configuration:

```bash
python web_server.py
```

Then visit: http://localhost:5000

### Using the Interface

1. **Start Intake**: Describe initial symptoms or concerns
2. **Interactive Q&A**: Answer AI-generated follow-up questions
3. **Monitor Progress**: Track completion status in real-time
4. **Generate Report**: Automatically creates comprehensive intake summary
5. **Review Results**: Export or review the final medical intake report

## Core Functionality

### AI Intake Areas

The system systematically covers:

- **Chief Complaint**: Primary symptoms and concerns
- **History of Present Illness**: Detailed symptom progression
- **Associated Symptoms**: Related or secondary symptoms
- **Past Medical History**: Previous conditions and treatments
- **Current Medications**: All current prescriptions and supplements
- **Allergies**: Drug, food, and environmental allergies
- **Social History**: Lifestyle factors affecting health
- **Lab Results**: Recent test results and findings

### Intelligent Features

- **Context-Aware Questions**: Each question builds on previous responses
- **Completion Detection**: AI determines when sufficient information is gathered
- **Medical Focus**: Questions follow established medical intake protocols
- **Flexible Conversation**: Adapts to patient communication style
- **Report Synthesis**: Generates structured medical summaries

## Configuration

### Model Options

The system supports various Ollama models.


### Advanced Settings

```env
TEMPERATURE=0.1              # Response consistency (0.0-1.0)
OLLAMA_HOST="localhost:11434" # Ollama server address
SECRET_KEY="unique-secret"    # Flask session security
```

## API Endpoints

### REST API

- `GET /` - Main chat interface
- `POST /start_chat` - Initialize intake session
- `POST /send_message` - Send patient response
- `GET /stream_question` - Stream AI question generation
- `GET /check_completion` - Check intake completion status
- `GET /generate_report` - Create intake report
- `GET /conversation_history` - Retrieve session history

### Streaming Endpoints

Real-time streaming for enhanced user experience:
- Question generation with live AI thinking process
- Completion status updates
- Report generation with progress indicators

## Development

### Project Structure

```
patient-intake/
├── patient-intake.py      # Core AI logic and intake management
├── web_server.py          # Flask web application
├── run_web_app.py         # Application launcher with checks
├── templates/
│   └── index.html         # Web interface template
├── static/
│   ├── css/style.css      # Application styling
│   └── js/app.js          # Frontend JavaScript
├── pyproject.toml         # Project configuration
├── default.env            # Environment template
└── README.md              # This documentation
```

### Code Quality

The project includes:
- **Ruff**: Code formatting and linting
- **Type Hints**: Comprehensive type annotations
- **Error Handling**: Robust error management
- **Documentation**: Detailed docstrings and comments

### Testing

Test the system components:
```bash
python run_web_app.py  # Runs built-in system tests
```

## Troubleshooting

### Common Issues

**Ollama Connection Failed**
```bash
# Check if Ollama is running
ollama list

# Restart Ollama service
ollama serve
```

**Model Not Found**
```bash
# Pull required model
ollama pull gemma3:12b

# List available models
ollama list
```

**Dependencies Missing**
```bash
# Install all requirements
uv sync
```

### Performance Optimization

- **Model Selection**: Choose appropriate model size for your hardware
- **Temperature Tuning**: Lower values (0.1) for consistent medical responses
- **Hardware**: GPU acceleration improves response times significantly

### Development Setup

```bash
uv sync --dev
uv run ruff check .
uv run ruff format .
```

## Support

For technical support or questions:
- Review the troubleshooting section
- Check Ollama documentation for AI model issues
- Ensure all prerequisites are properly installed
