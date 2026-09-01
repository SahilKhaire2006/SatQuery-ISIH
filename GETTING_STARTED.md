# 🚀 SatQuery - Complete Setup & Run Guide

**Satellite Image Query & Analysis System**  
A complete guide for setting up and running the project.

---

## 📋 Quick Start

1. **Clone & Navigate**
   ```bash
   git clone <repository-url>
   cd SatQuery
   ```

2. **Setup Environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure**
   ```bash
   copy .env.example .env
   ```

4. **Run**
   ```bash
   # Terminal 1 - Start Server
   python main.py
   
   # Terminal 2 - Start CLI
   python cli.py
   ```

5. **Test**
   - Open browser: http://localhost:8000/docs
   - Use test image: `data\raw\test_satellite_image.png`
   - Try query: "How many buildings are visible in this image?"

---

## 🔧 Detailed Setup Instructions

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)
- Git

### Step 1: Install Dependencies

After activating the virtual environment:

```bash
pip install -r requirements.txt
```

**Installed packages:**
- FastAPI, Uvicorn (API server)
- NumPy, OpenCV, Pillow (image processing)
- SQLAlchemy, Redis (database - optional)
- Pydantic, python-dotenv (configuration)
- And 30+ supporting packages

### Step 2: Environment Configuration (.env file)

Create `.env` file by copying the example:

```bash
copy .env.example .env
```

**Default Configuration (Works as-is):**

```env
# API Configuration
API_HOST=0.0.0.0          # Server host
API_PORT=8000             # Server port (change if 8000 is in use)
API_KEY=your-secret-api-key

# Database (Optional - prototype uses in-memory storage)
DATABASE_URL=postgresql://user:password@localhost:5432/satquery_db
REDIS_URL=redis://localhost:6379/0

# Paths
MODELS_PATH=./models
DATA_PATH=./data

# Processing
MAX_IMAGE_SIZE=1024       # Max image size (px)
BATCH_SIZE=4
CONFIDENCE_THRESHOLD=0.7
```

**Important Notes:**
- ⚠️ Never commit `.env` to Git (already in `.gitignore`)
- ✅ Database settings are optional for the prototype
- ✅ Default values work for most cases
- 🔄 Restart server after changing `.env`

**Common Customizations:**

| Scenario | Change |
|----------|--------|
| Port 8000 in use | `API_PORT=8001` |
| Low-resource PC | `MAX_IMAGE_SIZE=512`, `BATCH_SIZE=2` |
| High-end PC | `MAX_IMAGE_SIZE=2048`, `BATCH_SIZE=8` |

---

## 🎯 Running the Project

### Method 1: Quick Start Scripts

**Start API Server:**
```bash
start_server.bat
```

**Start CLI (in another terminal):**
```bash
start_cli.bat
```

### Method 2: Manual Execution

**Terminal 1 - API Server:**
```bash
venv\Scripts\activate
python main.py
```

Wait for: `INFO: Uvicorn running on http://0.0.0.0:8000`

**Terminal 2 - CLI Interface:**
```bash
venv\Scripts\activate
python cli.py
```

### Method 3: Docker

```bash
docker-compose up -d
```

---

## 🧪 Testing the System

### 1. Browser Test (Easiest)

Open: http://localhost:8000/docs

**Try it out:**
1. Click **POST /api/v1/query**
2. Click **"Try it out"**
3. Fill in:
   - **query**: `How many buildings are visible?`
   - **image**: Upload `data\raw\test_satellite_image.png`
4. Click **"Execute"**
5. View JSON response below

### 2. CLI Test

```
1. Select option: 1
2. Image path: data\raw\test_satellite_image.png
3. Query: How many buildings are visible in this image?
```

**Expected Output:**
```
Status: SUCCESS
Answer: There are approximately 5-7 objects visible in the image
Overall Confidence: 85-95%
Tools Used: VQA Model
```

### 3. Automated Test

```bash
python test_api_quick.py
```

This tests all endpoints automatically.

---

## 📁 Project Structure

```
SatQuery/
├── api/                    # REST API & Session Management
│   ├── gateway.py          # API endpoints
│   └── session_manager.py  # Session handling
│
├── agentic_layer/          # AI Orchestration Layer
│   ├── orchestrator.py     # Main coordinator
│   ├── input_validator.py  # Validates input
│   ├── query_interpreter.py# Classifies queries
│   ├── tool_selector.py    # Selects AI models
│   └── execution_engine.py # Executes models
│
├── models/                 # AI Models (4 types)
│   ├── vqa_model.py        # Visual Q&A
│   ├── grounding_model.py  # Object localization
│   ├── change_detection_model.py  # Change detection
│   └── sar_fusion_model.py # SAR-Optical fusion
│
├── cli/                    # Command-line Interface
├── data/                   # Data storage
│   ├── raw/                # Input images here
│   └── processed/          # Processed outputs
│
├── config/                 # Configuration
├── utils/                  # Utilities
├── tests/                  # Tests
│
├── .env                    # Your config (create this!)
├── main.py                 # API server entry
├── cli.py                  # CLI entry
└── requirements.txt        # Dependencies
```

---

## 🤖 Available AI Models

The system automatically selects the right model based on your query:

### 1. VQA Model (Visual Question Answering)
**Example Queries:**
- "How many buildings are visible?"
- "What type of terrain is shown?"
- "Describe what you see"

### 2. Grounding Model (Object Localization)
**Example Queries:**
- "Where are the buildings located?"
- "Find water bodies"
- "Locate agricultural fields"

### 3. Change Detection Model
**Example Queries:**
- "What changes occurred?"
- "Detect urban growth"
- "Compare temporal differences"

### 4. SAR Fusion Model
**Example Queries:**
- "Analyze using SAR and optical data"
- "Perform cross-modal fusion"

---

## 🌐 API Endpoints

### Base URL
```
http://localhost:8000
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Root endpoint |
| POST | `/api/v1/query` | Process image query |
| GET | `/api/v1/session/{id}` | Get session info |
| GET | `/api/v1/health` | Health check |
| GET | `/api/v1/tools` | List available models |
| GET | `/docs` | Interactive API docs |

### Example: Process Query

**cURL:**
```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -F "query=How many buildings?" \
  -F "image=@data/raw/test_satellite_image.png"
```

**Python:**
```python
import requests

url = "http://localhost:8000/api/v1/query"
files = {'image': open('data/raw/test_satellite_image.png', 'rb')}
data = {'query': 'How many buildings are visible?'}

response = requests.post(url, files=files, data=data)
print(response.json())
```

**Response:**
```json
{
  "session_id": "uuid",
  "query_id": "uuid",
  "status": "success",
  "results": {
    "answer": "There are approximately 5-7 buildings visible",
    "confidence_scores": {"vqa_model": 0.85}
  },
  "confidence": 0.85,
  "timestamp": "2026-09-01T..."
}
```

---

## 🐛 Troubleshooting

### Issue: Port 8000 already in use
**Solution:** 
```env
# Edit .env
API_PORT=8001
```

### Issue: "Cannot connect to API server"
**Solution:** Make sure API server is running first:
```bash
start_server.bat
```

### Issue: Import errors
**Solution:**
```bash
pip install -r requirements.txt
```

### Issue: Virtual environment won't activate
**Solution:**
```bash
# Use full path
.\venv\Scripts\activate.bat
```

### Issue: Test image not found
**Solution:**
```bash
python create_test_image.py
```

### Issue: Changes to .env not working
**Solution:** Restart the server (Ctrl+C, then run again)

---

## 💡 Tips for Teammates

### Backend Developers
- API logic: `/api/` and `/agentic_layer/`
- AI models: `/models/` (currently placeholders)
- Add model weights to `/models/` directory for production
- All endpoints use FastAPI with async support

### Frontend Developers
- API docs: http://localhost:8000/docs
- Test all endpoints in Swagger UI
- CORS enabled for development
- API returns standardized JSON responses
- Session management built-in

### DevOps
- Docker ready: `docker-compose.yml`
- Environment config: `.env` file
- Logs stored in `/logs/`
- Database optional for prototype

---

## 📚 Architecture Overview

The system follows a 5-layer architecture:

```
┌─────────────────────────────────────┐
│   Presentation Layer (CLI)          │
└─────────────────────────────────────┘
             ↓
┌─────────────────────────────────────┐
│   API & Session Layer (FastAPI)     │
└─────────────────────────────────────┘
             ↓
┌─────────────────────────────────────┐
│   Agentic Orchestration Layer       │
│   • Input Validator                 │
│   • Query Interpreter               │
│   • Tool Selector                   │
│   • Execution Engine                │
└─────────────────────────────────────┘
             ↓
┌─────────────────────────────────────┐
│   Specialist Model Layer            │
│   • VQA Model                       │
│   • Grounding Model                 │
│   • Change Detection Model          │
│   • SAR Fusion Model                │
└─────────────────────────────────────┘
             ↓
┌─────────────────────────────────────┐
│   Data & Benchmark Layer            │
└─────────────────────────────────────┘
```

**Data Flow:**
1. User uploads image + query
2. API Gateway receives request
3. Input Validator checks quality
4. Query Interpreter classifies task
5. Tool Selector picks appropriate model
6. Execution Engine runs model(s)
7. Results aggregated with confidence scores
8. Response returned with full audit log

---

## ✅ Setup Checklist

Before starting development:

- [ ] Python 3.10+ installed
- [ ] Repository cloned
- [ ] Virtual environment created
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] `.env` file created
- [ ] Test image available (`python create_test_image.py`)
- [ ] API server starts successfully
- [ ] CLI connects and works
- [ ] Health endpoint responds (http://localhost:8000/api/v1/health)
- [ ] API docs accessible (http://localhost:8000/docs)

---

## 🎯 Quick Reference

### Start Server
```bash
python main.py
```

### Start CLI
```bash
python cli.py
```

### API Documentation
```
http://localhost:8000/docs
```

### Test Image Path
```
data\raw\test_satellite_image.png
```

### Sample Query
```
How many buildings are visible in this image?
```

### Health Check
```
http://localhost:8000/api/v1/health
```

---

## 🔒 Security Notes

**Never commit to Git:**
- ❌ `.env` file
- ❌ API keys
- ❌ Database credentials
- ❌ Secret keys

**Safe to commit:**
- ✅ `.env.example` (template)
- ✅ All code files
- ✅ Documentation
- ✅ Configuration templates

---

## 🤝 Team Workflow

1. **Clone** repository
2. **Create** `.env` file (don't commit!)
3. **Install** dependencies
4. **Test** everything works
5. **Start** development

**Branch Strategy:**
- `main` - Production code
- `develop` - Development branch
- `feature/your-feature` - Your features

---

## 📞 Need Help?

1. Check this guide
2. Test with: `python test_api_quick.py`
3. View server logs for errors
4. Check API docs at `/docs`
5. Ask team lead

---

## 🎉 You're Ready!

Your SatQuery system is ready to use. The prototype is fully functional with all 5 architectural layers implemented.

**Happy Coding! 🚀**

---

**Project:** SatQuery - Satellite Image Query & Analysis System  
**Team:** VIT 3rd Semester ISIH Project  
**Last Updated:** September 1, 2026
