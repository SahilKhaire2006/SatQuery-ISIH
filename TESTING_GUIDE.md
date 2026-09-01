# 🧪 SatQuery Testing Guide

## Quick Start Testing

### Step 1: Start the API Server

Open a terminal and run:
```cmd
start_server.bat
```

OR manually:
```cmd
venv\Scripts\activate
python main.py
```

**Wait for this message:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

---

### Step 2: Test the API (Browser)

Open your browser and test these URLs:

1. **API Documentation (Interactive Swagger UI)**
   ```
   http://localhost:8000/docs
   ```
   - You'll see all endpoints with "Try it out" buttons
   - You can test the API directly from the browser!

2. **Health Check**
   ```
   http://localhost:8000/api/v1/health
   ```
   Expected response:
   ```json
   {
     "status": "healthy",
     "timestamp": "2026-09-01T...",
     "services": {
       "api": "operational",
       "orchestrator": "operational",
       "models": "operational"
     }
   }
   ```

3. **List Available Tools**
   ```
   http://localhost:8000/api/v1/tools
   ```
   Expected response:
   ```json
   {
     "tools": [
       "vqa_model",
       "grounding_model",
       "change_detection_model",
       "sar_fusion_model"
     ],
     "count": 4
   }
   ```

---

### Step 3: Test with CLI

Open a **NEW terminal** (keep the server running) and run:
```cmd
start_cli.bat
```

OR manually:
```cmd
venv\Scripts\activate
python cli.py
```

---

## 📋 CLI Testing Workflow

### Test Case 1: VQA (Visual Question Answering)

1. Select option: **1** (Process new query)

2. When prompted for image path, enter:
   ```
   data/raw/test_satellite_image.png
   ```

3. When prompted for query, enter:
   ```
   How many buildings are visible in this image?
   ```

4. **Expected Output:**
   ```
   ============================================================
   QUERY RESULTS
   ============================================================
   
   Status: SUCCESS
   
   Answer:
   There are approximately 5-7 objects visible in the image
   
   Overall Confidence: 0.85
   
   Tool-specific Confidences:
     - vqa_model: 0.85
   
   Tools Used: 1
     - VQA Model (BLIP-based)
   ```

---

### Test Case 2: Grounding (Object Localization)

1. Select option: **1**

2. Image path:
   ```
   data/raw/test_satellite_image.png
   ```

3. Query:
   ```
   Where are the buildings located?
   ```

4. **Expected Output:**
   - Status: SUCCESS
   - Answer with bounding box information
   - Confidence scores
   - Tools Used: Grounding Model

---

### Test Case 3: Change Detection

1. Select option: **1**

2. Image path:
   ```
   data/raw/test_satellite_image.png
   ```

3. Query:
   ```
   What changes occurred in this area?
   ```

4. **Expected Output:**
   - Status: SUCCESS
   - Change percentage
   - Confidence scores
   - Tools Used: Change Detection Model

---

### Test Case 4: SAR Fusion

1. Select option: **1**

2. Image path:
   ```
   data/raw/test_satellite_image.png
   ```

3. Query:
   ```
   Analyze this using SAR and optical data
   ```

4. **Expected Output:**
   - Status: SUCCESS
   - Fusion analysis results
   - Modalities used
   - Tools Used: SAR Fusion Model

---

## 🌐 Testing with API Directly (Using Browser Swagger UI)

1. Go to: http://localhost:8000/docs

2. Click on **POST /api/v1/query**

3. Click **"Try it out"**

4. Fill in the form:
   - **query**: "How many buildings are visible?"
   - **image**: Click "Choose File" and select `data/raw/test_satellite_image.png`

5. Click **"Execute"**

6. View the response below

---

## 🔧 Testing with PowerShell/CMD

### Test with curl (if available):

```powershell
# Health check
curl http://localhost:8000/api/v1/health

# List tools
curl http://localhost:8000/api/v1/tools

# Process query (multipart form)
curl -X POST "http://localhost:8000/api/v1/query" `
  -F "query=How many buildings are visible?" `
  -F "image=@data/raw/test_satellite_image.png"
```

### Test with Python script:

Create a file `test_api.py`:

```python
import requests

url = "http://localhost:8000/api/v1/query"

# Open the test image
with open('data/raw/test_satellite_image.png', 'rb') as f:
    files = {'image': f}
    data = {'query': 'How many buildings are visible in this image?'}
    
    response = requests.post(url, files=files, data=data)
    print(response.json())
```

Run it:
```cmd
venv\Scripts\python test_api.py
```

---

## 📊 Sample Test Queries

### VQA Queries:
- "How many buildings are visible in this image?"
- "What type of terrain is shown?"
- "Describe what you see in this satellite image"
- "Count the number of roads"

### Grounding Queries:
- "Where are the buildings located?"
- "Find water bodies in this image"
- "Locate the roads"
- "Identify urban areas"

### Change Detection Queries:
- "What changes occurred in this area?"
- "Compare the temporal differences"
- "Detect urban growth"

### SAR Fusion Queries:
- "Analyze using SAR and optical data"
- "Perform cross-modal fusion"
- "Extract features using radar and optical imagery"

---

## ✅ Expected Behavior

### Successful Response Structure:
```json
{
  "session_id": "uuid-string",
  "query_id": "uuid-string",
  "status": "success",
  "results": {
    "answer": "Answer text",
    "confidence_scores": {},
    "details": {}
  },
  "confidence": 0.85,
  "audit_log": {
    "validation": {},
    "interpretation": {},
    "selected_tools": [],
    "execution": []
  },
  "timestamp": "2026-09-01T..."
}
```

---

## 🐛 Troubleshooting

### CLI says "Cannot connect to API server"
- Make sure the API server is running (`start_server.bat`)
- Check if port 8000 is available
- Try accessing http://localhost:8000/health in browser

### "File not found" error
- Make sure you're using the correct path
- Use full path: `A:\VIT-3rd-sem\3rd_sem\isih\SatQuery\data\raw\test_satellite_image.png`
- Or relative path from project root: `data/raw/test_satellite_image.png`

### API returns 500 error
- Check the server terminal for error messages
- Ensure all dependencies are installed
- Try restarting the server

---

## 📈 Testing Checklist

- [ ] API server starts successfully
- [ ] Health endpoint responds
- [ ] Tools endpoint lists 4 models
- [ ] CLI connects to API
- [ ] Test image can be loaded
- [ ] VQA query works
- [ ] Grounding query works
- [ ] Change detection query works
- [ ] SAR fusion query works
- [ ] Results are displayed correctly
- [ ] Export to JSON works
- [ ] Session management works (multiple queries)

---

## 🎯 Next Steps

After successful testing:

1. ✅ Verify all 4 model types work
2. ✅ Test with your own satellite images
3. ✅ Review the architecture (ARCHITECTURE.md)
4. ✅ Explore API documentation (http://localhost:8000/docs)
5. ✅ Customize models for production use

---

## 📞 Quick Reference

**Start Server:** `start_server.bat`  
**Start CLI:** `start_cli.bat`  
**API Docs:** http://localhost:8000/docs  
**Test Image:** `data/raw/test_satellite_image.png`  
**Sample Query:** "How many buildings are visible?"

Happy Testing! 🚀
