# 🚀 SatQuery - Setup & Execution Guide

**Agentic Satellite Image Query & Evidence-Grounded AI System**  
A complete guide for running the full-stack system (FastAPI backend + React frontend).

---

## 📋 Quick Start Scripts (Recommended)

To launch the full system with a single command on Windows:

```cmd
start_all.bat
```
*(This launches the FastAPI Backend on `http://localhost:8000` and the React Frontend on `http://localhost:3000` in separate terminal windows.)*

---

## 🔧 Manual Step-by-Step Launch

### Terminal 1: Start FastAPI Backend Server
```powershell
# In project root (SatQuery-ISIH)
.\venv\Scripts\activate
uvicorn api.gateway:app --reload --port 8000
```
- API Docs: `http://localhost:8000/docs`
- Health Endpoint: `http://localhost:8000/api/v1/health`

### Terminal 2: Start React Frontend Application
```powershell
# In project root (SatQuery-ISIH)
cd frontend
npm run dev
```
- Open browser: `http://localhost:3000`

---

## 💡 Logs & Troubleshooting Notes

### ℹ️ Redis Warning Notice
If you see the log:
`utils.cache - WARNING - Redis get/set error: Error 10061 connecting to localhost:6379`
- **What it means**: A local Redis server is not running on your machine.
- **Is it an error?**: **No**. The system is built with automatic cache fallback to **In-Memory Query Cache**. Requests complete normally with `200 OK`.

---

## 🧪 Testing the Web UI

1. Open `http://localhost:3000` in your web browser.
2. Select or upload a satellite image (or click a sample preset like *Urban Aerial*, *Harbor Ships*, *Sentinel SAR*).
3. Type a query like:
   - *"Count the buildings and structures in this image"*
   - *"Locate the water body or river in this imagery"*
   - *"Process Sentinel-1 SAR radar imagery and fuse with optical data"*
4. Click **Run AI Query**.
5. Inspect the generated **AI Explanation**, **Visual Evidence Overlays** (Bounding Boxes, GradCAM Heatmap, Saliency Map), **Optical-SAR Radar Metrics (USP-1)**, and **Audit Trail Inspector**.
