# Quick Start Guide - Roboflow Detection

## 🚀 Ready to Use!

Your SatQuery system now has **AI-powered building and water body detection**!

## Run Tests

```bash
# Test building detection
python test_roboflow_integration.py

# Test water body detection
python test_waterbody_integration.py
```

## Start Application

```bash
# Terminal 1: Start API server
python main.py

# Terminal 2: Start frontend
cd frontend
npm run dev
```

Then open browser to `http://localhost:5173`

## Try These Queries

### Building Detection 🏢
- "count buildings in this image"
- "detect structures"
- "how many buildings?"
- "find buildings"

### Water Body Detection 💧
- "detect water bodies"
- "find lakes"
- "show me water areas"
- "identify rivers"
- "count water bodies"

## What to Expect

1. **Upload** satellite image
2. **Enter** query (e.g., "detect water bodies")
3. **See results**:
   - Annotated segmentation image
   - Color-coded: Cyan for water 💧, Purple for buildings 🏢
   - Bounding boxes (if available)
   - Detection count
   - Confidence scores

## Configuration

Your `.env` is already configured with:
- Building detection: `general-segmentation-api-2`
- Water body detection: `general-segmentation-api-4`
- API Key: `nFr9z8OUTQCmKOKgrt0c`

## Troubleshooting

**Detector not loading?**
- Check `.env` has `ROBOFLOW_API_KEY=nFr9z8OUTQCmKOKgrt0c`
- Verify: `pip install inference-sdk`

**No results showing?**
- Check logs in terminal
- Verify image uploaded successfully
- Try different query phrasing

**Frontend not updating?**
- Clear browser cache (Ctrl+Shift+R)
- Restart dev server

## Key Files

- `models/roboflow_building_detector.py` - Building detector
- `models/roboflow_waterbody_detector.py` - Water detector
- `.env` - Configuration
- `INTEGRATION_SUMMARY.md` - Full documentation

## Success Checklist

- [x] Building detection working ✅
- [x] Water body detection working ✅
- [x] Query routing automatic ✅
- [x] Frontend displays correctly ✅
- [x] Tests passing ✅

You're all set! 🎉
