# SatQuery - Satellite Image Query & Analysis System

A multi-layer AI system for satellite image analysis with VQA, grounding, SAR fusion, and change detection capabilities.

## Architecture Layers

1. **Presentation Layer** - CLI for image/query input and results display
2. **API & Session Layer** - API gateway and session state management
3. **Agentic Orchestration Layer** - Input validation, query interpretation, tool selection
4. **Specialist Model Layer** - VQA, grounding, change detection, SAR fusion models
5. **Data & Benchmark Layer** - Training datasets and evaluation benchmarks

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment:
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. Run the API server:
```bash
python main.py
```

4. Run CLI interface:
```bash
python cli.py
```

## Project Structure

- `/api` - API gateway and endpoints
- `/agentic_layer` - Orchestration logic
- `/models` - Specialist models
- `/data_layer` - Data management
- `/cli` - Command-line interface
- `/config` - Configuration files
- `/utils` - Utility functions
