import React, { useState, useEffect, useRef } from 'react';
import Header from './components/Header';
import ImageUploader from './components/ImageUploader';
import QueryConsole from './components/QueryConsole';
import VisualEvidenceViewer from './components/VisualEvidenceViewer';
import SARFusionCard from './components/SARFusionCard';
import AuditTrailInspector from './components/AuditTrailInspector';
import GeospatialCard from './components/GeospatialCard';
import { Activity, ShieldCheck, Database, RefreshCw, Cpu, CheckCircle2 } from 'lucide-react';

export default function App() {
  const [activeTab, setActiveTab] = useState('analysis');
  const [apiHealthy, setApiHealthy] = useState(true);
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [query, setQuery] = useState('');
  const [geoMetadata, setGeoMetadata] = useState('');
  const [loading, setLoading] = useState(false);
  const [loadingStep, setLoadingStep] = useState('');
  const [queryResponse, setQueryResponse] = useState(null);
  const [metricsData, setMetricsData] = useState(null);
  const [sessionID] = useState(() => 'sess_' + Math.random().toString(36).substring(2, 9));

  const evidenceRef = useRef(null);

  // Check API health on load
  useEffect(() => {
    fetch('/api/v1/health')
      .then((res) => res.json())
      .then((data) => setApiHealthy(data.status === 'healthy' || data.status === 'degraded'))
      .catch(() => setApiHealthy(false));
  }, []);

  // Fetch metrics when Telemetry tab is opened
  useEffect(() => {
    if (activeTab === 'metrics') {
      fetch('/metrics')
        .then((res) => res.text())
        .then((text) => setMetricsData(text))
        .catch(() => setMetricsData('Unable to reach Prometheus /metrics endpoint.'));
    }
  }, [activeTab]);

  const handleExecuteQuery = async () => {
    if (!query.trim()) {
      alert('Please enter a natural language satellite query.');
      return;
    }

    setLoading(true);
    setLoadingStep('Step 1: Input Validation & Image Preprocessing...');
    setQueryResponse(null);

    try {
      const formData = new FormData();
      formData.append('query', query);
      formData.append('session_id', sessionID);
      if (geoMetadata) {
        formData.append('geo_metadata', geoMetadata);
      }

      // If no image file selected, generate dummy 100x100 PNG blob
      let fileToUpload = selectedFile;
      if (!fileToUpload) {
        const canvas = document.createElement('canvas');
        canvas.width = 100;
        canvas.height = 100;
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#0f172a';
        ctx.fillRect(0, 0, 100, 100);
        const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/png'));
        fileToUpload = new File([blob], 'synthetic_sat.png', { type: 'image/png' });
      }

      formData.append('image', fileToUpload);

      setTimeout(() => setLoadingStep('Step 2: Groq LLM Query Interpretation & Spatial Extraction...'), 400);
      setTimeout(() => setLoadingStep('Step 3: Specialist Computer Vision Tool Routing...'), 900);
      setTimeout(() => setLoadingStep('Step 4: Vision Model Inference & Visual Evidence Generation...'), 1400);

      const res = await fetch('/api/v1/query', {
        method: 'POST',
        body: formData,
      });

      const data = await res.json();
      setQueryResponse(data);

      // Smooth scroll to evidence section
      if (evidenceRef.current) {
        setTimeout(() => {
          evidenceRef.current.scrollIntoView({ behavior: 'smooth' });
        }, 100);
      }
    } catch (err) {
      console.error('API execution error:', err);
      alert('Failed to connect to API endpoint. Ensure FastAPI backend is running.');
    } finally {
      setLoading(false);
      setLoadingStep('');
    }
  };

  const results = queryResponse?.results || {};
  const explanation = queryResponse?.explanation || results?.aggregated_summary;
  const visualEvidence = results?.visual_evidence || {};
  const auditLog = queryResponse?.audit_log || {};
  const fusionInfo = results?.sar_fusion_model?.fusion_info || results?.details?.sar_fusion_model?.fusion_info;
  const geoData = results?.geospatial_metadata || auditLog?.geospatial;

  return (
    <div style={{ maxWidth: '1400px', margin: '0 auto', padding: '20px' }}>
      {/* Navigation Header */}
      <Header
        apiHealthy={apiHealthy}
        sessionID={sessionID}
        activeTab={activeTab}
        setActiveTab={setActiveTab}
      />

      {activeTab === 'analysis' ? (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '24px' }}>
          
          {/* Top Row: Image Uploader & Query Console */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(380px, 1fr))', gap: '24px' }}>
            <ImageUploader
              selectedFile={selectedFile}
              setSelectedFile={setSelectedFile}
              previewUrl={previewUrl}
              setPreviewUrl={setPreviewUrl}
              geoMetadata={geoMetadata}
              setGeoMetadata={setGeoMetadata}
            />

            <QueryConsole
              query={query}
              setQuery={setQuery}
              onExecute={handleExecuteQuery}
              loading={loading}
            />
          </div>

          {/* Loading Progress Indicator */}
          {loading && (
            <div className="glass-panel" style={{ padding: '20px', textAlign: 'center', border: '1px solid var(--border-glass-accent)' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '10px', marginBottom: '8px' }}>
                <RefreshCw size={20} color="var(--primary-cyan)" className="animate-spin-slow" />
                <span style={{ fontWeight: '600', color: '#ffffff', fontSize: '0.95rem' }}>{loadingStep}</span>
              </div>
              <div style={{ width: '100%', height: '4px', background: 'rgba(15, 23, 42, 0.8)', borderRadius: '2px', overflow: 'hidden' }}>
                <div style={{ width: '80%', height: '100%', background: 'linear-gradient(90deg, #0ea5e9, #8b5cf6, #10b981)', animation: 'pulseGlow 1.5s infinite' }} />
              </div>
            </div>
          )}

          {/* Middle Row: Visual Evidence Viewer (USP-2) */}
          <div ref={evidenceRef}>
            <VisualEvidenceViewer
              results={results}
              explanation={explanation}
              visualEvidence={visualEvidence}
            />
          </div>

          {/* Bottom Row: SAR Radar Fusion Card (USP-1) & Geospatial Card */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))', gap: '24px' }}>
            <SARFusionCard fusionInfo={fusionInfo} />
            <GeospatialCard geoData={geoData} />
          </div>

          {/* Audit Trail Inspector */}
          {queryResponse && (
            <AuditTrailInspector
              auditLog={auditLog}
              confidence={queryResponse?.confidence}
            />
          )}

        </div>
      ) : (
        /* Telemetry & Metrics Dashboard Tab */
        <div className="glass-panel" style={{ padding: '32px' }}>
          <h2 style={{ fontSize: '1.4rem', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '10px' }}>
            <Activity size={24} color="var(--primary-cyan)" /> Prometheus System Metrics & Telemetry
          </h2>
          <p style={{ color: 'var(--text-muted)', marginBottom: '20px', fontSize: '0.9rem' }}>
            Live metrics exposition from <code style={{ color: '#38bdf8' }}>/metrics</code> endpoint:
          </p>

          <pre style={{
            background: 'rgba(7, 9, 14, 0.9)',
            padding: '20px',
            borderRadius: 'var(--radius-md)',
            border: '1px solid var(--border-glass)',
            fontFamily: 'var(--font-family-mono)',
            fontSize: '0.85rem',
            color: '#34d399',
            maxHeight: '450px',
            overflowY: 'auto',
            whiteSpace: 'pre-wrap'
          }}>
            {metricsData || 'Loading telemetry...'}
          </pre>
        </div>
      )}
    </div>
  );
}
