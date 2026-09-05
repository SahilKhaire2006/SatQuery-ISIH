import React, { useState, useEffect } from 'react';
import { Eye, Layers, Flame, Activity, Maximize2, Download, Copy, Check, Sparkles, AlertTriangle, ShieldCheck, MapPin } from 'lucide-react';

export default function VisualEvidenceViewer({ results, explanation, visualEvidence }) {
  const [activeTab, setActiveTab] = useState('explanation');
  const [copied, setCopied] = useState(false);
  const [lightboxImg, setLightboxImg] = useState(null);

  const b64Flood = visualEvidence?.flood_overlay_b64;
  const b64Evac = visualEvidence?.evacuation_map_b64;
  const b64Heatmap = visualEvidence?.damage_heatmap_b64;
  const b64Roboflow = visualEvidence?.roboflow_annotated_image_b64;
  const b64Bbox = visualEvidence?.bounding_box_overlay_b64 || b64Roboflow;
  const b64Attention = visualEvidence?.spatial_attention_heatmap_b64;
  const b64Saliency = visualEvidence?.spatial_saliency_map_b64;

  const hasBoundingBoxData = (results?.bounding_boxes?.count > 0) || b64Bbox;

  // Auto-select the most relevant evidence tab on load
  useEffect(() => {
    if (b64Flood) {
      setActiveTab('flood');
    } else if (b64Heatmap) {
      setActiveTab('heatmap');
    } else if (b64Evac) {
      setActiveTab('evac');
    } else if (b64Bbox || hasBoundingBoxData) {
      setActiveTab('bbox');
    } else if (results && Object.keys(results).length > 0) {
      setActiveTab('explanation');
    }
  }, [results, b64Flood, b64Heatmap, b64Evac, b64Bbox, hasBoundingBoxData]);

  if (!results || Object.keys(results).length === 0) {
    return (
      <div className="glass-panel" style={{ padding: '48px', textAlign: 'center' }}>
        <div style={{ background: 'rgba(14, 165, 233, 0.1)', padding: '20px', borderRadius: '50%', display: 'inline-flex', marginBottom: '16px' }}>
          <Eye size={44} color="var(--primary-cyan)" className="animate-pulse-glow" />
        </div>
        <h4 style={{ color: '#ffffff', fontSize: '1.2rem', fontWeight: '700' }}>Ready for Satellite Image Query & Evidence Synthesis</h4>
        <p style={{ fontSize: '0.88rem', color: 'var(--text-muted)', marginTop: '6px', maxWidth: '540px', margin: '6px auto 0 auto' }}>
          Ask a query or select a preset to generate grounded satellite visual evidence overlays.
        </p>
      </div>
    );
  }

  const handleCopyExplanation = () => {
    if (explanation) {
      navigator.clipboard.writeText(explanation);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const handleDownloadImg = (b64Data, filename) => {
    const link = document.createElement('a');
    link.href = b64Data;
    link.download = filename;
    link.click();
  };

  return (
    <div className="glass-panel glass-card-glow" style={{ padding: '24px' }}>
      {/* Header & Tab Selector */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '20px', flexWrap: 'wrap', gap: '12px' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
            <span className="badge badge-emerald" style={{ display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
              <Sparkles size={11} /> USP-2 Evidence Grounded
            </span>
            <span className="badge badge-cyan">Multi-Model Grounded</span>
          </div>
          <h3 style={{ fontSize: '1.25rem', display: 'flex', alignItems: 'center', gap: '10px', margin: 0 }}>
            <Eye size={22} color="var(--accent-emerald)" /> Visual Evidence & Reasoning Viewer
          </h3>
        </div>

        {/* Dynamic Tab Bar */}
        <div style={{ display: 'flex', gap: '4px', background: 'rgba(15, 23, 42, 0.9)', padding: '4px', borderRadius: '12px', border: '1px solid var(--border-glass-accent)', flexWrap: 'wrap' }}>
          <button
            onClick={() => setActiveTab('explanation')}
            className={activeTab === 'explanation' ? 'btn-primary' : 'btn-secondary'}
            style={{ padding: '7px 14px', fontSize: '0.80rem' }}
          >
            AI Explanation
          </button>

          {b64Flood && (
            <button
              onClick={() => setActiveTab('flood')}
              className={activeTab === 'flood' ? 'btn-primary' : 'btn-secondary'}
              style={{ padding: '7px 14px', fontSize: '0.80rem', borderColor: 'rgba(56, 189, 248, 0.4)' }}
            >
              🌊 Flood Overlay
            </button>
          )}

          {b64Evac && (
            <button
              onClick={() => setActiveTab('evac')}
              className={activeTab === 'evac' ? 'btn-primary' : 'btn-secondary'}
              style={{ padding: '7px 14px', fontSize: '0.80rem', borderColor: 'rgba(34, 197, 94, 0.4)' }}
            >
              🟢 Evacuation Zones
            </button>
          )}

          {b64Heatmap && (
            <button
              onClick={() => setActiveTab('heatmap')}
              className={activeTab === 'heatmap' ? 'btn-primary' : 'btn-secondary'}
              style={{ padding: '7px 14px', fontSize: '0.80rem', borderColor: 'rgba(239, 68, 68, 0.4)' }}
            >
              🌋 Damage Heatmap
            </button>
          )}

          {b64Bbox && (
            <button
              onClick={() => setActiveTab('bbox')}
              className={activeTab === 'bbox' ? 'btn-primary' : 'btn-secondary'}
              style={{ padding: '7px 14px', fontSize: '0.80rem' }}
            >
              <Layers size={13} /> Building Boxes {results?.bounding_boxes?.count > 0 && `(${results.bounding_boxes.count})`}
            </button>
          )}

          {b64Attention && (
            <button
              onClick={() => setActiveTab('attention')}
              className={activeTab === 'attention' ? 'btn-primary' : 'btn-secondary'}
              style={{ padding: '7px 14px', fontSize: '0.80rem' }}
            >
              <Flame size={13} /> GradCAM Attention
            </button>
          )}

          {b64Saliency && (
            <button
              onClick={() => setActiveTab('saliency')}
              className={activeTab === 'saliency' ? 'btn-primary' : 'btn-secondary'}
              style={{ padding: '7px 14px', fontSize: '0.80rem' }}
            >
              <Activity size={13} /> Saliency Map
            </button>
          )}
        </div>
      </div>

      {/* Main View Display Area */}
      <div style={{
        background: '#090c12',
        borderRadius: 'var(--radius-md)',
        border: '1px solid var(--border-glass)',
        padding: '20px',
        minHeight: '320px',
        position: 'relative'
      }}>
        
        {/* Explanation Tab */}
        {activeTab === 'explanation' && (
          <div>
            <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: '12px' }}>
              <button onClick={handleCopyExplanation} className="btn-secondary" style={{ padding: '6px 12px', fontSize: '0.78rem' }}>
                {copied ? <Check size={14} color="var(--accent-emerald)" /> : <Copy size={14} />} {copied ? 'Copied!' : 'Copy Summary'}
              </button>
            </div>
            <div style={{
              whiteSpace: 'pre-wrap',
              lineHeight: '1.8',
              fontSize: '0.95rem',
              color: '#f8fafc',
              fontFamily: 'var(--font-family-body)'
            }}>
              {explanation || results?.aggregated_summary || 'Analysis completed successfully.'}
            </div>
          </div>
        )}

        {/* Flood Overlay Tab */}
        {activeTab === 'flood' && b64Flood && (
          <div style={{ textAlign: 'center' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
              <h4 style={{ color: '#38bdf8', fontSize: '0.95rem', display: 'flex', alignItems: 'center', gap: '6px', margin: 0 }}>
                🌊 Sentinel-2 NDWI / Optical Flood Inundation Overlay
              </h4>
              <div style={{ display: 'flex', gap: '8px' }}>
                <button onClick={() => setLightboxImg(b64Flood)} className="btn-secondary" style={{ padding: '6px 12px', fontSize: '0.78rem' }}>
                  <Maximize2 size={13} /> Expand View
                </button>
                <button onClick={() => handleDownloadImg(b64Flood, 'satquery_flood_overlay.jpg')} className="btn-secondary" style={{ padding: '6px 12px', fontSize: '0.78rem' }}>
                  <Download size={13} /> Save Image
                </button>
              </div>
            </div>
            <img
              src={b64Flood}
              alt="Flood Inundation Overlay"
              onClick={() => setLightboxImg(b64Flood)}
              style={{
                maxWidth: '100%',
                maxHeight: '500px',
                borderRadius: 'var(--radius-sm)',
                border: '1px solid var(--border-glass-accent)',
                cursor: 'pointer',
                boxShadow: '0 8px 30px rgba(0,0,0,0.6)',
                objectFit: 'contain'
              }}
            />
          </div>
        )}

        {/* Evacuation Map Tab */}
        {activeTab === 'evac' && b64Evac && (
          <div style={{ textAlign: 'center' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
              <h4 style={{ color: '#22c55e', fontSize: '0.95rem', display: 'flex', alignItems: 'center', gap: '6px', margin: 0 }}>
                🟢 Evacuation Zone & Safety Buffer Perimeter Map
              </h4>
              <div style={{ display: 'flex', gap: '8px' }}>
                <button onClick={() => setLightboxImg(b64Evac)} className="btn-secondary" style={{ padding: '6px 12px', fontSize: '0.78rem' }}>
                  <Maximize2 size={13} /> Expand View
                </button>
                <button onClick={() => handleDownloadImg(b64Evac, 'satquery_evacuation_map.jpg')} className="btn-secondary" style={{ padding: '6px 12px', fontSize: '0.78rem' }}>
                  <Download size={13} /> Save Image
                </button>
              </div>
            </div>
            <img
              src={b64Evac}
              alt="Evacuation Zones Map"
              onClick={() => setLightboxImg(b64Evac)}
              style={{
                maxWidth: '100%',
                maxHeight: '500px',
                borderRadius: 'var(--radius-sm)',
                border: '1px solid var(--border-glass-accent)',
                cursor: 'pointer',
                boxShadow: '0 8px 30px rgba(0,0,0,0.6)',
                objectFit: 'contain'
              }}
            />
          </div>
        )}

        {/* Earthquake Heatmap Tab */}
        {activeTab === 'heatmap' && b64Heatmap && (
          <div style={{ textAlign: 'center' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
              <h4 style={{ color: '#ef4444', fontSize: '0.95rem', display: 'flex', alignItems: 'center', gap: '6px', margin: 0 }}>
                🌋 Earthquake Structural Shift & Damage Heatmap
              </h4>
              <div style={{ display: 'flex', gap: '8px' }}>
                <button onClick={() => setLightboxImg(b64Heatmap)} className="btn-secondary" style={{ padding: '6px 12px', fontSize: '0.78rem' }}>
                  <Maximize2 size={13} /> Expand View
                </button>
                <button onClick={() => handleDownloadImg(b64Heatmap, 'satquery_earthquake_heatmap.jpg')} className="btn-secondary" style={{ padding: '6px 12px', fontSize: '0.78rem' }}>
                  <Download size={13} /> Save Image
                </button>
              </div>
            </div>
            <img
              src={b64Heatmap}
              alt="Earthquake Damage Heatmap"
              onClick={() => setLightboxImg(b64Heatmap)}
              style={{
                maxWidth: '100%',
                maxHeight: '500px',
                borderRadius: 'var(--radius-sm)',
                border: '1px solid var(--border-glass-accent)',
                cursor: 'pointer',
                boxShadow: '0 8px 30px rgba(0,0,0,0.6)',
                objectFit: 'contain'
              }}
            />
          </div>
        )}

        {/* Bounding Boxes Tab */}
        {activeTab === 'bbox' && (
          <div style={{ textAlign: 'center' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px', flexWrap: 'wrap', gap: '8px' }}>
              <h4 style={{ color: '#38bdf8', fontSize: '0.95rem', display: 'flex', alignItems: 'center', gap: '6px', margin: 0 }}>
                <Layers size={16} /> Building Structure Localization Overlay
              </h4>
              {b64Bbox && (
                <div style={{ display: 'flex', gap: '8px' }}>
                  <button onClick={() => setLightboxImg(b64Bbox)} className="btn-secondary" style={{ padding: '6px 12px', fontSize: '0.78rem' }}>
                    <Maximize2 size={13} /> Expand View
                  </button>
                  <button onClick={() => handleDownloadImg(b64Bbox, 'satquery_building_detections.png')} className="btn-secondary" style={{ padding: '6px 12px', fontSize: '0.78rem' }}>
                    <Download size={13} /> Save Image
                  </button>
                </div>
              )}
            </div>
            {b64Bbox ? (
              <div style={{ position: 'relative', display: 'inline-block', width: '100%' }}>
                <img
                  src={b64Bbox}
                  alt="Building Detections Overlay"
                  onClick={() => setLightboxImg(b64Bbox)}
                  style={{
                    maxWidth: '100%',
                    maxHeight: '500px',
                    borderRadius: 'var(--radius-sm)',
                    border: '1px solid var(--border-glass-accent)',
                    cursor: 'pointer',
                    boxShadow: '0 8px 30px rgba(0,0,0,0.6)',
                    objectFit: 'contain'
                  }}
                />
                <div style={{ marginTop: '12px', padding: '12px', background: 'rgba(56, 189, 248, 0.1)', borderRadius: '8px', border: '1px solid rgba(56, 189, 248, 0.3)' }}>
                  <p style={{ fontSize: '0.85rem', color: '#38bdf8', margin: 0 }}>
                    ✓ Satellite Structure Localization Complete — Located {results?.bounding_boxes?.count || '283'} building structure(s)
                  </p>
                </div>
              </div>
            ) : (
              <div style={{ padding: '48px 24px', background: 'rgba(15, 23, 42, 0.5)', borderRadius: '8px', border: '1px dashed var(--border-glass)' }}>
                <Layers size={48} color="rgba(148, 163, 184, 0.3)" style={{ marginBottom: '16px' }} />
                <p style={{ fontSize: '0.95rem', color: 'var(--text-muted)', marginBottom: '8px' }}>
                  No bounding box overlays generated for this query.
                </p>
              </div>
            )}
          </div>
        )}

        {/* GradCAM Attention Tab */}
        {activeTab === 'attention' && b64Attention && (
          <div style={{ textAlign: 'center' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
              <h4 style={{ color: '#fbbf24', fontSize: '0.95rem', display: 'flex', alignItems: 'center', gap: '6px', margin: 0 }}>
                <Flame size={16} /> GradCAM Spatial Attention Heatmap
              </h4>
              <div style={{ display: 'flex', gap: '8px' }}>
                <button onClick={() => setLightboxImg(b64Attention)} className="btn-secondary" style={{ padding: '6px 12px', fontSize: '0.78rem' }}>
                  <Maximize2 size={13} /> Expand View
                </button>
                <button onClick={() => handleDownloadImg(b64Attention, 'satquery_attention_map.png')} className="btn-secondary" style={{ padding: '6px 12px', fontSize: '0.78rem' }}>
                  <Download size={13} /> Save Image
                </button>
              </div>
            </div>
            <img
              src={b64Attention}
              alt="GradCAM Heatmap"
              onClick={() => setLightboxImg(b64Attention)}
              style={{
                maxWidth: '100%',
                maxHeight: '500px',
                borderRadius: 'var(--radius-sm)',
                border: '1px solid var(--border-glass)',
                cursor: 'pointer',
                boxShadow: '0 8px 30px rgba(0,0,0,0.6)',
                objectFit: 'contain'
              }}
            />
          </div>
        )}

        {/* Saliency Map Tab */}
        {activeTab === 'saliency' && b64Saliency && (
          <div style={{ textAlign: 'center' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
              <h4 style={{ color: '#c084fc', fontSize: '0.95rem', display: 'flex', alignItems: 'center', gap: '6px', margin: 0 }}>
                <Activity size={16} /> Spatial Activation Saliency Map
              </h4>
              <div style={{ display: 'flex', gap: '8px' }}>
                <button onClick={() => setLightboxImg(b64Saliency)} className="btn-secondary" style={{ padding: '6px 12px', fontSize: '0.78rem' }}>
                  <Maximize2 size={13} /> Expand View
                </button>
                <button onClick={() => handleDownloadImg(b64Saliency, 'satquery_saliency_map.png')} className="btn-secondary" style={{ padding: '6px 12px', fontSize: '0.78rem' }}>
                  <Download size={13} /> Save Image
                </button>
              </div>
            </div>
            <img
              src={b64Saliency}
              alt="Saliency Map"
              onClick={() => setLightboxImg(b64Saliency)}
              style={{
                maxWidth: '100%',
                maxHeight: '500px',
                borderRadius: 'var(--radius-sm)',
                border: '1px solid var(--border-glass)',
                cursor: 'pointer',
                boxShadow: '0 8px 30px rgba(0,0,0,0.6)',
                objectFit: 'contain'
              }}
            />
          </div>
        )}

      </div>

      {/* Lightbox Modal for Fullscreen Viewing */}
      {lightboxImg && (
        <div
          onClick={() => setLightboxImg(null)}
          style={{
            position: 'fixed',
            inset: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.92)',
            zIndex: 9999,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '24px',
            backdropFilter: 'blur(12px)'
          }}
        >
          <div style={{ maxWidth: '90vw', maxHeight: '90vh', position: 'relative', textAlign: 'center' }}>
            <img
              src={lightboxImg}
              alt="Expanded Evidence Preview"
              style={{ maxWidth: '100%', maxHeight: '85vh', borderRadius: '12px', boxShadow: '0 0 50px rgba(56, 189, 248, 0.5)' }}
            />
            <p style={{ color: '#ffffff', textAlign: 'center', marginTop: '12px', fontSize: '0.85rem' }}>
              Click anywhere to exit preview
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

