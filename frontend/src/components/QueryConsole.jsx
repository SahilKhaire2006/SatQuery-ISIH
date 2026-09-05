import React from 'react';
import { Search, Sparkles, Compass, Zap, Target, RefreshCw } from 'lucide-react';

export default function QueryConsole({ query, setQuery, onExecute, loading }) {
  const suggestions = [
    { label: "Count buildings & structures", query: "Count the buildings and structures in this image", task: "vqa" },
    { label: "Locate water body", query: "Locate the water body or river in this imagery", task: "grounding" },
    { label: "Sentinel-1 SAR Radar Analysis", query: "Process Sentinel-1 SAR radar imagery and fuse with optical data", task: "sar" },
    { label: "Detect urban land change", query: "Detect land surface changes and urban expansion", task: "change" },
    { label: "🚨 Kerala Flood Assessment", query: "Show flood extent and evacuation plan in Wayanad, Kerala", task: "disaster" },
    { label: "🌋 Earthquake Damage Audit", query: "Earthquake structural damage assessment in Turkey", task: "disaster" },
  ];

  return (
    <div className="glass-panel" style={{ padding: '24px' }}>
      <h3 style={{ fontSize: '1.1rem', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
        <Compass size={20} color="var(--primary-cyan)" /> Natural Language Query Console
      </h3>

      <div style={{ position: 'relative', marginBottom: '16px' }}>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask a question about the satellite image (e.g. 'Locate buildings at 28.6139, 77.2090' or 'Process SAR radar data')..."
          onKeyDown={(e) => e.key === 'Enter' && onExecute()}
          style={{
            width: '100%',
            padding: '16px 140px 16px 48px',
            backgroundColor: 'rgba(15, 23, 42, 0.9)',
            border: '1px solid var(--border-glass-accent)',
            borderRadius: 'var(--radius-md)',
            color: '#ffffff',
            fontSize: '0.95rem',
            fontFamily: 'var(--font-family-body)',
            outline: 'none',
            boxShadow: '0 0 15px rgba(14, 165, 233, 0.15)'
          }}
        />
        <Search
          size={20}
          color="var(--text-muted)"
          style={{ position: 'absolute', left: '16px', top: '50%', transform: 'translateY(-50%)' }}
        />

        <button
          onClick={onExecute}
          disabled={loading}
          className="btn-primary"
          style={{
            position: 'absolute',
            right: '8px',
            top: '50%',
            transform: 'translateY(-50%)',
            padding: '10px 20px',
            fontSize: '0.88rem'
          }}
        >
          {loading ? (
            <>
              <RefreshCw size={16} className="animate-spin-slow" /> Analyzing...
            </>
          ) : (
            <>
              <Zap size={16} /> Run AI Query
            </>
          )}
        </button>
      </div>

      {/* Preset Suggestions */}
      <div>
        <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '4px' }}>
          <Sparkles size={13} color="var(--accent-amber)" /> Suggested Satellite Queries:
        </p>
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
          {suggestions.map((s, idx) => (
            <button
              key={idx}
              onClick={() => setQuery(s.query)}
              className="chip"
              style={{ fontSize: '0.8rem' }}
            >
              <Target size={12} color="var(--primary-cyan)" /> {s.label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
