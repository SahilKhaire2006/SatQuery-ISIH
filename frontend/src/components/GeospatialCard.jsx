import React from 'react';
import { MapPin, Globe, Compass, Box } from 'lucide-react';

export default function GeospatialCard({ geoData }) {
  if (!geoData) return null;

  const bounds = geoData.bounds || [77.1000, 28.5000, 77.3000, 28.7000];
  const centerLat = geoData.center_lat ?? 28.6000;
  const centerLon = geoData.center_lon ?? 77.2000;
  const crs = geoData.crs || 'EPSG:4326';

  return (
    <div className="glass-panel" style={{ padding: '20px', borderLeft: '4px solid var(--primary-cyan)' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
        <h4 style={{ fontSize: '1.05rem', color: '#38bdf8', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <MapPin size={18} color="#38bdf8" /> Spatial Metadata & Coordinates
        </h4>
        <span className="badge badge-cyan">{crs}</span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '12px' }}>
        <div style={{ background: 'rgba(15, 23, 42, 0.8)', padding: '10px 14px', borderRadius: '8px', border: '1px solid var(--border-glass)' }}>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Center Latitude</p>
          <p style={{ fontSize: '0.95rem', fontWeight: '700', fontFamily: 'var(--font-family-mono)', color: '#ffffff' }}>
            {centerLat.toFixed(4)}° N
          </p>
        </div>

        <div style={{ background: 'rgba(15, 23, 42, 0.8)', padding: '10px 14px', borderRadius: '8px', border: '1px solid var(--border-glass)' }}>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Center Longitude</p>
          <p style={{ fontSize: '0.95rem', fontWeight: '700', fontFamily: 'var(--font-family-mono)', color: '#ffffff' }}>
            {centerLon.toFixed(4)}° E
          </p>
        </div>

        <div style={{ background: 'rgba(15, 23, 42, 0.8)', padding: '10px 14px', borderRadius: '8px', border: '1px solid var(--border-glass)' }}>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Spatial Extent Bounds</p>
          <p style={{ fontSize: '0.78rem', fontFamily: 'var(--font-family-mono)', color: 'var(--text-muted)' }}>
            [{bounds[0].toFixed(2)}, {bounds[1].toFixed(2)}, {bounds[2].toFixed(2)}, {bounds[3].toFixed(2)}]
          </p>
        </div>
      </div>
    </div>
  );
}
