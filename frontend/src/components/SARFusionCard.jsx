import React from 'react';
import { Radio, ShieldCheck, Waves, Sun, Database } from 'lucide-react';

export default function SARFusionCard({ fusionInfo }) {
  if (!fusionInfo) return null;

  const sar = fusionInfo.sar_features || {};
  const optical = fusionInfo.optical_features || {};
  const fused = fusionInfo.fused_analysis || {};

  return (
    <div className="glass-panel" style={{ padding: '20px', borderLeft: '4px solid var(--accent-violet)' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
        <h4 style={{ fontSize: '1.05rem', color: '#c084fc', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Radio size={18} color="#c084fc" /> Sentinel-1 SAR & Optical Fusion (USP-1)
        </h4>
        <span className="badge badge-violet">Radar Multi-Modal</span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '12px' }}>
        {/* SAR Radar Backscatter */}
        <div style={{ background: 'rgba(15, 23, 42, 0.8)', padding: '12px', borderRadius: '8px', border: '1px solid var(--border-glass)' }}>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '4px' }}>
            <Waves size={14} color="#38bdf8" /> Radar Backscatter
          </p>
          <div style={{ marginTop: '4px', fontSize: '0.9rem', fontWeight: '700', fontFamily: 'var(--font-family-mono)' }}>
            VV: <span style={{ color: '#38bdf8' }}>{sar.sar_vv_backscatter_db ?? -12.5} dB</span>
          </div>
          <div style={{ fontSize: '0.9rem', fontWeight: '700', fontFamily: 'var(--font-family-mono)' }}>
            VH: <span style={{ color: '#c084fc' }}>{sar.sar_vh_backscatter_db ?? -18.0} dB</span>
          </div>
        </div>

        {/* Optical NDVI/NDWI Spectral Indices */}
        <div style={{ background: 'rgba(15, 23, 42, 0.8)', padding: '12px', borderRadius: '8px', border: '1px solid var(--border-glass)' }}>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '4px' }}>
            <Sun size={14} color="#fbbf24" /> Spectral Indices
          </p>
          <div style={{ marginTop: '4px', fontSize: '0.85rem' }}>
            NDVI Vegetation: <strong>{optical.ndvi_approx ?? 0.42}</strong>
          </div>
          <div style={{ fontSize: '0.85rem' }}>
            NDWI Water: <strong>{optical.ndwi_approx ?? -0.15}</strong>
          </div>
        </div>

        {/* Fused Classification & USP-1 Tag */}
        <div style={{ background: 'rgba(15, 23, 42, 0.8)', padding: '12px', borderRadius: '8px', border: '1px solid var(--border-glass)' }}>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '4px' }}>
            <ShieldCheck size={14} color="#34d399" /> Land-Cover Class
          </p>
          <div style={{ marginTop: '4px', fontSize: '0.9rem', fontWeight: '700', color: '#34d399', textTransform: 'capitalize' }}>
            {(fused.classified_land_cover || 'Multi-Modal Terrain').replace(/_/g, ' ')}
          </div>
          <span style={{ fontSize: '0.7rem', color: 'var(--text-dim)' }}>All-Weather Radar Penetration Active</span>
        </div>
      </div>
    </div>
  );
}
