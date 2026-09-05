import React from 'react';
import { ShieldAlert, AlertTriangle, Activity, MapPin, Compass, ShieldCheck } from 'lucide-react';

export default function DisasterAnalysisCard({ disasterData }) {
  if (!disasterData) return null;

  const {
    disaster_type = 'Disaster',
    escalation_level = 'ADVISORY',
    location = {},
    flood_extent = {},
    earthquake_analysis = {},
    evacuation_zones = {},
    recommendations = [],
    visual_evidence = {},
  } = disasterData;

  const getEscalationBadge = (level) => {
    switch (level?.toUpperCase()) {
      case 'CATASTROPHIC':
        return { bg: '#ef4444', text: '#ffffff', label: 'CATASTROPHIC EMERGENCY' };
      case 'EMERGENCY':
        return { bg: '#f97316', text: '#ffffff', label: 'EMERGENCY ALERT' };
      case 'WARNING':
        return { bg: '#eab308', text: '#000000', label: 'WARNING ADVISORY' };
      default:
        return { bg: '#3b82f6', text: '#ffffff', label: 'MONITORING ADVISORY' };
    }
  };

  const badge = getEscalationBadge(escalation_level);

  return (
    <div className="glass-panel" style={{ padding: '24px', marginTop: '20px' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
        <h3 style={{ fontSize: '1.2rem', display: 'flex', alignItems: 'center', gap: '8px', margin: 0 }}>
          <ShieldAlert size={22} color="var(--accent-amber)" />
          {disaster_type.toUpperCase()} DISASTER GROUNDING REPORT
        </h3>
        <span
          style={{
            backgroundColor: badge.bg,
            color: badge.text,
            padding: '6px 14px',
            borderRadius: '20px',
            fontSize: '0.80rem',
            fontWeight: '700',
            letterSpacing: '0.5px',
          }}
        >
          {badge.label}
        </span>
      </div>

      {/* Location Bar */}
      {location.name && (
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.90rem', color: 'var(--text-muted)', marginBottom: '16px' }}>
          <MapPin size={16} color="var(--primary-cyan)" />
          <span>{location.name}</span>
          {location.coordinates && (
            <span style={{ fontSize: '0.80rem', opacity: 0.8 }}>
              ({location.coordinates[0]?.toFixed(4)}°N, {location.coordinates[1]?.toFixed(4)}°E)
            </span>
          )}
        </div>
      )}

      {/* Grid of Analysis Metrics */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '12px', marginBottom: '20px' }}>
        {flood_extent.flooded_area_km2 !== undefined && (
          <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: '12px 16px', borderRadius: '8px', borderLeft: '4px solid #38bdf8' }}>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>Flooded Area</div>
            <div style={{ fontSize: '1.3rem', fontWeight: 'bold', color: '#38bdf8' }}>{flood_extent.flooded_area_km2} km²</div>
            <div style={{ fontSize: '0.75rem', opacity: 0.8 }}>{flood_extent.flood_percentage}% of area</div>
          </div>
        )}

        {earthquake_analysis.change_percentage !== undefined && (
          <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: '12px 16px', borderRadius: '8px', borderLeft: '4px solid #ef4444' }}>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>Structural Shift</div>
            <div style={{ fontSize: '1.3rem', fontWeight: 'bold', color: '#ef4444' }}>{earthquake_analysis.change_percentage}%</div>
            <div style={{ fontSize: '0.75rem', opacity: 0.8 }}>Severity: {earthquake_analysis.severity}</div>
          </div>
        )}

        {evacuation_zones.danger_zone_pct !== undefined && (
          <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: '12px 16px', borderRadius: '8px', borderLeft: '4px solid #dc2626' }}>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>🔴 Danger Zone (&lt;100m)</div>
            <div style={{ fontSize: '1.3rem', fontWeight: 'bold', color: '#dc2626' }}>{evacuation_zones.danger_zone_pct}%</div>
            <div style={{ fontSize: '0.75rem', opacity: 0.8 }}>Immediate evacuation</div>
          </div>
        )}

        {evacuation_zones.safe_zone_pct !== undefined && (
          <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: '12px 16px', borderRadius: '8px', borderLeft: '4px solid #22c55e' }}>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>🟢 Safe High Ground (&gt;500m)</div>
            <div style={{ fontSize: '1.3rem', fontWeight: 'bold', color: '#22c55e' }}>{evacuation_zones.safe_zone_pct}%</div>
            <div style={{ fontSize: '0.75rem', opacity: 0.8 }}>Assembly perimeters</div>
          </div>
        )}
      </div>

      {/* Evacuation Recommendations */}
      {recommendations.length > 0 && (
        <div style={{ background: 'rgba(30, 41, 59, 0.5)', padding: '16px', borderRadius: '8px', marginBottom: '16px' }}>
          <h4 style={{ fontSize: '0.90rem', margin: '0 0 8px 0', display: 'flex', alignItems: 'center', gap: '6px' }}>
            <ShieldCheck size={16} color="#22c55e" /> Emergency Action Guidance
          </h4>
          <ul style={{ margin: 0, paddingLeft: '20px', fontSize: '0.85rem', lineHeight: '1.6' }}>
            {recommendations.map((rec, i) => (
              <li key={i}>{rec}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
