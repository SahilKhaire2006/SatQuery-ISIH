import React from 'react';
import { Satellite, ShieldCheck, Activity, Cpu, Sparkles } from 'lucide-react';

export default function Header({ apiHealthy, sessionID, activeTab, setActiveTab }) {
  return (
    <header className="glass-panel" style={{ padding: '16px 28px', marginBottom: '24px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '16px' }}>
        
        {/* Brand Logo & Tagline */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
          <div style={{
            background: 'linear-gradient(135deg, #0284c7 0%, #7c3aed 100%)',
            padding: '10px',
            borderRadius: '12px',
            boxShadow: '0 0 15px rgba(14, 165, 233, 0.4)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            <Satellite size={26} color="#ffffff" className="animate-spin-slow" />
          </div>

          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <h1 style={{ fontSize: '1.5rem', fontWeight: '800' }}>
                Sat<span className="gradient-text">Query</span>
              </h1>
              <span className="badge badge-cyan" style={{ fontSize: '0.7rem' }}>v2.0 Production</span>
            </div>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
              Agentic Satellite Image Query & Evidence-Grounded AI System
            </p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div style={{ display: 'flex', gap: '8px', background: 'rgba(15, 23, 42, 0.6)', padding: '6px', borderRadius: '12px', border: '1px solid var(--border-glass)' }}>
          <button
            onClick={() => setActiveTab('analysis')}
            className={activeTab === 'analysis' ? 'btn-primary' : 'btn-secondary'}
            style={{ padding: '8px 16px', fontSize: '0.85rem' }}
          >
            <Sparkles size={16} /> Analysis Workspace
          </button>
          <button
            onClick={() => setActiveTab('metrics')}
            className={activeTab === 'metrics' ? 'btn-primary' : 'btn-secondary'}
            style={{ padding: '8px 16px', fontSize: '0.85rem' }}
          >
            <Activity size={16} /> Telemetry & Metrics
          </button>
        </div>

        {/* System Health Badges */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', background: 'rgba(15, 23, 42, 0.8)', padding: '6px 14px', borderRadius: '20px', border: '1px solid var(--border-glass)' }}>
            <Cpu size={15} color="var(--primary-cyan)" />
            <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', fontFamily: 'var(--font-family-mono)' }}>
              LLM: Groq Llama-3.3
            </span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span
              style={{
                width: '10px',
                height: '10px',
                borderRadius: '50%',
                backgroundColor: apiHealthy ? 'var(--accent-emerald)' : 'var(--accent-rose)',
                boxShadow: apiHealthy ? '0 0 10px var(--accent-emerald)' : '0 0 10px var(--accent-rose)'
              }}
            />
            <span style={{ fontSize: '0.8rem', color: apiHealthy ? 'var(--accent-emerald)' : 'var(--accent-rose)', fontWeight: '600' }}>
              {apiHealthy ? 'API Online' : 'Connecting...'}
            </span>
          </div>
        </div>

      </div>
    </header>
  );
}
