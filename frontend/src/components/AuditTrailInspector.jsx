import React from 'react';
import { Terminal, Check, Clock, ShieldCheck, Layers } from 'lucide-react';

export default function AuditTrailInspector({ auditLog, confidence }) {
  if (!auditLog) return null;

  const tools = auditLog.selected_tools || [];
  const execution = auditLog.execution || [];
  const validation = auditLog.validation || {};
  const interpretation = auditLog.interpretation || {};

  return (
    <div className="glass-panel" style={{ padding: '20px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '14px' }}>
        <h4 style={{ fontSize: '1.05rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Terminal size={18} color="var(--primary-cyan)" /> Agentic Audit Trail & Execution Inspector
        </h4>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Overall System Confidence:</span>
          <span className="badge badge-cyan" style={{ fontSize: '0.85rem' }}>
            {((confidence || 0.85) * 100).toFixed(1)}%
          </span>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '14px' }}>
        {/* Task Classification & Intent */}
        <div style={{ background: 'rgba(15, 23, 42, 0.7)', padding: '14px', borderRadius: '8px', border: '1px solid var(--border-glass)' }}>
          <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: '6px' }}>Task Classification & Intent</p>
          <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
            <span className="badge badge-cyan">{interpretation.task_type || 'VQA'}</span>
            <span className="badge badge-violet">{interpretation.intent || 'General Query'}</span>
            {interpretation.target_object && (
              <span className="badge badge-amber">Target: {interpretation.target_object}</span>
            )}
          </div>
          {interpretation.reasoning && (
            <p style={{ fontSize: '0.78rem', color: 'var(--text-dim)', marginTop: '8px', fontStyle: 'italic' }}>
              "{interpretation.reasoning}"
            </p>
          )}
        </div>

        {/* Selected Specialist Tools Sequence */}
        <div style={{ background: 'rgba(15, 23, 42, 0.7)', padding: '14px', borderRadius: '8px', border: '1px solid var(--border-glass)' }}>
          <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: '6px' }}>Specialist Model Sequence</p>
          {tools.length > 0 ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
              {tools.map((t, idx) => (
                <div key={idx} style={{ fontSize: '0.82rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between', background: 'rgba(30, 41, 59, 0.5)', padding: '4px 8px', borderRadius: '4px' }}>
                  <span>{idx + 1}. {t.tool_name || t.tool_id}</span>
                  <span style={{ color: 'var(--accent-emerald)', fontSize: '0.75rem' }}><Check size={12} inline /> Ready</span>
                </div>
              ))}
            </div>
          ) : (
            <p style={{ fontSize: '0.8rem', color: 'var(--text-dim)' }}>Default model sequence active.</p>
          )}
        </div>
      </div>
    </div>
  );
}
