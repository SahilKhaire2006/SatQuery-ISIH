import React, { useState } from 'react';
import { Terminal, Check, Clock, ShieldCheck, Layers, ChevronDown, ChevronUp, Cpu, Activity } from 'lucide-react';

export default function AuditTrailInspector({ auditLog, confidence }) {
  const [expanded, setExpanded] = useState(true);

  if (!auditLog) return null;

  const tools = auditLog.selected_tools || [];
  const execution = auditLog.execution || [];
  const validation = auditLog.validation || {};
  const interpretation = auditLog.interpretation || {};
  const evidenceSummary = auditLog.evidence_summary || [];

  return (
    <div className="glass-panel" style={{ padding: '20px' }}>
      {/* Audit Trail Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '14px' }}>
        <h4 style={{ fontSize: '1.05rem', display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }} onClick={() => setExpanded(!expanded)}>
          <Terminal size={18} color="var(--primary-cyan)" /> Agentic Audit Trail & Execution Inspector
          {expanded ? <ChevronUp size={16} color="var(--text-muted)" /> : <ChevronDown size={16} color="var(--text-muted)" />}
        </h4>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Overall System Confidence:</span>
            <span className="badge badge-cyan" style={{ fontSize: '0.85rem' }}>
              {((confidence || 0.85) * 100).toFixed(1)}%
            </span>
          </div>
          <button
            onClick={() => setExpanded(!expanded)}
            style={{
              background: 'rgba(30, 41, 59, 0.6)',
              border: '1px solid var(--border-glass)',
              color: 'var(--text-muted)',
              borderRadius: '4px',
              padding: '4px 8px',
              fontSize: '0.75rem',
              cursor: 'pointer'
            }}
          >
            {expanded ? 'Collapse Log' : 'Expand Log'}
          </button>
        </div>
      </div>

      {expanded && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
          {/* Top Info Grid */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '14px' }}>
            {/* Task Classification & Intent */}
            <div style={{ background: 'rgba(15, 23, 42, 0.7)', padding: '14px', borderRadius: '8px', border: '1px solid var(--border-glass)' }}>
              <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                <Layers size={14} color="var(--primary-cyan)" /> Task Classification & Intent
              </p>
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
              <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                <Cpu size={14} color="var(--primary-cyan)" /> Specialist Model Sequence
              </p>
              {tools.length > 0 ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                  {tools.map((t, idx) => (
                    <div key={idx} style={{ fontSize: '0.82rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between', background: 'rgba(30, 41, 59, 0.5)', padding: '4px 8px', borderRadius: '4px' }}>
                      <span style={{ color: '#e2e8f0' }}>{idx + 1}. {typeof t === 'string' ? t : (t.tool_name || t.tool_id)}</span>
                      <span style={{ color: 'var(--accent-emerald)', fontSize: '0.75rem', display: 'flex', alignItems: 'center', gap: '4px' }}>
                        <Check size={12} /> Executed
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <p style={{ fontSize: '0.8rem', color: 'var(--text-dim)' }}>Default model sequence active.</p>
              )}
            </div>
          </div>

          {/* Execution Trace Timeline */}
          <div style={{ background: 'rgba(7, 10, 18, 0.85)', padding: '14px', borderRadius: '8px', border: '1px solid var(--border-glass)' }}>
            <p style={{ fontSize: '0.8rem', color: 'var(--primary-cyan)', fontWeight: '600', marginBottom: '10px', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Activity size={14} /> Agentic Execution Step Log
            </p>
            <div style={{ fontFamily: 'var(--font-family-mono)', fontSize: '0.78rem', color: '#cbd5e1', display: 'flex', flexDirection: 'column', gap: '6px' }}>
              <div style={{ borderLeft: '2px solid #0ea5e9', paddingLeft: '8px' }}>
                <span style={{ color: '#38bdf8' }}>[Step 1 Input Validation]</span> Image Valid: {validation.valid ? 'Yes' : 'No'} {validation.image_shape ? `(Shape: ${validation.image_shape.join('x')})` : ''}
              </div>
              <div style={{ borderLeft: '2px solid #8b5cf6', paddingLeft: '8px' }}>
                <span style={{ color: '#a78bfa' }}>[Step 2 Query Routing]</span> Task: {interpretation.task_type || 'VQA'} | Intent: {interpretation.intent || 'detection'}
              </div>
              {execution.map((item, idx) => (
                <div key={idx} style={{ borderLeft: '2px solid #10b981', paddingLeft: '8px' }}>
                  <span style={{ color: '#34d399' }}>[Step 3 Model Execution - {item.tool_id}]</span> Status: {item.status} | Conf: {((item.confidence || 0.85) * 100).toFixed(0)}% {item.execution_time ? `| Time: ${(item.execution_time).toFixed(2)}s` : ''}
                </div>
              ))}
              <div style={{ borderLeft: '2px solid #f59e0b', paddingLeft: '8px' }}>
                <span style={{ color: '#fbbf24' }}>[Step 4 Evidence Compiler]</span> Compiled {evidenceSummary.length} evidence source(s) with multi-modal visual grounding overlays.
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
