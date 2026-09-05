import React from 'react';
import { Folder, Trash2, MessageSquare, Clock, X } from 'lucide-react';

export default function HistorySidebar({
  chatHistory = [],
  activeSessionID,
  onSelectSession,
  onDeleteSession,
  onClearAllHistory
}) {
  return (
    <aside className="right-history-sidebar">
      {/* History Header */}
      <div className="history-header">
        <h3 className="history-title">History</h3>
        <span className="badge badge-cyan" style={{ fontSize: '0.70rem' }}>
          {chatHistory.length}
        </span>
      </div>

      {/* Scrollable History List */}
      <div className="history-list">
        {chatHistory.length === 0 ? (
          <div className="history-empty">
            <Clock size={28} color="var(--text-dim)" style={{ marginBottom: '8px' }} />
            <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>No saved queries yet</p>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>Your satellite query sessions will appear here.</p>
          </div>
        ) : (
          chatHistory.map((item) => (
            <div
              key={item.session_id}
              onClick={() => onSelectSession(item.session_id)}
              className={`history-item ${item.session_id === activeSessionID ? 'active' : ''}`}
            >
              <div className="history-item-icon">
                <Folder size={16} color="var(--primary-cyan)" />
              </div>
              <div className="history-item-content">
                <div className="history-item-title">{item.title || 'Satellite Analysis Query'}</div>
                <div className="history-item-snippet">{item.snippet || 'Analyzed satellite imagery'}</div>
                <div className="history-item-time">{item.timestamp}</div>
              </div>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onDeleteSession(item.session_id);
                }}
                className="history-item-delete"
                title="Delete query"
              >
                <X size={13} />
              </button>
            </div>
          ))
        )}
      </div>

      {/* Delete History Action Button (Matching Reference Image) */}
      <div className="history-footer">
        <button
          onClick={onClearAllHistory}
          disabled={chatHistory.length === 0}
          className="btn-delete-history"
        >
          <Trash2 size={15} color="#ef4444" /> Delete history
        </button>
      </div>
    </aside>
  );
}
