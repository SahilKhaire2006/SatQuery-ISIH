import React from 'react';
import { 
  Satellite, 
  Plus, 
  Compass, 
  Layers, 
  Activity, 
  Settings, 
  Zap, 
  ArrowUpRight, 
  ShieldCheck, 
  Cpu
} from 'lucide-react';

export default function LeftSidebar({ 
  onNewChat, 
  activeNav, 
  setActiveNav, 
  apiHealthy 
}) {
  return (
    <aside className="left-sidebar">
      {/* Brand Header */}
      <div className="sidebar-brand">
        <div className="brand-logo-box">
          <Satellite size={22} color="#ffffff" className="animate-spin-slow" />
        </div>
        <div>
          <div className="brand-title">
            Sat<span className="gradient-text">Query</span>
          </div>
          <div className="brand-subtitle">AI Satellite Grounding</div>
        </div>
      </div>

      {/* New Chat Button */}
      <button onClick={onNewChat} className="btn-new-chat">
        <Plus size={18} /> New chat
      </button>

      {/* Main Navigation Links */}
      <nav className="sidebar-nav">
        <button
          onClick={() => setActiveNav('analysis')}
          className={`nav-item ${activeNav === 'analysis' ? 'active' : ''}`}
        >
          <Compass size={18} /> Explore
        </button>

        <button
          onClick={() => setActiveNav('templates')}
          className={`nav-item ${activeNav === 'templates' ? 'active' : ''}`}
        >
          <Layers size={18} /> Templates
        </button>

        <button
          onClick={() => setActiveNav('metrics')}
          className={`nav-item ${activeNav === 'metrics' ? 'active' : ''}`}
        >
          <Activity size={18} /> Telemetry
        </button>

        <button
          onClick={() => setActiveNav('settings')}
          className={`nav-item ${activeNav === 'settings' ? 'active' : ''}`}
        >
          <Settings size={18} /> Settings
        </button>
      </nav>

      {/* Bottom Feature Card (Reference Image "Premium Plan" Banner) */}
      <div className="sidebar-banner-card">
        <div className="banner-title">Model 2 Grounding</div>
        <div className="banner-subtitle">
          Real-time Sentinel-2, Sentinel-1 SAR & Esri satellite imagery engine
        </div>
        <button 
          onClick={() => setActiveNav('metrics')}
          className="banner-btn"
        >
          Engine Status <ArrowUpRight size={14} />
        </button>
      </div>

      {/* System Status Footer */}
      <div className="sidebar-footer">
        <div className="status-indicator">
          <span 
            className="status-dot" 
            style={{ 
              backgroundColor: apiHealthy ? 'var(--accent-emerald)' : 'var(--accent-rose)',
              boxShadow: apiHealthy ? '0 0 8px var(--accent-emerald)' : '0 0 8px var(--accent-rose)'
            }} 
          />
          <span>{apiHealthy ? 'API Online • Groq LLM' : 'Connecting...'}</span>
        </div>
      </div>
    </aside>
  );
}
