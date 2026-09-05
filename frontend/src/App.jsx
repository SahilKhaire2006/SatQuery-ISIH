import React, { useState, useEffect, useRef } from 'react';
import LeftSidebar from './components/LeftSidebar';
import HistorySidebar from './components/HistorySidebar';
import Header from './components/Header';
import VisualEvidenceViewer from './components/VisualEvidenceViewer';
import SARFusionCard from './components/SARFusionCard';
import AuditTrailInspector from './components/AuditTrailInspector';
import GeospatialCard from './components/GeospatialCard';
import DisasterAnalysisCard from './components/DisasterAnalysisCard';
import { 
  Paperclip, 
  Send, 
  Sparkles, 
  RefreshCw, 
  Activity, 
  ShieldAlert, 
  Layers, 
  Building, 
  X, 
  AlertCircle,
  Brain,
  Globe,
  Code,
  MoreHorizontal
} from 'lucide-react';

export default function App() {
  const [activeNav, setActiveNav] = useState('analysis');
  const [isHistoryOpen, setIsHistoryOpen] = useState(true);
  const [apiHealthy, setApiHealthy] = useState(true);
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState([]);
  const [chatHistory, setChatHistory] = useState([]);
  const [metricsData, setMetricsData] = useState(null);
  const [sessionID, setSessionID] = useState(() => 'sess_' + Math.random().toString(36).substring(2, 9));
  const [currentTitle, setCurrentTitle] = useState('SatQuery Satellite Intelligence');

  const fileInputRef = useRef(null);
  const chatEndRef = useRef(null);

  // Check API health on load
  useEffect(() => {
    fetch('/api/v1/health')
      .then((res) => res.json())
      .then((data) => setApiHealthy(data.status === 'healthy' || data.status === 'degraded'))
      .catch(() => setApiHealthy(false));
  }, []);

  // Fetch metrics when Telemetry tab is opened
  useEffect(() => {
    if (activeNav === 'metrics') {
      fetch('/metrics')
        .then((res) => res.text())
        .then((text) => setMetricsData(text))
        .catch(() => setMetricsData('Unable to reach Prometheus /metrics endpoint.'));
    }
  }, [activeNav]);

  // Auto-scroll to bottom of chat
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const handleNewChat = () => {
    setMessages([]);
    setSelectedFile(null);
    setPreviewUrl(null);
    setQuery('');
    setCurrentTitle('SatQuery Satellite Intelligence');
    setSessionID('sess_' + Math.random().toString(36).substring(2, 9));
  };

  const handleFileSelect = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
    }
  };

  const handleRemoveImage = (e) => {
    if (e) e.stopPropagation();
    setSelectedFile(null);
    setPreviewUrl(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleSelectHistorySession = (sessId) => {
    const found = chatHistory.find((item) => item.session_id === sessId);
    if (found && found.messages) {
      setSessionID(found.session_id);
      setMessages(found.messages);
      setCurrentTitle(found.title);
    }
  };

  const handleDeleteHistorySession = (sessId) => {
    setChatHistory((prev) => prev.filter((item) => item.session_id !== sessId));
  };

  const handleClearAllHistory = () => {
    setChatHistory([]);
  };

  const handleExecuteQuery = async (customQuery = null, customFile = null) => {
    const activeQuery = (customQuery || query).trim();
    if (!activeQuery) return;

    const fileToUpload = customFile || selectedFile;
    const currentPreview = customFile ? URL.createObjectURL(customFile) : previewUrl;

    if (messages.length === 0) {
      setCurrentTitle(activeQuery.length > 32 ? activeQuery.substring(0, 32) + '...' : activeQuery);
    }

    // Create user message entry
    const userMsgId = 'usr_' + Date.now();
    const userMessage = {
      id: userMsgId,
      sender: 'user',
      text: activeQuery,
      attachedImagePreview: currentPreview,
      fileName: fileToUpload?.name || null,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    // Create placeholder assistant message entry
    const aiMsgId = 'ai_' + Date.now();
    const initialAiMessage = {
      id: aiMsgId,
      sender: 'assistant',
      loading: true,
      loadingStep: 'Step 1: Preprocessing & Spatial Validation...',
      responseData: null,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    const newMessages = [...messages, userMessage, initialAiMessage];
    setMessages(newMessages);
    setQuery('');
    setSelectedFile(null);
    setPreviewUrl(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
    setLoading(true);

    try {
      const formData = new FormData();
      formData.append('query', activeQuery);
      formData.append('session_id', sessionID);

      // If no file selected, create synthetic dummy image blob
      let payloadFile = fileToUpload;
      if (!payloadFile) {
        const canvas = document.createElement('canvas');
        canvas.width = 100;
        canvas.height = 100;
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#0b0f17';
        ctx.fillRect(0, 0, 100, 100);
        const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/png'));
        payloadFile = new File([blob], 'synthetic_sat.png', { type: 'image/png' });
      }

      formData.append('image', payloadFile);

      // Progress animation steps
      setTimeout(() => {
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === aiMsgId ? { ...msg, loadingStep: 'Step 2: Groq LLM Spatial & Tool Intent Extraction...' } : msg
          )
        );
      }, 500);

      setTimeout(() => {
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === aiMsgId ? { ...msg, loadingStep: 'Step 3: Vision Model Tool Routing (SAR / Flood / Roboflow)...' } : msg
          )
        );
      }, 1000);

      setTimeout(() => {
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === aiMsgId ? { ...msg, loadingStep: 'Step 4: Inference & Visual Evidence Overlay Synthesis...' } : msg
          )
        );
      }, 1500);

      const res = await fetch('/api/v1/query', {
        method: 'POST',
        body: formData,
      });

      const data = await res.json();

      setMessages((prev) => {
        const updated = prev.map((msg) =>
          msg.id === aiMsgId
            ? {
                ...msg,
                loading: false,
                loadingStep: '',
                responseData: data,
              }
            : msg
        );

        // Save to History Sidebar
        setChatHistory((hPrev) => {
          const titleSnippet = activeQuery.length > 28 ? activeQuery.substring(0, 28) + '...' : activeQuery;
          const existingIdx = hPrev.findIndex((item) => item.session_id === sessionID);
          const historyEntry = {
            session_id: sessionID,
            title: titleSnippet,
            snippet: data?.results?.aggregated_summary ? data.results.aggregated_summary.substring(0, 45) + '...' : 'Satellite analysis complete',
            timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
            messages: updated
          };

          if (existingIdx >= 0) {
            const clone = [...hPrev];
            clone[existingIdx] = historyEntry;
            return clone;
          }
          return [historyEntry, ...hPrev];
        });

        return updated;
      });
    } catch (err) {
      console.error('API execution error:', err);
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === aiMsgId
            ? {
                ...msg,
                loading: false,
                loadingStep: '',
                error: 'Failed to connect to SatQuery API backend. Ensure FastAPI server is running.',
              }
            : msg
        )
      );
    } finally {
      setLoading(false);
    }
  };

  const heroPresets = [
    {
      icon: <ShieldAlert size={20} color="var(--accent-amber)" />,
      title: "Wayanad Flood & Evacuation",
      query: "Show flood extent, NDWI water detection, and evacuation routes in Wayanad, Kerala",
      badge: "Model 2 Real-Time"
    },
    {
      icon: <Activity size={20} color="var(--accent-rose)" />,
      title: "Kathmandu Structural Shift",
      query: "Perform earthquake structural shift and damage heatmap analysis in Kathmandu, Nepal",
      badge: "Earthquake Grounding"
    },
    {
      icon: <Layers size={20} color="var(--accent-violet)" />,
      title: "Sentinel-1 SAR Radar Fusion",
      query: "Process Sentinel-1 SAR radar imagery and fuse VV/VH polarizations with optical data",
      badge: "USP-1 SAR Fusion"
    },
    {
      icon: <Building size={20} color="var(--primary-cyan)" />,
      title: "Roboflow Building Localization",
      query: "Count the buildings and structures in this imagery and highlight bounding boxes",
      badge: "Object Detection"
    }
  ];

  const quickPills = [
    { icon: <Brain size={14} color="var(--accent-amber)" />, label: "Brainstorm" },
    { icon: <Globe size={14} color="var(--primary-cyan)" />, label: "Web search" },
    { icon: <Code size={14} color="var(--accent-emerald)" />, label: "Code" },
    { icon: <MoreHorizontal size={14} color="var(--text-muted)" />, label: "More" },
  ];

  return (
    <div className={`dashboard-layout ${isHistoryOpen ? '' : 'history-closed'}`}>
      
      {/* 1. Left Navigation Sidebar */}
      <LeftSidebar
        onNewChat={handleNewChat}
        activeNav={activeNav}
        setActiveNav={setActiveNav}
        apiHealthy={apiHealthy}
      />

      {/* 2. Center Main Chat Workspace */}
      <main className="main-chat-area">
        <Header
          currentTitle={currentTitle}
          onToggleHistory={() => setIsHistoryOpen(!isHistoryOpen)}
          isHistoryOpen={isHistoryOpen}
        />

        {activeNav === 'analysis' || activeNav === 'templates' ? (
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', position: 'relative', overflow: 'hidden' }}>
            
            {/* Scrollable Chat Feed Stream */}
            <div className="chat-feed-scroll">
              <div className="chat-feed-inner">
                
                {messages.length === 0 ? (
                  /* Hero Welcome View (ChatGPT / ChatEase Styled) */
                  <div style={{ textAlign: 'center', padding: '40px 0 20px 0' }}>
                    <div style={{
                      background: 'linear-gradient(135deg, rgba(2, 132, 199, 0.15) 0%, rgba(124, 58, 237, 0.15) 100%)',
                      width: '64px',
                      height: '64px',
                      borderRadius: '20px',
                      display: 'inline-flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      marginBottom: '20px',
                      border: '1px solid var(--border-glass-accent)',
                      boxShadow: '0 0 30px rgba(56, 189, 248, 0.2)'
                    }}>
                      <Sparkles size={32} color="var(--primary-cyan)" className="animate-pulse-glow" />
                    </div>

                    <h2 style={{ fontSize: '1.8rem', fontWeight: '800', marginBottom: '10px' }}>
                      SatQuery Intelligence Hub
                    </h2>
                    <p style={{ color: 'var(--text-muted)', fontSize: '0.92rem', marginBottom: '36px', maxWidth: '600px', margin: '0 auto 36px auto' }}>
                      Ask questions in natural language or attach satellite/drone images to trigger real-time Sentinel-1 SAR fusion, disaster grounding reports, or object detection.
                    </p>

                    {/* Grid of Preset Suggestion Cards */}
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '16px', textAlign: 'left' }}>
                      {heroPresets.map((card, idx) => (
                        <div
                          key={idx}
                          className="hero-preset-card"
                          onClick={() => handleExecuteQuery(card.query)}
                        >
                          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                            {card.icon}
                            <span className="badge badge-cyan" style={{ fontSize: '0.65rem' }}>{card.badge}</span>
                          </div>
                          <div>
                            <h4 style={{ fontSize: '0.95rem', fontWeight: '700', color: '#ffffff', marginBottom: '4px' }}>
                              {card.title}
                            </h4>
                            <p style={{ fontSize: '0.80rem', color: 'var(--text-muted)', lineHeight: '1.4', margin: 0 }}>
                              "{card.query}"
                            </p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : (
                  /* Active Chat Stream */
                  messages.map((msg) => (
                    <React.Fragment key={msg.id}>
                      {msg.sender === 'user' ? (
                        /* User Message Bubble */
                        <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                          <div className="chat-bubble-user">
                            {msg.attachedImagePreview && (
                              <div style={{ marginBottom: '10px', position: 'relative' }}>
                                <img
                                  src={msg.attachedImagePreview}
                                  alt="Attached Satellite Input"
                                  style={{
                                    maxHeight: '180px',
                                    maxWidth: '100%',
                                    borderRadius: '8px',
                                    border: '1px solid rgba(255,255,255,0.2)',
                                    boxShadow: '0 4px 12px rgba(0,0,0,0.4)'
                                  }}
                                />
                                {msg.fileName && (
                                  <div style={{ fontSize: '0.72rem', color: 'var(--primary-cyan)', marginTop: '4px', fontFamily: 'var(--font-family-mono)' }}>
                                    📷 {msg.fileName}
                                  </div>
                                )}
                              </div>
                            )}
                            <div>{msg.text}</div>
                            <div style={{ fontSize: '0.68rem', color: 'rgba(255,255,255,0.5)', textAlign: 'right', marginTop: '6px' }}>
                              {msg.timestamp}
                            </div>
                          </div>
                        </div>
                      ) : (
                        /* AI Assistant Message Bubble */
                        <div style={{ display: 'flex', gap: '14px', alignItems: 'flex-start' }}>
                          <div className="ai-avatar">
                            <Sparkles size={20} color="#ffffff" />
                          </div>

                          <div className="chat-bubble-ai">
                            {msg.loading ? (
                              <div style={{ padding: '12px 0' }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '12px' }}>
                                  <RefreshCw size={18} color="var(--primary-cyan)" className="animate-spin-slow" />
                                  <span style={{ fontWeight: '600', color: '#ffffff', fontSize: '0.90rem' }}>
                                    {msg.loadingStep}
                                  </span>
                                </div>
                                <div style={{ width: '100%', height: '4px', background: 'rgba(15, 23, 42, 0.8)', borderRadius: '2px', overflow: 'hidden' }}>
                                  <div style={{ width: '75%', height: '100%', background: 'linear-gradient(90deg, #0ea5e9, #8b5cf6, #10b981)', animation: 'pulseGlow 1.5s infinite' }} />
                                </div>
                              </div>
                            ) : msg.error ? (
                              <div style={{ background: 'rgba(239, 68, 68, 0.15)', border: '1px solid rgba(239, 68, 68, 0.4)', borderRadius: '12px', padding: '16px', color: '#fca5a5', display: 'flex', alignItems: 'center', gap: '10px' }}>
                                <AlertCircle size={20} color="#ef4444" />
                                <span>{msg.error}</span>
                              </div>
                            ) : (
                              <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                                
                                {/* Visual Evidence & Reasoning Tabs */}
                                <VisualEvidenceViewer
                                  results={msg.responseData?.results || {}}
                                  explanation={msg.responseData?.explanation || msg.responseData?.results?.aggregated_summary}
                                  visualEvidence={msg.responseData?.results?.visual_evidence || {}}
                                />

                                {/* Disaster Grounding Report */}
                                {msg.responseData?.results?.disaster_grounding_model?.output && (
                                  <DisasterAnalysisCard
                                    disasterData={msg.responseData.results.disaster_grounding_model.output}
                                  />
                                )}

                                {/* SAR Fusion & Geospatial Details */}
                                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '16px' }}>
                                  <SARFusionCard
                                    fusionInfo={
                                      msg.responseData?.results?.sar_fusion_model?.fusion_info ||
                                      msg.responseData?.results?.details?.sar_fusion_model?.fusion_info
                                    }
                                  />
                                  <GeospatialCard
                                    geoData={
                                      msg.responseData?.results?.geospatial_metadata ||
                                      msg.responseData?.audit_log?.geospatial
                                    }
                                  />
                                </div>

                                {/* Audit Trail Inspector */}
                                {msg.responseData && (
                                  <AuditTrailInspector
                                    auditLog={msg.responseData?.audit_log || {}}
                                    confidence={msg.responseData?.confidence}
                                  />
                                )}

                                <div style={{ fontSize: '0.70rem', color: 'var(--text-dim)', textAlign: 'right', marginTop: '4px' }}>
                                  SatQuery AI Grounding • {msg.timestamp}
                                </div>
                              </div>
                            )}
                          </div>
                        </div>
                      )}
                    </React.Fragment>
                  ))
                )}
                <div ref={chatEndRef} />
              </div>
            </div>

            {/* Fixed Bottom Input Workspace (Matching Reference Image) */}
            <div className="bottom-input-bar-container">
              <div className="bottom-input-inner">
                
                {/* Attached Image Preview Thumbnail */}
                {previewUrl && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', background: 'rgba(17, 20, 28, 0.95)', padding: '6px 12px', borderRadius: '12px', border: '1px solid var(--border-glass-accent)', width: 'fit-content' }}>
                    <img src={previewUrl} alt="Thumbnail preview" style={{ width: '28px', height: '28px', borderRadius: '4px', objectFit: 'cover' }} />
                    <span style={{ fontSize: '0.78rem', color: '#ffffff', maxWidth: '200px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {selectedFile?.name || 'Attached Satellite Image'}
                    </span>
                    <button onClick={handleRemoveImage} style={{ background: 'none', border: 'none', color: 'var(--accent-rose)', cursor: 'pointer', display: 'flex', alignItems: 'center', padding: '2px' }}>
                      <X size={14} />
                    </button>
                  </div>
                )}

                {/* Quick Action Pill Chips (Matching Reference Image "Brainstorm", "Web search", "Code", "More") */}
                <div className="input-pill-row">
                  {quickPills.map((pill, idx) => (
                    <button
                      key={idx}
                      className="action-pill"
                      onClick={() => handleExecuteQuery(`${pill.label}: ${query || 'Analyze satellite area'}`)}
                    >
                      {pill.icon}
                      <span>{pill.label}</span>
                    </button>
                  ))}
                </div>

                {/* Main Pitch Black Input Box */}
                <div className="main-input-box">
                  {/* File Upload Trigger */}
                  <input
                    type="file"
                    ref={fileInputRef}
                    onChange={handleFileSelect}
                    accept="image/*,.tif,.tiff"
                    style={{ display: 'none' }}
                  />

                  <button
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    className="btn-secondary"
                    style={{ padding: '8px 12px', borderRadius: '16px', fontSize: '0.80rem', borderColor: selectedFile ? 'var(--primary-cyan)' : 'var(--border-glass)' }}
                    title="Add Satellite / Aerial Image"
                  >
                    <Paperclip size={16} color={selectedFile ? 'var(--primary-cyan)' : 'var(--text-muted)'} />
                    <span>{selectedFile ? 'Attached' : 'Add Image'}</span>
                  </button>

                  <textarea
                    className="chat-textarea"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        handleExecuteQuery();
                      }
                    }}
                    placeholder="Ask me something....."
                    rows={1}
                  />

                  <button
                    type="button"
                    onClick={() => handleExecuteQuery()}
                    disabled={loading || !query.trim()}
                    className="btn-send"
                    title="Send Query"
                  >
                    {loading ? (
                      <RefreshCw size={18} className="animate-spin-slow" />
                    ) : (
                      <Send size={18} />
                    )}
                  </button>
                </div>

              </div>
            </div>

          </div>
        ) : (
          /* Telemetry & Metrics View */
          <div style={{ flex: 1, padding: '28px', overflowY: 'auto' }}>
            <div className="glass-panel" style={{ padding: '32px' }}>
              <h2 style={{ fontSize: '1.4rem', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '10px' }}>
                <Activity size={24} color="var(--primary-cyan)" /> Prometheus Telemetry & System Diagnostics
              </h2>
              <p style={{ color: 'var(--text-muted)', marginBottom: '20px', fontSize: '0.9rem' }}>
                Real-time telemetry output from SatQuery API endpoint <code style={{ color: '#38bdf8' }}>/metrics</code>:
              </p>

              <pre style={{
                background: '#07090e',
                padding: '20px',
                borderRadius: 'var(--radius-md)',
                border: '1px solid var(--border-glass)',
                fontFamily: 'var(--font-family-mono)',
                fontSize: '0.85rem',
                color: '#34d399',
                maxHeight: '520px',
                overflowY: 'auto',
                whiteSpace: 'pre-wrap'
              }}>
                {metricsData || 'Loading telemetry...'}
              </pre>
            </div>
          </div>
        )}
      </main>

      {/* 3. Right History Drawer */}
      <HistorySidebar
        chatHistory={chatHistory}
        activeSessionID={sessionID}
        onSelectSession={handleSelectHistorySession}
        onDeleteSession={handleDeleteHistorySession}
        onClearAllHistory={handleClearAllHistory}
      />

    </div>
  );
}

