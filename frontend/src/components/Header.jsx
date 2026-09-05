import React, { useState } from 'react';
import { Share2, Bookmark, MoreVertical, PanelRight, Check } from 'lucide-react';

export default function Header({ currentTitle, onToggleHistory, isHistoryOpen }) {
  const [shared, setShared] = useState(false);
  const [bookmarked, setBookmarked] = useState(false);

  const handleShare = () => {
    navigator.clipboard.writeText(window.location.href);
    setShared(true);
    setTimeout(() => setShared(false), 2000);
  };

  return (
    <header className="main-chat-header">
      {/* Title */}
      <div className="chat-header-title">
        <h2>{currentTitle || 'SatQuery Satellite Intelligence'}</h2>
      </div>

      {/* Actions (Share, Bookmark, Options, History Toggle) */}
      <div className="chat-header-actions">
        <button onClick={handleShare} className="header-action-btn">
          {shared ? <Check size={14} color="var(--accent-emerald)" /> : <Share2 size={14} />}
          <span>{shared ? 'Copied' : 'Share'}</span>
        </button>

        <button 
          onClick={() => setBookmarked(!bookmarked)} 
          className={`header-action-btn ${bookmarked ? 'active' : ''}`}
          title="Bookmark Query"
        >
          <Bookmark size={14} color={bookmarked ? 'var(--accent-amber)' : 'currentColor'} />
        </button>

        <button 
          onClick={onToggleHistory}
          className={`header-action-btn ${isHistoryOpen ? 'active' : ''}`}
          title="Toggle History Sidebar"
        >
          <PanelRight size={14} />
        </button>
      </div>
    </header>
  );
}


