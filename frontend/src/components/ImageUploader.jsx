import React, { useRef, useState } from 'react';
import { UploadCloud, Image as ImageIcon, CheckCircle2, Layers, X, FileText } from 'lucide-react';

export default function ImageUploader({ selectedFile, setSelectedFile, previewUrl, setPreviewUrl, geoMetadata, setGeoMetadata }) {
  const fileInputRef = useRef(null);
  const [dragActive, setDragActive] = useState(false);
  const [imgDimensions, setImgDimensions] = useState(null);

  const handleFile = (file) => {
    if (file && (file.type.startsWith('image/') || file.name.endsWith('.tif') || file.name.endsWith('.tiff'))) {
      setSelectedFile(file);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);

      const img = new Image();
      img.onload = () => {
        setImgDimensions({ width: img.width, height: img.height });
      };
      img.src = url;
    }
  };

  const handleClear = (e) => {
    e.stopPropagation();
    setSelectedFile(null);
    setPreviewUrl(null);
    setImgDimensions(null);
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const loadSampleImage = (name, color, type) => {
    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 384;
    const ctx = canvas.getContext('2d');
    
    ctx.fillStyle = color;
    ctx.fillRect(0, 0, 512, 384);
    
    // Draw synthetic terrain/structures
    ctx.fillStyle = 'rgba(255, 255, 255, 0.2)';
    ctx.fillRect(80, 80, 140, 100);
    ctx.fillRect(260, 180, 160, 90);
    
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 18px Outfit';
    ctx.fillText(`SatQuery Sample: ${name}`, 24, 40);

    canvas.toBlob((blob) => {
      const file = new File([blob], `${name.toLowerCase().replace(/\s+/g, '_')}.png`, { type: 'image/png' });
      setSelectedFile(file);
      const url = canvas.toDataURL();
      setPreviewUrl(url);
      setImgDimensions({ width: 512, height: 384 });
    }, 'image/png');
  };

  return (
    <div className="glass-panel" style={{ padding: '24px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
        <h3 style={{ fontSize: '1.1rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <UploadCloud size={20} color="var(--primary-cyan)" /> Satellite Imagery Input
        </h3>
        {selectedFile && (
          <button onClick={handleClear} className="btn-secondary" style={{ padding: '4px 10px', fontSize: '0.75rem', color: 'var(--accent-rose)' }}>
            <X size={13} /> Clear Image
          </button>
        )}
      </div>

      {/* Drag and Drop Zone */}
      <div
        onDragEnter={handleDrag}
        onDragOver={handleDrag}
        onDragLeave={handleDrag}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        style={{
          border: `2px dashed ${dragActive ? 'var(--primary-cyan)' : 'var(--border-glass-accent)'}`,
          borderRadius: 'var(--radius-md)',
          padding: '20px',
          textAlign: 'center',
          cursor: 'pointer',
          background: dragActive ? 'rgba(14, 165, 233, 0.1)' : 'rgba(15, 23, 42, 0.6)',
          transition: 'var(--transition-smooth)',
          minHeight: '220px',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          position: 'relative'
        }}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*,.tif,.tiff"
          style={{ display: 'none' }}
          onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
        />

        {previewUrl ? (
          <div style={{ width: '100%', height: '100%', position: 'relative', textAlign: 'center' }}>
            <img
              src={previewUrl}
              alt="Satellite Preview"
              style={{
                maxHeight: '170px',
                maxWidth: '100%',
                borderRadius: 'var(--radius-sm)',
                objectFit: 'contain',
                boxShadow: '0 8px 24px rgba(0,0,0,0.6)',
                border: '1px solid var(--border-glass)'
              }}
            />
            {imgDimensions && (
              <div style={{ marginTop: '10px', display: 'flex', justifyContent: 'center', gap: '8px', flexWrap: 'wrap' }}>
                <span className="badge badge-cyan">{imgDimensions.width} x {imgDimensions.height} px</span>
                <span className="badge badge-emerald">{(selectedFile.size / 1024).toFixed(1)} KB</span>
              </div>
            )}
          </div>
        ) : (
          <>
            <div style={{ background: 'rgba(14, 165, 233, 0.15)', padding: '16px', borderRadius: '50%', marginBottom: '12px' }}>
              <ImageIcon size={32} color="var(--primary-cyan)" className="animate-pulse-glow" />
            </div>
            <p style={{ fontWeight: '600', marginBottom: '4px', color: '#ffffff' }}>Drag & Drop Satellite Image Here</p>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Supports GeoTIFF, PNG, JPEG, Sentinel-1/2 formats</p>
          </>
        )}
      </div>

      {/* Preset Quick Load Samples */}
      <div style={{ marginTop: '16px' }}>
        <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: '8px' }}>Or load sample satellite imagery:</p>
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
          <button onClick={() => loadSampleImage('Urban Compound', '#1e293b', 'urban')} className="chip">
            <Layers size={13} color="#38bdf8" /> Urban Compound
          </button>
          <button onClick={() => loadSampleImage('Coastal Harbor', '#0f172a', 'harbor')} className="chip">
            <Layers size={13} color="#34d399" /> Harbor Ships
          </button>
          <button onClick={() => loadSampleImage('Sentinel-1 SAR Radar', '#312e81', 'sar')} className="chip">
            <Layers size={13} color="#c084fc" /> Sentinel SAR (USP-1)
          </button>
        </div>
      </div>
    </div>
  );
}
