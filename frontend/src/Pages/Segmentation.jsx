import { useState } from 'react'
import {
  Activity, UploadCloud, Sparkles, Layers,
  CheckCircle2, Info, Trash2, FileImage
} from 'lucide-react'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'

export default function Segmentation() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [modelType, setModelType] = useState('attention')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')

  const handleFileChange = (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    setSelectedFile(file)
    setPreviewUrl(URL.createObjectURL(file))
    setResult(null)
    setError('')
  }

  const clearScan = () => {
    setSelectedFile(null)
    setPreviewUrl(null)
    setResult(null)
    setError('')
  }

  const runSegmentation = async (overrideModel = modelType) => {
    if (!selectedFile) return
    setLoading(true)
    setError('')

    try {
      const formData = new FormData()
      formData.append('file', selectedFile)
      formData.append('model_type', overrideModel)

      const res = await fetch(`${API_BASE_URL}/analyze-scan`, {
        method: 'POST',
        body: formData
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Segmentation failed')

      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ color: '#f8fafc' }}>
      {/* HEADER */}
      <div className="page-heading">
        <div>
          <span className="eyebrow">MILESTONE 2 · VESSEL EXTRACTION</span>
          <h1>Artery Segmentation Workspace</h1>
          <p>
            Direct coronary angiogram upload for CLAHE enhancement, binary lumen mask extraction, and centerline skeletonization.
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <select
            value={modelType}
            onChange={(e) => {
              setModelType(e.target.value)
              if (selectedFile) runSegmentation(e.target.value)
            }}
            style={{
              background: '#0f172a',
              color: '#38bdf8',
              border: '1px solid #334155',
              borderRadius: '10px',
              padding: '9px 14px',
              fontSize: '13px',
              fontWeight: 600
            }}
          >
            <option value="attention">Attention U-Net (v2.0 — scSE)</option>
            <option value="baseline">Baseline U-Net (v1.0)</option>
          </select>
        </div>
      </div>

      {error && (
        <div style={{ padding: '12px 16px', background: '#450a0a', border: '1px solid #dc2626', borderRadius: '10px', color: '#fca5a5', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px', fontSize: '13px' }}>
          <Info size={16} /> {error}
        </div>
      )}

      {/* MAIN WORKSPACE GRID */}
      <div style={{ display: 'grid', gridTemplateColumns: '320px 1fr', gap: '20px', alignItems: 'start' }}>
        {/* LEFT: UPLOAD & CONTROLS */}
        <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: '16px', padding: '18px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <span className="eyebrow">INPUT ANGIOGRAM</span>

          {!previewUrl ? (
            <label style={{ border: '1.5px dashed #334155', borderRadius: '12px', padding: '28px 16px', textAlign: 'center', cursor: 'pointer', background: '#0b1120', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '8px' }}>
              <UploadCloud size={30} color="#38bdf8" />
              <strong style={{ fontSize: '14px' }}>Upload Coronary Scan</strong>
              <span style={{ fontSize: '12px', color: '#64748b' }}>PNG / JPG from ARCADE or clinical PACS</span>
              <input type="file" accept="image/*" onChange={handleFileChange} hidden />
            </label>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              <div style={{ background: '#050912', border: '1px solid #1e293b', borderRadius: '12px', padding: '10px', textAlign: 'center' }}>
                <img src={previewUrl} alt="Uploaded Scan" style={{ width: '100%', maxHeight: '220px', objectFit: 'contain', borderRadius: '8px' }} />
              </div>

              <div style={{ background: '#0b1120', border: '1px solid #1e293b', borderRadius: '10px', padding: '10px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', overflow: 'hidden' }}>
                  <FileImage size={16} color="#38bdf8" />
                  <span style={{ fontSize: '12px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {selectedFile?.name}
                  </span>
                </div>
                <button type="button" onClick={clearScan} style={{ background: 'none', border: 'none', color: '#f87171', cursor: 'pointer' }}>
                  <Trash2 size={15} />
                </button>
              </div>
            </div>
          )}

          <button
            type="button"
            disabled={!selectedFile || loading}
            onClick={() => runSegmentation()}
            className="primary-button"
            style={{ width: '100%', justifyContent: 'center', opacity: !selectedFile || loading ? 0.6 : 1 }}
          >
            <Sparkles size={16} />
            {loading ? 'Extracting Vessel Mask...' : 'Run Artery Segmentation'}
          </button>

          {result && (
            <div style={{ background: '#0b1120', border: '1px solid #1e293b', borderRadius: '12px', padding: '14px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: '#34d399', fontSize: '12px', fontWeight: 700 }}>
                <CheckCircle2 size={15} /> Segmented in {result.inference_time_ms} ms
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', fontSize: '12px' }}>
                <div style={{ background: '#0f172a', padding: '8px', borderRadius: '8px' }}>
                  <span style={{ color: '#94a3b8', display: 'block', fontSize: '11px' }}>Centerline Length</span>
                  <strong style={{ fontFamily: 'JetBrains Mono, monospace' }}>{result.vessel_structure.vessel_length_px} px</strong>
                </div>
                <div style={{ background: '#0f172a', padding: '8px', borderRadius: '8px' }}>
                  <span style={{ color: '#94a3b8', display: 'block', fontSize: '11px' }}>Bifurcations</span>
                  <strong style={{ fontFamily: 'JetBrains Mono, monospace', color: '#10b981' }}>{result.vessel_structure.branching_points}</strong>
                </div>
                <div style={{ background: '#0f172a', padding: '8px', borderRadius: '8px' }}>
                  <span style={{ color: '#94a3b8', display: 'block', fontSize: '11px' }}>Tortuosity Index</span>
                  <strong style={{ fontFamily: 'JetBrains Mono, monospace', color: '#f59e0b' }}>{result.vessel_structure.tortuosity_index}</strong>
                </div>
                <div style={{ background: '#0f172a', padding: '8px', borderRadius: '8px' }}>
                  <span style={{ color: '#94a3b8', display: 'block', fontSize: '11px' }}>Reference Caliber</span>
                  <strong style={{ fontFamily: 'JetBrains Mono, monospace', color: '#38bdf8' }}>{result.stenosis.reference_diameter_px} px</strong>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* RIGHT: 4-PANEL SEGMENTATION OUTPUT */}
        <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: '16px', padding: '20px', minHeight: '480px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Layers size={18} color="#38bdf8" />
              <h2 style={{ margin: 0, fontSize: '17px' }}>Segmentation & Skeletonization Outputs</h2>
            </div>
            {result && (
              <span style={{ fontSize: '12px', color: '#38bdf8', fontFamily: 'JetBrains Mono, monospace' }}>
                {result.model_used}
              </span>
            )}
          </div>

          {!result ? (
            <div style={{ height: '390px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', background: '#050912', borderRadius: '12px', border: '1px dashed #1e293b', color: '#64748b' }}>
              <Activity size={44} style={{ opacity: 0.4, marginBottom: '10px' }} />
              <p style={{ margin: 0, color: '#94a3b8', fontSize: '14px' }}>No Segmentation Output Yet</p>
              <span style={{ fontSize: '12px', marginTop: '4px' }}>Upload an angiogram on the left and click "Run Artery Segmentation"</span>
            </div>
          ) : (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '16px' }}>
              {[
                { title: '1. Original Input Scan', img: result.images.original, tag: '512×512 Raw' },
                { title: '2. CLAHE + Bilateral Denoised', img: result.images.preprocessed, tag: 'Normalized Contrast' },
                { title: '3. Binary Artery Mask', img: result.images.segmentation_mask, tag: 'Lumen Segmentation' },
                { title: '4. Extracted Vessel Centerline', img: result.images.centerline, tag: '1px Topological Skeleton' }
              ].map((panel) => (
                <div key={panel.title} style={{ background: '#050912', border: '1px solid #1e293b', borderRadius: '12px', padding: '12px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                    <span style={{ fontSize: '12.5px', fontWeight: 600, color: '#e2e8f0' }}>{panel.title}</span>
                    <span style={{ fontSize: '10.5px', color: '#38bdf8', background: 'rgba(56,189,248,0.1)', padding: '2px 8px', borderRadius: '999px' }}>{panel.tag}</span>
                  </div>
                  <img
                    src={panel.img}
                    alt={panel.title}
                    style={{ width: '100%', maxHeight: '240px', objectFit: 'contain', borderRadius: '8px', background: '#000' }}
                  />
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}