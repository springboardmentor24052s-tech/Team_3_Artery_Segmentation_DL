import { useState } from 'react'
import {
  Activity, UploadCloud, Sparkles,
  Info, Trash2, FileImage, ShieldAlert, Cpu
} from 'lucide-react'
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, Legend, CartesianGrid
} from 'recharts'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'

export default function Stenosis() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [loading, setLoading] = useState(false)
  const [baselineRes, setBaselineRes] = useState(null)
  const [attentionRes, setAttentionRes] = useState(null)
  const [error, setError] = useState('')

  const handleFileChange = (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    setSelectedFile(file)
    setPreviewUrl(URL.createObjectURL(file))
    setBaselineRes(null)
    setAttentionRes(null)
    setError('')
  }

  const clearScan = () => {
    setSelectedFile(null)
    setPreviewUrl(null)
    setBaselineRes(null)
    setAttentionRes(null)
    setError('')
  }

  const runDualStenosisAnalysis = async () => {
    if (!selectedFile) return
    setLoading(true)
    setError('')

    try {
      const makeRequest = async (modelType) => {
        const formData = new FormData()
        formData.append('file', selectedFile)
        formData.append('model_type', modelType)

        const res = await fetch(`${API_BASE_URL}/analyze-scan`, {
          method: 'POST',
          body: formData
        })
        const data = await res.json()
        if (!res.ok) throw new Error(data.detail || `Failed on ${modelType}`)
        return data
      }

      const [baseData, attData] = await Promise.all([
        makeRequest('baseline'),
        makeRequest('attention')
      ])

      setBaselineRes(baseData)
      setAttentionRes(attData)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const sevColors = {
    Normal: '#10b981',
    Mild: '#38bdf8',
    Moderate: '#f59e0b',
    Severe: '#ef4444',
    Critical: '#dc2626'
  }

  const combinedDiameterData = attentionRes?.stenosis?.diameter_profile?.map((pt, i) => ({
    idx: pt.idx,
    AttentionUNet: pt.diameter,
    BaselineUNet: baselineRes?.stenosis?.diameter_profile?.[i]?.diameter ?? null
  })) || []

  const renderModelPanel = (title, badge, data) => {
    const color = sevColors[data.stenosis.severity] || '#38bdf8'

    return (
      <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: '16px', padding: '18px', display: 'flex', flexDirection: 'column', gap: '14px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Cpu size={17} color="#38bdf8" />
            <h3 style={{ margin: 0, fontSize: '16px' }}>{title}</h3>
          </div>
          <span style={{ fontSize: '11px', fontFamily: 'JetBrains Mono, monospace', color: '#38bdf8', background: 'rgba(56,189,248,0.12)', padding: '3px 9px', borderRadius: '999px' }}>
            {badge} · {data.inference_time_ms} ms
          </span>
        </div>

        {/* SEVERITY BANNER */}
        <div style={{ background: '#0b1120', border: `1px solid ${color}`, borderRadius: '12px', padding: '14px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <span style={{ fontSize: '11px', color: '#94a3b8', display: 'block' }}>ESTIMATED NARROWING</span>
            <strong style={{ fontSize: '28px', fontFamily: 'JetBrains Mono, monospace', color }}>
              {data.stenosis.narrowing_percentage}%
            </strong>
          </div>

          <div style={{ textAlign: 'right' }}>
            <span style={{ display: 'inline-block', padding: '4px 12px', borderRadius: '999px', background: `${color}22`, color, fontWeight: 700, fontSize: '12px' }}>
              {data.stenosis.severity.toUpperCase()}
            </span>
            <span style={{ display: 'block', fontSize: '11px', color: '#94a3b8', marginTop: '5px', fontFamily: 'JetBrains Mono, monospace' }}>
              Conf: {data.stenosis.confidence_score}%
            </span>
          </div>
        </div>

        {/* SIDE-BY-SIDE VISUALS: LOCALIZATION & GRAD-CAM++ */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
          <div style={{ background: '#050912', border: '1px solid #1e293b', borderRadius: '10px', padding: '8px', textAlign: 'center' }}>
            <span style={{ fontSize: '11px', color: '#94a3b8', display: 'block', marginBottom: '6px', fontWeight: 600 }}>
              Stenosis Localization
            </span>
            <img
              src={data.images.stenosis_localized}
              alt="Stenosis Box"
              style={{ width: '100%', maxHeight: '210px', objectFit: 'contain', borderRadius: '6px' }}
            />
          </div>

          <div style={{ background: '#050912', border: '1px solid #1e293b', borderRadius: '10px', padding: '8px', textAlign: 'center' }}>
            <span style={{ fontSize: '11px', color: '#94a3b8', display: 'block', marginBottom: '6px', fontWeight: 600 }}>
              Grad-CAM++ XAI Heatmap
            </span>
            <img
              src={data.images.gradcam}
              alt="Grad-CAM++ XAI"
              style={{ width: '100%', maxHeight: '210px', objectFit: 'contain', borderRadius: '6px' }}
            />
          </div>
        </div>

        {/* DIAMETER & MORPHOLOGY METRICS */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '8px', textAlign: 'center' }}>
          <div style={{ background: '#0b1120', padding: '8px 4px', borderRadius: '8px' }}>
            <span style={{ fontSize: '10px', color: '#94a3b8', display: 'block' }}>Ref D_ref</span>
            <strong style={{ fontSize: '13px', fontFamily: 'JetBrains Mono, monospace', color: '#38bdf8' }}>
              {data.stenosis.reference_diameter_px}px
            </strong>
          </div>
          <div style={{ background: '#0b1120', padding: '8px 4px', borderRadius: '8px' }}>
            <span style={{ fontSize: '10px', color: '#94a3b8', display: 'block' }}>Min D_sten</span>
            <strong style={{ fontSize: '13px', fontFamily: 'JetBrains Mono, monospace', color: '#f43f5e' }}>
              {data.stenosis.stenosis_diameter_px}px
            </strong>
          </div>
          <div style={{ background: '#0b1120', padding: '8px 4px', borderRadius: '8px' }}>
            <span style={{ fontSize: '10px', color: '#94a3b8', display: 'block' }}>Branches</span>
            <strong style={{ fontSize: '13px', fontFamily: 'JetBrains Mono, monospace', color: '#10b981' }}>
              {data.vessel_structure.branching_points}
            </strong>
          </div>
          <div style={{ background: '#0b1120', padding: '8px 4px', borderRadius: '8px' }}>
            <span style={{ fontSize: '10px', color: '#94a3b8', display: 'block' }}>Tortuosity</span>
            <strong style={{ fontSize: '13px', fontFamily: 'JetBrains Mono, monospace', color: '#f59e0b' }}>
              {data.vessel_structure.tortuosity_index}
            </strong>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div style={{ color: '#f8fafc' }}>
      {/* HEADER */}
      <div className="page-heading">
        <div>
          <span className="eyebrow">MILESTONE 3 · STENOSIS & EXPLAINABLE AI</span>
          <h1>Stenosis Detection & Dual-Model Comparison</h1>
          <p>
            Upload any raw coronary angiogram directly—automatic segmentation, narrowing quantification, and Grad-CAM++ localization run across both architectures.
          </p>
        </div>
      </div>

      {error && (
        <div style={{ padding: '12px 16px', background: '#450a0a', border: '1px solid #dc2626', borderRadius: '10px', color: '#fca5a5', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px', fontSize: '13px' }}>
          <Info size={16} /> {error}
        </div>
      )}

      {/* UPLOAD & TRIGGER BAR */}
      <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: '16px', padding: '18px', marginBottom: '20px', display: 'flex', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between', gap: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px', flex: 1, minWidth: '260px' }}>
          {!previewUrl ? (
            <label style={{ display: 'flex', alignItems: 'center', gap: '12px', padding: '12px 18px', border: '1.5px dashed #334155', borderRadius: '12px', background: '#0b1120', cursor: 'pointer', flex: 1 }}>
              <UploadCloud size={24} color="#38bdf8" />
              <div>
                <strong style={{ display: 'block', fontSize: '13.5px' }}>Select Raw Angiogram (ARCADE / Clinical PNG)</strong>
                <span style={{ fontSize: '11.5px', color: '#64748b' }}>No pre-segmented mask needed—full pipeline runs automatically</span>
              </div>
              <input type="file" accept="image/*" onChange={handleFileChange} hidden />
            </label>
          ) : (
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', background: '#0b1120', border: '1px solid #1e293b', padding: '8px 14px', borderRadius: '12px' }}>
              <img src={previewUrl} alt="Preview" style={{ width: '46px', height: '46px', objectFit: 'cover', borderRadius: '8px' }} />
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13px', fontWeight: 600 }}>
                  <FileImage size={15} color="#38bdf8" />
                  <span>{selectedFile?.name}</span>
                </div>
                <span style={{ fontSize: '11px', color: '#94a3b8' }}>Ready for Baseline vs Attention U-Net stenosis comparison</span>
              </div>
              <button type="button" onClick={clearScan} style={{ background: 'none', border: 'none', color: '#f87171', cursor: 'pointer', marginLeft: '8px' }}>
                <Trash2 size={16} />
              </button>
            </div>
          )}
        </div>

        <button
          type="button"
          disabled={!selectedFile || loading}
          onClick={runDualStenosisAnalysis}
          className="primary-button"
          style={{ opacity: !selectedFile || loading ? 0.6 : 1, padding: '12px 22px' }}
        >
          <Sparkles size={16} />
          {loading ? 'Analyzing Both Models...' : 'Detect Stenosis & Run Grad-CAM++'}
        </button>
      </div>

      {/* RESULTS SECTION */}
      {!attentionRes ? (
        <div style={{ background: '#0f172a', border: '1px dashed #1e293b', borderRadius: '16px', padding: '54px 20px', textAlign: 'center', color: '#64748b' }}>
          <Activity size={46} style={{ opacity: 0.4, marginBottom: '10px' }} />
          <p style={{ margin: 0, color: '#94a3b8', fontSize: '15px', fontWeight: 600 }}>
            Upload an Angiogram Above to Compare Stenosis Detection
          </p>
          <span style={{ fontSize: '12.5px', display: 'block', marginTop: '4px' }}>
            Automatically computes D_ref, D_stenosis, narrowing %, severity grade, and Grad-CAM++ XAI maps.
          </span>
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          {/* DUAL MODEL CARDS */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(420px, 1fr))', gap: '18px' }}>
            {renderModelPanel('Model v1.0 — Baseline U-Net', 'Standard Encoder-Decoder', baselineRes)}
            {renderModelPanel('Model v2.0 — Attention U-Net (scSE)', 'Spatial + Channel Attention', attentionRes)}
          </div>

          {/* COMPARATIVE DIAMETER CURVE */}
          <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: '16px', padding: '20px' }}>
            <div style={{ marginBottom: '12px' }}>
              <span className="eyebrow">CENTERLINE LUMEN CALIBER COMPARISON</span>
              <h3 style={{ margin: '4px 0 0', fontSize: '16px' }}>Vessel Diameter Profile Along Skeleton (Baseline vs Attention U-Net)</h3>
            </div>

            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={combinedDiameterData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="idx" stroke="#64748b" fontSize={11} />
                <YAxis stroke="#64748b" fontSize={11} unit="px" />
                <Tooltip contentStyle={{ background: '#0b1120', border: '1px solid #334155', fontSize: '12px' }} />
                <Legend />
                <Line type="monotone" dataKey="AttentionUNet" name="Attention U-Net (v2.0)" stroke="#38bdf8" strokeWidth={2.5} dot={false} />
                <Line type="monotone" dataKey="BaselineUNet" name="Baseline U-Net (v1.0)" stroke="#f59e0b" strokeWidth={2} strokeDasharray="4 4" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* CLINICAL DISCLAIMER */}
      <div style={{ marginTop: '18px', padding: '12px 16px', background: 'rgba(245, 158, 11, 0.08)', border: '1px solid rgba(245, 158, 11, 0.25)', borderRadius: '10px', fontSize: '12px', color: '#fcd34d', display: 'flex', alignItems: 'center', gap: '8px' }}>
        <ShieldAlert size={16} style={{ flexShrink: 0 }} />
        <span>
          AI-assisted geometric lumen measurements for clinical decision support and research only; final stenosis grading requires cardiologist verification.
        </span>
      </div>
    </div>
  )
}