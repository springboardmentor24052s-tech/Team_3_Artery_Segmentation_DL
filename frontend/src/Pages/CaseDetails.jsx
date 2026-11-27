import { useState, useRef, useEffect } from 'react'
import {
  ArrowLeft, FileImage, HeartPulse,
  Info, ScanLine, Trash2, UploadCloud, UserRound, AlertTriangle,
  Sparkles, SlidersHorizontal, FileCheck2, Play, Download,
  Grid, BellRing, Cpu, GitBranch, CheckCircle2, Stethoscope
} from 'lucide-react'
import {
  ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip, ReferenceLine
} from 'recharts'

import ModelComparison from '../Components/ModelComparison'
import PerformanceReport from '../Components/PerformanceReport'
import { saveCaseAnalysis, getCaseAnalysis, deleteCaseAnalysis } from '../utils/scanStorage'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'

const createDemoAngiogram = (presetMode = 'moderate_lcx') => {
  const canvas = document.createElement('canvas')
  canvas.width = 512
  canvas.height = 512
  const ctx = canvas.getContext('2d')

  const grad = ctx.createRadialGradient(256, 256, 30, 256, 256, 370)
  grad.addColorStop(0, '#475569')
  grad.addColorStop(1, '#0f172a')
  ctx.fillStyle = grad
  ctx.fillRect(0, 0, 512, 512)

  ctx.lineCap = 'round'
  ctx.strokeStyle = '#090d16'

  const configs = {
    critical_lad: { start: [90, 60], mid1: [210, 195], mid2: [245, 265], end: [390, 450], refW: 22, stenW: 2.5, branchAngle: [370, 120] },
    severe_rca:   { start: [140, 55], mid1: [260, 190], mid2: [275, 270], end: [210, 455], refW: 20, stenW: 5,   branchAngle: [395, 290] },
    moderate_lcx: { start: [105, 80], mid1: [195, 210], mid2: [235, 285], end: [365, 440], refW: 19, stenW: 8.5, branchAngle: [120, 340] },
    mild_diag:    { start: [115, 70], mid1: [220, 200], mid2: [255, 275], end: [380, 435], refW: 18, stenW: 11.5,branchAngle: [360, 150] },
    normal_clean: { start: [100, 65], mid1: [215, 205], mid2: [255, 280], end: [390, 445], refW: 17, stenW: 15.5,branchAngle: [375, 135] },
    severe_lmca:  { start: [85, 90],  mid1: [185, 180], mid2: [220, 240], end: [375, 430], refW: 23, stenW: 4.5, branchAngle: [390, 160] },
    moderate_rca: { start: [155, 50], mid1: [275, 195], mid2: [285, 275], end: [195, 455], refW: 19, stenW: 7.5, branchAngle: [400, 310] },
    mild_lad:     { start: [95, 65],  mid1: [210, 200], mid2: [248, 275], end: [385, 445], refW: 18, stenW: 11,  branchAngle: [365, 125] },
    critical_lcx: { start: [110, 75], mid1: [200, 205], mid2: [238, 280], end: [360, 445], refW: 22, stenW: 2,   branchAngle: [115, 355] },
    normal_rca:   { start: [150, 55], mid1: [270, 195], mid2: [280, 275], end: [205, 450], refW: 17, stenW: 15,  branchAngle: [390, 300] },
  }

  const c = configs[presetMode] || configs.moderate_lcx

  ctx.lineWidth = c.refW
  ctx.beginPath()
  ctx.moveTo(c.start[0], c.start[1])
  ctx.quadraticCurveTo((c.start[0] + c.mid1[0]) / 2 + 15, (c.start[1] + c.mid1[1]) / 2 - 10, c.mid1[0], c.mid1[1])
  ctx.stroke()

  ctx.lineWidth = c.stenW
  ctx.beginPath()
  ctx.moveTo(c.mid1[0], c.mid1[1])
  ctx.quadraticCurveTo((c.mid1[0] + c.mid2[0]) / 2, (c.mid1[1] + c.mid2[1]) / 2, c.mid2[0], c.mid2[1])
  ctx.stroke()

  ctx.lineWidth = c.refW - 2
  ctx.beginPath()
  ctx.moveTo(c.mid2[0], c.mid2[1])
  ctx.quadraticCurveTo((c.mid2[0] + c.end[0]) / 2 - 10, (c.mid2[1] + c.end[1]) / 2 + 15, c.end[0], c.end[1])
  ctx.stroke()

  ctx.lineWidth = Math.max(6, c.refW * 0.5)
  ctx.beginPath()
  ctx.moveTo(c.mid1[0] - 15, c.mid1[1] - 25)
  ctx.quadraticCurveTo(280, 165, c.branchAngle[0], c.branchAngle[1])
  ctx.stroke()

  ctx.lineWidth = Math.max(5, c.refW * 0.45)
  ctx.beginPath()
  ctx.moveTo(c.mid2[0] + 15, c.mid2[1] + 35)
  ctx.quadraticCurveTo(205, 385, 135, 445)
  ctx.stroke()

  return canvas.toDataURL('image/png')
}

export default function CaseDetails({ caseData, onBack }) {
  const caseKey = caseData?.caseId || caseData?.patientId || 'DEFAULT'

  const [activeTab, setActiveTab] = useState('studio')
  const [modelType, setModelType] = useState('attention')
  const [fileName, setFileName] = useState('')
  const [selectedFile, setSelectedFile] = useState(null)
  const [localPreview, setLocalPreview] = useState(null)
  const [analysis, setAnalysis] = useState(null)
  const [pipelineStep, setPipelineStep] = useState(0)
  const [savedBadge, setSavedBadge] = useState(false)
  const [error, setError] = useState('')

  const [viewMode, setViewMode] = useState('split')
  const [splitPos, setSplitPos] = useState(50)
  const [rightLayer, setRightLayer] = useState('stenosis_localized')
  const [showReportModal, setShowReportModal] = useState(false)
  const viewportRef = useRef(null)

  // Load previously saved scan & analysis from IndexedDB
  useEffect(() => {
    let active = true
    getCaseAnalysis(caseKey).then((record) => {
      if (!active || !record?.analysis) return
      setAnalysis(record.analysis)
      setModelType(record.modelType || 'attention')
      setFileName(record.fileName || 'saved_scan.png')
      setLocalPreview(record.analysis.images?.original || null)
      setPipelineStep(5)
      setSavedBadge(true)
    })
    return () => { active = false }
  }, [caseKey])

  const persistCaseAnalysis = async (resultData, usedModel, scanName) => {
    try {
      await saveCaseAnalysis(caseKey, {
        analysis: resultData,
        modelType: usedModel,
        fileName: scanName,
        updatedAt: new Date().toISOString()
      })

      const casesList = JSON.parse(localStorage.getItem('cases') || '[]')
      if (Array.isArray(casesList)) {
        const derivedRisk = resultData?.clinical_profile?.inferred_risk_profile || 'Scan Analyzed'
        const derivedHistory = resultData?.clinical_profile?.morphological_finding || 'Completed'
        const updatedCases = casesList.map(c =>
          c.caseId === caseKey
            ? {
                ...c,
                status: 'Completed',
                riskFactors: derivedRisk,
                medicalHistory: derivedHistory
              }
            : c
        )
        localStorage.setItem('cases', JSON.stringify(updatedCases))
      }
      setSavedBadge(true)
    } catch (e) {
      console.error('IndexedDB save failed:', e)
    }
  }

  const clearSavedCaseData = async () => {
    await deleteCaseAnalysis(caseKey)
    try {
      const casesList = JSON.parse(localStorage.getItem('cases') || '[]')
      if (Array.isArray(casesList)) {
        const resetCases = casesList.map(c =>
          c.caseId === caseKey
            ? {
                ...c,
                status: 'Pending',
                riskFactors: 'Awaiting AI Scan Assessment',
                medicalHistory: 'Pending angiogram analysis'
              }
            : c
        )
        localStorage.setItem('cases', JSON.stringify(resetCases))
      }
    } catch {
      // ignore
    }
    setSelectedFile(null)
    setFileName('')
    setLocalPreview(null)
    setAnalysis(null)
    setPipelineStep(0)
    setSavedBadge(false)
  }

  const stages = [
    '1. Bilateral Denoising & CLAHE Contrast Normalization',
    '2. Deep Learning Artery Segmentation Inference',
    '3. Skeletonization, Branching & Tortuosity Extraction',
    '4. Stenosis Localization & Grad-CAM++ Saliency'
  ]

  const runPipeline = async (fileToProcess = selectedFile, selectedModel = modelType, explicitName = fileName) => {
    if (!fileToProcess) return
    setError('')
    setSavedBadge(false)
    setPipelineStep(1)

    const t1 = setTimeout(() => setPipelineStep(2), 350)
    const t2 = setTimeout(() => setPipelineStep(3), 700)
    const t3 = setTimeout(() => setPipelineStep(4), 1050)

    try {
      const formData = new FormData()
      formData.append('file', fileToProcess)
      formData.append('model_type', selectedModel)

      const res = await fetch(`${API_BASE_URL}/analyze-scan`, {
        method: 'POST',
        body: formData
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Analysis failed')

      setTimeout(() => {
        setAnalysis(data)
        setLocalPreview(data.images.original)
        setPipelineStep(5)
        persistCaseAnalysis(data, selectedModel, explicitName || fileToProcess.name)
      }, 1250)
    } catch (err) {
      clearTimeout(t1); clearTimeout(t2); clearTimeout(t3)
      setPipelineStep(0)
      setError(err.message)
    }
  }

  const handlePreset = async (mode, name) => {
    const url = createDemoAngiogram(mode)
    const res = await fetch(url)
    const blob = await res.blob()
    const file = new File([blob], `${name}.png`, { type: 'image/png' })
    setSelectedFile(file)
    setFileName(`${name}.png`)
    setLocalPreview(url)
    setAnalysis(null)
    runPipeline(file, modelType, `${name}.png`)
  }

  const handleFileChange = (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    setSelectedFile(file)
    setFileName(file.name)
    setLocalPreview(URL.createObjectURL(file))
    setAnalysis(null)
    setPipelineStep(0)
    setSavedBadge(false)
  }

  const handleModelSwitch = async (newModel) => {
    setModelType(newModel)
    if (selectedFile) {
      runPipeline(selectedFile, newModel, fileName)
    } else if (analysis?.images?.original) {
      const res = await fetch(analysis.images.original)
      const blob = await res.blob()
      const file = new File([blob], fileName || 'saved_scan.png', { type: 'image/png' })
      setSelectedFile(file)
      runPipeline(file, newModel, fileName || 'saved_scan.png')
    }
  }

  const downloadFile = (content, filename, type) => {
    const blob = new Blob([content], { type })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    a.click()
    URL.revokeObjectURL(url)
  }

  const exportJSON = () => {
    if (!analysis) return
    const payload = {
      case_id: caseData?.caseId,
      patient_id: caseData?.patientId,
      scan_file: fileName,
      model_version: analysis.model_used,
      inference_time_ms: analysis.inference_time_ms,
      vessel_structure: analysis.vessel_structure,
      stenosis_findings: analysis.stenosis,
      ai_clinical_profile: analysis.clinical_profile,
      disclaimer: 'AI-assisted decision support only; not an autonomous clinical diagnosis.'
    }
    downloadFile(JSON.stringify(payload, null, 2), `${caseKey}_report.json`, 'application/json')
  }

  const exportCSV = () => {
    if (!analysis) return
    const rows = [
      ['Case ID', 'Patient ID', 'Scan File', 'Model', 'Narrowing %', 'Severity', 'Inferred Risk Profile', 'Ref Diameter (px)', 'Min Diameter (px)', 'Vessel Length (px)', 'Branch Points', 'Tortuosity'],
      [
        caseData?.caseId,
        caseData?.patientId,
        fileName,
        analysis.model_used,
        analysis.stenosis.narrowing_percentage,
        analysis.stenosis.severity,
        `"${analysis.clinical_profile?.inferred_risk_profile || ''}"`,
        analysis.stenosis.reference_diameter_px,
        analysis.stenosis.stenosis_diameter_px,
        analysis.vessel_structure.vessel_length_px,
        analysis.vessel_structure.branching_points,
        analysis.vessel_structure.tortuosity_index
      ]
    ]
    const csv = rows.map(r => r.join(',')).join('\n')
    downloadFile(csv, `${caseKey}_report.csv`, 'text/csv')
  }

  const handleSplitDrag = (e) => {
    if (!viewportRef.current) return
    const rect = viewportRef.current.getBoundingClientRect()
    const clientX = e.touches ? e.touches[0].clientX : e.clientX
    setSplitPos(Math.max(5, Math.min(95, ((clientX - rect.left) / rect.width) * 100)))
  }

  const sevColors = {
    Normal: '#10b981',
    Mild: '#38bdf8',
    Moderate: '#f59e0b',
    Severe: '#ef4444',
    Critical: '#dc2626'
  }
  const activeColor = analysis ? (sevColors[analysis.stenosis.severity] || '#38bdf8') : '#38bdf8'

  return (
    <div style={{ color: '#f8fafc' }}>
      {analysis?.alert?.triggered && (
        <div style={{ background: 'linear-gradient(90deg, #7f1d1d, #991b1b)', border: '1px solid #ef4444', borderRadius: '12px', padding: '12px 18px', marginBottom: '16px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '12px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <BellRing size={20} color="#fca5a5" />
            <div>
              <strong style={{ fontSize: '13px' }}>CRITICAL CLINICAL ALERT · {analysis.alert.level.toUpperCase()} PRIORITY</strong>
              <p style={{ margin: 0, fontSize: '12px', color: '#fecaca' }}>{analysis.alert.message} Immediate cardiologist review recommended.</p>
            </div>
          </div>
          <span style={{ fontSize: '11px', fontFamily: 'JetBrains Mono, monospace', background: '#450a0a', padding: '4px 10px', borderRadius: '999px' }}>
            CONFIDENCE: {analysis.stenosis.confidence_score}%
          </span>
        </div>
      )}

      {/* HEADER */}
      <div className="details-page-header">
        <div>
          <button type="button" className="details-back-button" onClick={onBack} style={{ marginBottom: '8px' }}>
            <ArrowLeft size={15} /> Back to Patient Cases
          </button>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
            <h1 style={{ margin: 0, fontSize: '23px' }}>VascularAI Clinical Workstation</h1>
            <span style={{ fontSize: '12px', padding: '3px 10px', borderRadius: '999px', background: '#0f172a', border: '1px solid #1e293b', color: '#38bdf8', fontFamily: 'JetBrains Mono, monospace' }}>
              {caseData?.caseId} · {caseData?.patientId}
            </span>
            {savedBadge && (
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: '5px', fontSize: '11px', padding: '3px 10px', borderRadius: '999px', background: 'rgba(16,185,129,0.14)', border: '1px solid rgba(16,185,129,0.35)', color: '#34d399', fontWeight: 600 }}>
                <CheckCircle2 size={12} /> Scan & Clinical Profile Saved
              </span>
            )}
          </div>
        </div>

        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
          {[
            { id: 'studio', label: '1. AI Analysis & XAI Studio' },
            { id: 'benchmarks', label: '2. Model Evaluation & Metrics' }
          ].map(t => (
            <button
              key={t.id}
              type="button"
              onClick={() => setActiveTab(t.id)}
              style={{
                padding: '8px 16px', borderRadius: '10px', border: '1px solid #1e293b', cursor: 'pointer', fontSize: '13px', fontWeight: 600,
                background: activeTab === t.id ? '#0284c7' : '#0f172a', color: activeTab === t.id ? '#fff' : '#94a3b8'
              }}
            >
              {t.label}
            </button>
          ))}
        </div>
      </div>

      {/* TELEMETRY STRIP */}
      <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '18px', padding: '12px 16px', background: '#0f172a', border: '1px solid #1e293b', borderRadius: '12px', marginBottom: '18px', fontSize: '13px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <UserRound size={15} color="#38bdf8" />
          <span>Patient:</span>
          <strong>{caseData?.age || '58'}y / {caseData?.gender || 'Male'}</strong>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <HeartPulse size={15} color={analysis ? activeColor : '#64748b'} />
          <span>AI Risk Profile:</span>
          <strong style={{ color: analysis ? activeColor : '#94a3b8' }}>
            {analysis?.clinical_profile?.inferred_risk_profile || 'Awaiting Scan Analysis'}
          </strong>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <Cpu size={15} color="#10b981" />
          <span>Architecture:</span>
          <select
            value={modelType}
            onChange={(e) => handleModelSwitch(e.target.value)}
            style={{ background: '#1e293b', color: '#38bdf8', border: '1px solid #334155', borderRadius: '6px', padding: '4px 8px', fontSize: '12px', fontWeight: 600 }}
          >
            <option value="attention">Model v2.0 — Attention U-Net (scSE)</option>
            <option value="baseline">Model v1.0 — Baseline U-Net</option>
          </select>
        </div>

        {analysis && (
          <div style={{ marginLeft: 'auto', display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
            <button type="button" onClick={exportJSON} style={{ display: 'inline-flex', alignItems: 'center', gap: '5px', background: '#1e293b', border: '1px solid #334155', color: '#e2e8f0', padding: '6px 11px', borderRadius: '8px', cursor: 'pointer', fontSize: '12px' }}>
              <Download size={13} /> JSON
            </button>
            <button type="button" onClick={exportCSV} style={{ display: 'inline-flex', alignItems: 'center', gap: '5px', background: '#1e293b', border: '1px solid #334155', color: '#e2e8f0', padding: '6px 11px', borderRadius: '8px', cursor: 'pointer', fontSize: '12px' }}>
              <Download size={13} /> CSV
            </button>
            <button type="button" onClick={() => setShowReportModal(true)} style={{ display: 'inline-flex', alignItems: 'center', gap: '5px', background: '#0284c7', border: 'none', color: '#fff', padding: '6px 12px', borderRadius: '8px', cursor: 'pointer', fontSize: '12px', fontWeight: 600 }}>
              <FileCheck2 size={14} /> Clinical Report
            </button>
          </div>
        )}
      </div>

      {error && (
        <div style={{ padding: '12px 16px', background: '#450a0a', border: '1px solid #dc2626', borderRadius: '10px', color: '#fca5a5', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px', fontSize: '13px' }}>
          <Info size={16} /> {error}
        </div>
      )}

      {activeTab === 'studio' && (
        <div style={{ display: 'grid', gridTemplateColumns: '300px 1fr 350px', gap: '18px', alignItems: 'start' }}>
          {/* LEFT PANEL */}
          <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: '16px', padding: '16px', display: 'flex', flexDirection: 'column', gap: '14px' }}>
            <span className="eyebrow">1. IMAGE ACQUISITION</span>

            {!localPreview ? (
              <label style={{ border: '1.5px dashed #334155', borderRadius: '12px', padding: '20px 12px', textAlign: 'center', cursor: 'pointer', background: '#0b1120', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '6px' }}>
                <UploadCloud size={26} color="#38bdf8" />
                <strong style={{ fontSize: '13px' }}>Upload ARCADE / Clinical Scan</strong>
                <span style={{ fontSize: '11px', color: '#64748b' }}>Angiography / CTA Slice (PNG, JPG)</span>
                <input type="file" accept="image/*" onChange={handleFileChange} hidden />
              </label>
            ) : (
              <div style={{ background: '#0b1120', border: '1px solid #1e293b', borderRadius: '10px', padding: '10px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', overflow: 'hidden' }}>
                  <FileImage size={17} color="#38bdf8" />
                  <span style={{ fontSize: '12px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{fileName || 'patient_angiogram.png'}</span>
                </div>
                <button type="button" title="Clear saved scan" onClick={clearSavedCaseData} style={{ background: 'none', border: 'none', color: '#f87171', cursor: 'pointer' }}>
                  <Trash2 size={15} />
                </button>
              </div>
            )}

            <div>
              <span style={{ fontSize: '11px', color: '#94a3b8', display: 'block', marginBottom: '6px' }}>OR SYNTHETIC FALLBACK SCAN:</span>
              <button
                type="button"
                onClick={() => handlePreset(caseData?.presetMode || 'severe_rca', `${caseData?.patientId}_Synthetic`)}
                style={{ width: '100%', padding: '9px', borderRadius: '8px', background: '#1e293b', border: '1px solid #334155', color: '#cbd5e1', fontSize: '12px', fontWeight: 600, cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px' }}
              >
                <Play size={13} color="#38bdf8" /> Run Sample Preset ({caseData?.patientId})
              </button>
            </div>

            {selectedFile && pipelineStep === 0 && (
              <button type="button" onClick={() => runPipeline()} style={{ width: '100%', padding: '10px', borderRadius: '10px', border: 'none', background: 'linear-gradient(135deg, #0284c7, #2563eb)', color: '#fff', fontWeight: 700, fontSize: '13px', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px' }}>
                <Sparkles size={15} /> Execute AI Pipeline
              </button>
            )}

            {pipelineStep > 0 && (
              <div style={{ background: '#0b1120', border: '1px solid #1e293b', borderRadius: '10px', padding: '12px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: '#38bdf8', fontWeight: 700 }}>
                  <span>PIPELINE STATUS</span>
                  {analysis && <span>{analysis.inference_time_ms} ms</span>}
                </div>
                {stages.map((st, idx) => {
                  const done = pipelineStep > idx + 1
                  const active = pipelineStep === idx + 1
                  return (
                    <div key={st} style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '11px', color: done ? '#34d399' : active ? '#fff' : '#475569' }}>
                      <span style={{ width: '7px', height: '7px', borderRadius: '50%', background: done ? '#10b981' : active ? '#38bdf8' : '#334155', flexShrink: 0 }} />
                      <span>{st}</span>
                    </div>
                  )
                })}
              </div>
            )}

            {analysis && (
              <div style={{ background: '#0b1120', border: '1px solid #1e293b', borderRadius: '12px', padding: '12px' }}>
                <span style={{ fontSize: '11px', letterSpacing: '1px', color: '#38bdf8', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '5px', marginBottom: '10px' }}>
                  <GitBranch size={13} /> VESSEL MORPHOMETRY
                </span>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '6px', textAlign: 'center' }}>
                  <div style={{ background: '#0f172a', padding: '8px 4px', borderRadius: '8px' }}>
                    <span style={{ fontSize: '10px', color: '#94a3b8', display: 'block' }}>Length</span>
                    <strong style={{ fontSize: '13px', fontFamily: 'JetBrains Mono, monospace' }}>{analysis.vessel_structure.vessel_length_px}px</strong>
                  </div>
                  <div style={{ background: '#0f172a', padding: '8px 4px', borderRadius: '8px' }}>
                    <span style={{ fontSize: '10px', color: '#94a3b8', display: 'block' }}>Branches</span>
                    <strong style={{ fontSize: '13px', fontFamily: 'JetBrains Mono, monospace', color: '#10b981' }}>{analysis.vessel_structure.branching_points}</strong>
                  </div>
                  <div style={{ background: '#0f172a', padding: '8px 4px', borderRadius: '8px' }}>
                    <span style={{ fontSize: '10px', color: '#94a3b8', display: 'block' }}>Tortuosity</span>
                    <strong style={{ fontSize: '13px', fontFamily: 'JetBrains Mono, monospace', color: '#f59e0b' }}>{analysis.vessel_structure.tortuosity_index}</strong>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* CENTER PANEL */}
          <div style={{ background: '#050912', border: '1px solid #1e293b', borderRadius: '16px', padding: '16px', minHeight: '520px', display: 'flex', flexDirection: 'column' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px', flexWrap: 'wrap', gap: '8px' }}>
              <div style={{ display: 'flex', gap: '6px' }}>
                <button type="button" onClick={() => setViewMode('split')} style={{ padding: '6px 12px', borderRadius: '8px', border: '1px solid #334155', background: viewMode === 'split' ? '#0284c7' : '#0f172a', color: '#fff', fontSize: '12px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '5px' }}>
                  <SlidersHorizontal size={13} /> Interactive Wipe
                </button>
                <button type="button" onClick={() => setViewMode('matrix')} disabled={!analysis} style={{ padding: '6px 12px', borderRadius: '8px', border: '1px solid #334155', background: viewMode === 'matrix' ? '#0284c7' : '#0f172a', color: analysis ? '#fff' : '#475569', fontSize: '12px', cursor: analysis ? 'pointer' : 'not-allowed', display: 'flex', alignItems: 'center', gap: '5px' }}>
                  <Grid size={13} /> 6-Stage Pipeline Grid
                </button>
              </div>

              {analysis && viewMode === 'split' && (
                <div style={{ display: 'flex', gap: '5px' }}>
                  {[
                    { id: 'stenosis_localized', label: 'Stenosis + Mask' },
                    { id: 'gradcam', label: 'Grad-CAM++ XAI' },
                    { id: 'centerline', label: 'Centerline' }
                  ].map(l => (
                    <button key={l.id} type="button" onClick={() => setRightLayer(l.id)} style={{ padding: '5px 10px', borderRadius: '7px', border: '1px solid #334155', background: rightLayer === l.id ? '#1e293b' : 'transparent', color: rightLayer === l.id ? '#38bdf8' : '#94a3b8', fontSize: '11.5px', cursor: 'pointer' }}>
                      {l.label}
                    </button>
                  ))}
                </div>
              )}
            </div>

            {!localPreview ? (
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: '#475569', background: '#02040a', borderRadius: '12px' }}>
                <ScanLine size={48} style={{ opacity: 0.35, marginBottom: '10px' }} />
                <p style={{ margin: 0, color: '#94a3b8', fontSize: '14px' }}>Awaiting Angiography Scan</p>
                <span style={{ fontSize: '12px', marginTop: '4px' }}>Upload an image from backend/dataset/data/stenosis/test/images</span>
              </div>
            ) : viewMode === 'matrix' && analysis ? (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '10px' }}>
                {[
                  { title: '1. Original Scan', img: analysis.images.original },
                  { title: '2. CLAHE Preprocessed', img: analysis.images.preprocessed },
                  { title: '3. Artery Segmentation', img: analysis.images.segmentation_mask },
                  { title: '4. Vessel Centerline', img: analysis.images.centerline },
                  { title: '5. Stenosis Localization', img: analysis.images.stenosis_localized },
                  { title: '6. Grad-CAM++ XAI', img: analysis.images.gradcam },
                ].map(cell => (
                  <div key={cell.title} style={{ background: '#0b1120', border: '1px solid #1e293b', borderRadius: '10px', padding: '8px', textAlign: 'center' }}>
                    <span style={{ fontSize: '11px', color: '#94a3b8', display: 'block', marginBottom: '6px', fontWeight: 600 }}>{cell.title}</span>
                    <img src={cell.img} alt={cell.title} style={{ width: '100%', aspectRatio: '1/1', objectFit: 'contain', borderRadius: '6px' }} />
                  </div>
                ))}
              </div>
            ) : (
              <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#02040a', borderRadius: '12px', padding: '12px' }}>
                <div
                  ref={viewportRef}
                  onMouseMove={(e) => e.buttons === 1 && handleSplitDrag(e)}
                  onClick={handleSplitDrag}
                  style={{ position: 'relative', width: '430px', height: '430px', cursor: analysis ? 'ew-resize' : 'default', userSelect: 'none' }}
                >
                  <img src={analysis ? analysis.images.original : localPreview} alt="Original" style={{ width: '100%', height: '100%', objectFit: 'contain', borderRadius: '10px' }} />

                  {analysis && (
                    <div style={{ position: 'absolute', inset: 0, clipPath: `inset(0 0 0 ${splitPos}%)` }}>
                      <img src={analysis.images[rightLayer]} alt="AI Output" style={{ width: '100%', height: '100%', objectFit: 'contain', borderRadius: '10px' }} />
                    </div>
                  )}

                  {analysis && (
                    <div style={{ position: 'absolute', top: 0, bottom: 0, left: `${splitPos}%`, width: '2px', background: '#38bdf8', boxShadow: '0 0 10px #38bdf8', pointerEvents: 'none' }}>
                      <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', background: '#0284c7', color: '#fff', fontSize: '10px', fontWeight: 700, padding: '4px 8px', borderRadius: '999px', whiteSpace: 'nowrap' }}>
                        ◄ RAW | AI ►
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* RIGHT PANEL */}
          <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: '16px', padding: '16px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
            <span className="eyebrow">3. STENOSIS & CLINICAL ASSESSMENT</span>

            {!analysis ? (
              <div style={{ padding: '30px 14px', textAlign: 'center', color: '#64748b', fontSize: '13px', background: '#0b1120', borderRadius: '12px', border: '1px solid #1e293b' }}>
                Upload & analyze a scan to extract narrowing %, severity classification, and AI clinical impression.
              </div>
            ) : (
              <>
                <div style={{ background: '#0b1120', border: `1px solid ${activeColor}`, borderRadius: '14px', padding: '14px', textAlign: 'center' }}>
                  <div style={{ fontSize: '34px', fontWeight: 800, color: activeColor, fontFamily: 'JetBrains Mono, monospace' }}>
                    {analysis.stenosis.narrowing_percentage}%
                  </div>
                  <div style={{ fontSize: '11px', color: '#94a3b8', letterSpacing: '1px' }}>ARTERIAL NARROWING</div>
                  <div style={{ marginTop: '8px', display: 'inline-flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ padding: '4px 12px', borderRadius: '999px', background: `${activeColor}22`, color: activeColor, fontWeight: 700, fontSize: '12px' }}>
                      {analysis.stenosis.severity.toUpperCase()}
                    </span>
                    <span style={{ fontSize: '11px', color: '#94a3b8', fontFamily: 'JetBrains Mono, monospace' }}>
                      Conf: {analysis.stenosis.confidence_score}%
                    </span>
                  </div>
                </div>

                {/* SCAN-DERIVED CLINICAL IMPRESSION CARD */}
                {analysis.clinical_profile && (
                  <div style={{ background: '#0b1120', border: '1px solid #1e293b', borderRadius: '12px', padding: '12px' }}>
                    <span style={{ fontSize: '11px', color: '#38bdf8', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '5px', marginBottom: '6px' }}>
                      <Stethoscope size={13} /> SCAN-DERIVED CLINICAL IMPRESSION
                    </span>
                    <p style={{ margin: '0 0 6px', fontSize: '12px', color: '#e2e8f0', lineHeight: 1.45 }}>
                      {analysis.clinical_profile.ai_clinical_impression}
                    </p>
                    <span style={{ fontSize: '11px', color: '#94a3b8', display: 'block' }}>
                      • {analysis.clinical_profile.morphological_finding}
                    </span>
                  </div>
                )}

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
                  <div style={{ background: '#0b1120', border: '1px solid #1e293b', padding: '10px 12px', borderRadius: '10px' }}>
                    <span style={{ fontSize: '11px', color: '#94a3b8' }}>Reference D_ref</span>
                    <div style={{ fontSize: '16px', fontWeight: 700, color: '#38bdf8', fontFamily: 'JetBrains Mono, monospace', marginTop: '2px' }}>{analysis.stenosis.reference_diameter_px} px</div>
                  </div>
                  <div style={{ background: '#0b1120', border: '1px solid #1e293b', padding: '10px 12px', borderRadius: '10px' }}>
                    <span style={{ fontSize: '11px', color: '#94a3b8' }}>Minimum D_sten</span>
                    <div style={{ fontSize: '16px', fontWeight: 700, color: '#f43f5e', fontFamily: 'JetBrains Mono, monospace', marginTop: '2px' }}>{analysis.stenosis.stenosis_diameter_px} px</div>
                  </div>
                </div>

                <div style={{ background: '#0b1120', border: '1px solid #1e293b', borderRadius: '12px', padding: '12px' }}>
                  <span style={{ fontSize: '11px', color: '#94a3b8', display: 'block', marginBottom: '6px' }}>DIAMETER PROFILE ALONG VESSEL (PX)</span>
                  <ResponsiveContainer width="100%" height={120}>
                    <AreaChart data={analysis.stenosis.diameter_profile}>
                      <XAxis dataKey="idx" hide />
                      <YAxis stroke="#64748b" fontSize={11} width={28} />
                      <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #334155', fontSize: '12px' }} />
                      <ReferenceLine y={analysis.stenosis.reference_diameter_px} stroke="#38bdf8" strokeDasharray="3 3" />
                      <Area type="monotone" dataKey="diameter" stroke="#f43f5e" strokeWidth={2} fill="#f43f5e" fillOpacity={0.2} />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>

                <div style={{ display: 'flex', gap: '8px', padding: '10px', borderRadius: '10px', background: 'rgba(245,158,11,0.08)', border: '1px solid rgba(245,158,11,0.25)', fontSize: '11px', color: '#fbbf24' }}>
                  <AlertTriangle size={15} style={{ flexShrink: 0 }} />
                  <span>AI-assisted geometric lumen measurements for clinical decision support and research only; final stenosis grading requires cardiologist verification.</span>
                </div>
              </>
            )}
          </div>
        </div>
      )}

      {activeTab === 'benchmarks' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          <ModelComparison />
          <PerformanceReport />
        </div>
      )}

      {showReportModal && analysis && (
        <div style={{ position: 'fixed', inset: 0, background: 'rgba(2,6,23,0.88)', zIndex: 50, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '20px' }}>
          <div style={{ background: '#0f172a', border: '1px solid #334155', borderRadius: '16px', maxWidth: '680px', width: '100%', padding: '24px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid #1e293b', paddingBottom: '12px' }}>
              <div>
                <span className="eyebrow">AUTOMATED VASCULAR ANALYSIS REPORT</span>
                <h2 style={{ margin: '4px 0 0', fontSize: '18px' }}>Case {caseData?.caseId} — Patient {caseData?.patientId}</h2>
              </div>
              <button type="button" className="overview-primary-btn" onClick={() => window.print()}>
                Print / Save PDF
              </button>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', margin: '16px 0', fontSize: '13px' }}>
              <div><strong>Architecture:</strong> {analysis.model_used}</div>
              <div><strong>Narrowing Estimate:</strong> {analysis.stenosis.narrowing_percentage}% ({analysis.stenosis.severity})</div>
              <div><strong>Reference / Stenosis Width:</strong> {analysis.stenosis.reference_diameter_px}px / {analysis.stenosis.stenosis_diameter_px}px</div>
              <div><strong>Vessel Length / Branches:</strong> {analysis.vessel_structure.vessel_length_px}px / {analysis.vessel_structure.branching_points}</div>
              <div style={{ gridColumn: '1 / -1', background: '#0b1120', padding: '10px', borderRadius: '8px', border: '1px solid #1e293b' }}>
                <strong>AI Clinical Impression:</strong> {analysis.clinical_profile?.ai_clinical_impression}
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', marginTop: '12px' }}>
              <img src={analysis.images.stenosis_localized} alt="Localized Stenosis" style={{ width: '100%', borderRadius: '8px', border: '1px solid #1e293b' }} />
              <img src={analysis.images.gradcam} alt="Grad-CAM++" style={{ width: '100%', borderRadius: '8px', border: '1px solid #1e293b' }} />
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '16px' }}>
              <span style={{ fontSize: '11px', color: '#94a3b8' }}>Disclaimer: AI-assisted research output; not a standalone clinical diagnosis.</span>
              <button type="button" className="cancel-button" onClick={() => setShowReportModal(false)}>
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}