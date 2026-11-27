import { useState, useRef, useEffect } from 'react'
import {
  ArrowLeft, FileImage, HeartPulse,
  Info, ScanLine, Trash2, UploadCloud, UserRound,
  Sparkles, SlidersHorizontal, FileCheck2, Download,
  Grid, Cpu, GitBranch, CheckCircle2
} from 'lucide-react'
import { apiGet, apiPost, fetchImageUrl, downloadReport } from '../Services/api'

const severityOrder = { Unavailable: 0, Mild: 1, Moderate: 2, Severe: 3 }

function buildAnalysis(run, originalUrl, artifactUrls, modelName) {
  const regions = (run.stenosis_findings || []).map((finding, index) => {
    const details = finding.measurement_details || {}
    return {
      region_id: index + 1,
      severity: finding.severity || 'Unavailable',
      percentage_narrowing: finding.stenosis_percentage,
      ...details
    }
  })
  const successful = regions.filter(region =>
    region.measurement_available &&
    region.percentage_narrowing !== null &&
    region.percentage_narrowing !== undefined &&
    Number.isFinite(Number(region.percentage_narrowing))
  )
  const maximum = successful.reduce(
    (best, region) => !best || Number(region.percentage_narrowing) > Number(best.percentage_narrowing)
      ? region
      : best,
    null
  )
  const highestSeverity = successful.reduce(
    (highest, region) => severityOrder[region.severity] > severityOrder[highest]
      ? region.severity
      : highest,
    'Unavailable'
  )
  const probabilities = regions
    .map(region => Number(region.mean_probability))
    .filter(Number.isFinite)

  return {
    analysis_run_id: run.id,
    status: run.status,
    completed_at: run.completed_at,
    model_used: modelName || 'FrozenFineTuned MultiScaleResidualUNet (stenosis detection)',
    successful_measurements: successful.length,
    unavailable_measurements: regions.length - successful.length,
    regions,
    stenosis: {
      detected_region_count: regions.length,
      narrowing_percentage: maximum ? Number(maximum.percentage_narrowing) : null,
      severity: highestSeverity,
      reference_diameter_px: maximum?.reference_diameter_pixels ?? null,
      stenosis_diameter_px: maximum?.minimum_diameter_pixels ?? null,
      confidence_score: probabilities.length
        ? Number((probabilities.reduce((sum, value) => sum + value, 0) / probabilities.length * 100).toFixed(1))
        : null
    },
    images: {
      original: originalUrl,
      stenosis_mask: artifactUrls.stenosis_mask || null,
      probability_map: artifactUrls.probability_map || null,
      localization_overlay: artifactUrls.localization_overlay || null
    },
    reports: run.reports || []
  }
}

export default function CaseDetails({ caseData, onBack }) {
  const caseKey = caseData?.backendCaseId || caseData?.caseId

  const [activeTab, setActiveTab] = useState('studio')
  const [fileName, setFileName] = useState('')
  const [selectedFile, setSelectedFile] = useState(null)
  const [localPreview, setLocalPreview] = useState(null)
  const [analysis, setAnalysis] = useState(null)
  const [analysisHistory, setAnalysisHistory] = useState([])
  const [pipelineStep, setPipelineStep] = useState(0)
  const [savedBadge, setSavedBadge] = useState(false)
  const [error, setError] = useState('')

  const [viewMode, setViewMode] = useState('split')
  const [splitPos, setSplitPos] = useState(50)
  const [rightLayer, setRightLayer] = useState('localization_overlay')
  const viewportRef = useRef(null)

  useEffect(() => {
    let active = true
    const objectUrls = []
    async function loadHistory() {
      try {
        const runs = await apiGet(`/cases/${encodeURIComponent(caseKey)}/analysis-runs`)
        const stenosisRuns = runs.filter(run =>
          run.status === 'completed' &&
          (run.artifacts || []).some(
            artifact => artifact.artifact_type === 'stenosis_mask'
          )
        )
        if (active) setAnalysisHistory(stenosisRuns)
        const latest = stenosisRuns[0]
        if (!latest || !active) return
        const originalUrl = await fetchImageUrl(`/images/${encodeURIComponent(latest.image_id)}/content`)
        objectUrls.push(originalUrl)
        const artifactUrls = {}
        for (const artifact of latest.artifacts || []) {
          artifactUrls[artifact.artifact_type] = await fetchImageUrl(
            `/analysis-runs/${encodeURIComponent(latest.id)}/artifacts/${encodeURIComponent(artifact.artifact_type)}`
          )
          objectUrls.push(artifactUrls[artifact.artifact_type])
        }
        if (!active) return
        const result = buildAnalysis(latest, originalUrl, artifactUrls, caseData?.modelVersion)
        setAnalysis(result)
        setFileName('Previously uploaded image')
        setLocalPreview(originalUrl)
        setPipelineStep(5)
        setSavedBadge(true)
      } catch (loadError) {
        if (active) setError(loadError.message)
      }
    }
    loadHistory()
    return () => {
      active = false
      objectUrls.forEach(URL.revokeObjectURL)
    }
  }, [caseKey, caseData?.modelVersion])

  const clearSavedCaseData = () => {
    setSelectedFile(null)
    setFileName('')
    setLocalPreview(null)
    setAnalysis(null)
    setPipelineStep(0)
    setSavedBadge(false)
  }

  const stages = [
    '1. Image validation and upload',
    '2. Image preprocessing and normalization',
    '3. Stenosis model inference and experimental measurements'
  ]

  const runPipeline = async (fileToProcess = selectedFile) => {
    if (!fileToProcess) return
    setError('')
    setSavedBadge(false)
    setPipelineStep(1)

    try {
      const formData = new FormData()
      formData.append('file', fileToProcess)
      formData.append('case_id', caseKey)
      const uploaded = await apiPost('/images', formData)
      setPipelineStep(2)
      await apiPost(`/images/${encodeURIComponent(uploaded.id)}/preprocess`)
      setPipelineStep(3)
      const prediction = await apiPost(`/images/${encodeURIComponent(uploaded.id)}/predict`)
      setPipelineStep(4)
      const run = await apiGet(`/analysis-runs/${encodeURIComponent(prediction.analysis_run_id)}`)
      const artifactUrls = {}
      for (const artifact of run.artifacts || []) {
        artifactUrls[artifact.artifact_type] = await fetchImageUrl(
          `/analysis-runs/${encodeURIComponent(run.id)}/artifacts/${encodeURIComponent(artifact.artifact_type)}`
        )
      }
      const originalUrl = URL.createObjectURL(fileToProcess)
      setLocalPreview(originalUrl)
      setAnalysis(buildAnalysis(run, originalUrl, artifactUrls, prediction.model))
      setFileName(fileToProcess.name)
      setPipelineStep(5)
      setSavedBadge(true)
    } catch (err) {
      setPipelineStep(0)
      setError(err.message)
    }
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
      analysis_run_id: analysis.analysis_run_id,
      analysis_status: analysis.status,
      completed_at: analysis.completed_at,
      stenosis_findings: analysis.regions,
      disclaimer: 'AI-assisted decision support only; not an autonomous clinical diagnosis.'
    }
    downloadFile(JSON.stringify(payload, null, 2), `${caseKey}_report.json`, 'application/json')
  }

  const exportCSV = () => {
    if (!analysis) return
    const rows = [
      ['Case ID', 'Patient ID', 'Scan File', 'Model', 'Narrowing %', 'Severity', 'Ref Diameter (px)', 'Min Diameter (px)'],
      [
        caseData?.caseId,
        caseData?.patientId,
        fileName,
        analysis.model_used,
        analysis.stenosis.narrowing_percentage,
        analysis.stenosis.severity,
        analysis.stenosis.reference_diameter_px,
        analysis.stenosis.stenosis_diameter_px
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
      {/* HEADER */}
      <div className="details-page-header">
        <div>
          <button type="button" className="details-back-button" onClick={onBack} style={{ marginBottom: '8px' }}>
            <ArrowLeft size={15} /> Back to Patient Cases
          </button>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
            <h1 style={{ margin: 0, fontSize: '23px' }}>Stenosis Analysis Workstation</h1>
            <span style={{ fontSize: '12px', padding: '3px 10px', borderRadius: '999px', background: '#0f172a', border: '1px solid #1e293b', color: '#38bdf8', fontFamily: 'JetBrains Mono, monospace' }}>
              {caseData?.caseId} · {caseData?.patientId}
            </span>
            {savedBadge && (
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: '5px', fontSize: '11px', padding: '3px 10px', borderRadius: '999px', background: 'rgba(16,185,129,0.14)', border: '1px solid rgba(16,185,129,0.35)', color: '#34d399', fontWeight: 600 }}>
                <CheckCircle2 size={12} /> Analysis saved to this case
              </span>
            )}
          </div>
        </div>

        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
          {[
            { id: 'studio', label: '1. Stenosis Analysis' },
            { id: 'benchmarks', label: '2. Analysis History' }
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
          <span>Analysis status:</span>
          <strong style={{ color: analysis ? activeColor : '#94a3b8' }}>
            {analysis?.status || 'Awaiting analysis'}
          </strong>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <Cpu size={15} color="#10b981" />
          <span>Model:</span>
          <strong>Multiscale Residual U-Net (stenosis detection only)</strong>
        </div>

        {analysis && (
          <div style={{ marginLeft: 'auto', display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
            <button type="button" onClick={exportJSON} style={{ display: 'inline-flex', alignItems: 'center', gap: '5px', background: '#1e293b', border: '1px solid #334155', color: '#e2e8f0', padding: '6px 11px', borderRadius: '8px', cursor: 'pointer', fontSize: '12px' }}>
              <Download size={13} /> JSON
            </button>
            <button type="button" onClick={exportCSV} style={{ display: 'inline-flex', alignItems: 'center', gap: '5px', background: '#1e293b', border: '1px solid #334155', color: '#e2e8f0', padding: '6px 11px', borderRadius: '8px', cursor: 'pointer', fontSize: '12px' }}>
              <Download size={13} /> CSV
            </button>
            <button type="button" onClick={async () => {
              try {
                const reportId = analysis.reports?.[0]?.id
                if (!reportId) throw new Error('No persisted report is associated with this analysis.')
                await downloadReport(reportId)
              } catch (downloadError) {
                setError(downloadError.message)
              }
            }} style={{ display: 'inline-flex', alignItems: 'center', gap: '5px', background: '#0284c7', border: 'none', color: '#fff', padding: '6px 12px', borderRadius: '8px', cursor: 'pointer', fontSize: '12px', fontWeight: 600 }}>
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
                <button type="button" title="Clear current view" onClick={clearSavedCaseData} style={{ background: 'none', border: 'none', color: '#f87171', cursor: 'pointer' }}>
                  <Trash2 size={15} />
                </button>
              </div>
            )}

            {selectedFile && pipelineStep === 0 && (
              <button type="button" onClick={() => runPipeline()} style={{ width: '100%', padding: '10px', borderRadius: '10px', border: 'none', background: 'linear-gradient(135deg, #0284c7, #2563eb)', color: '#fff', fontWeight: 700, fontSize: '13px', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px' }}>
                <Sparkles size={15} /> Execute AI Pipeline
              </button>
            )}

            {pipelineStep > 0 && (
              <div style={{ background: '#0b1120', border: '1px solid #1e293b', borderRadius: '10px', padding: '12px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: '#38bdf8', fontWeight: 700 }}>
                  <span>PIPELINE STATUS</span>
                  {analysis && <span>{analysis.status}</span>}
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
                  <GitBranch size={13} /> STENOSIS MODEL SUMMARY
                </span>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px', textAlign: 'center' }}>
                  <div style={{ background: '#0f172a', padding: '8px 4px', borderRadius: '8px' }}>
                    <span style={{ fontSize: '10px', color: '#94a3b8', display: 'block' }}>Detected regions</span>
                    <strong style={{ fontSize: '13px', fontFamily: 'JetBrains Mono, monospace' }}>{analysis.stenosis.detected_region_count}</strong>
                  </div>
                  <div style={{ background: '#0f172a', padding: '8px 4px', borderRadius: '8px' }}>
                    <span style={{ fontSize: '10px', color: '#94a3b8', display: 'block' }}>Successful measurements</span>
                    <strong style={{ fontSize: '13px', fontFamily: 'JetBrains Mono, monospace', color: '#10b981' }}>{analysis.successful_measurements}</strong>
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
                  <Grid size={13} /> Available Outputs
                </button>
              </div>

              {analysis && viewMode === 'split' && (
                <div style={{ display: 'flex', gap: '5px' }}>
                  {[
                    { id: 'localization_overlay', label: 'Localization overlay' },
                    { id: 'stenosis_mask', label: 'Predicted stenosis mask' },
                    { id: 'probability_map', label: 'Probability map' }
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
                  { title: 'Original angiogram', img: analysis.images.original },
                  { title: 'Stenosis bounding-box overlay', img: analysis.images.localization_overlay },
                  { title: 'Predicted stenosis mask', img: analysis.images.stenosis_mask },
                  { title: 'Model probability map', img: analysis.images.probability_map }
                ].filter(cell => cell.img).map(cell => (
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
                      <img src={analysis.images[rightLayer]} alt={rightLayer.replaceAll('_', ' ')} style={{ width: '100%', height: '100%', objectFit: 'contain', borderRadius: '10px' }} />
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
                      {analysis.stenosis.narrowing_percentage === null ? 'Unavailable' : `${analysis.stenosis.narrowing_percentage}%`}
                    </div>
                    <div style={{ fontSize: '11px', color: '#94a3b8', letterSpacing: '1px' }}>MAX ESTIMATED NARROWING · EXPERIMENTAL</div>
                  <div style={{ marginTop: '8px', display: 'inline-flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ padding: '4px 12px', borderRadius: '999px', background: `${activeColor}22`, color: activeColor, fontWeight: 700, fontSize: '12px' }}>
                      {analysis.stenosis.severity.toUpperCase()}
                    </span>
                    <span style={{ fontSize: '11px', color: '#94a3b8', fontFamily: 'JetBrains Mono, monospace' }}>
                      Mean region probability: {analysis.stenosis.confidence_score === null ? 'Unavailable' : `${analysis.stenosis.confidence_score}%`}
                    </span>
                  </div>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
                  <div style={{ background: '#0b1120', border: '1px solid #1e293b', padding: '10px 12px', borderRadius: '10px' }}>
                    <span style={{ fontSize: '11px', color: '#94a3b8' }}>Reference D_ref</span>
                    <div style={{ fontSize: '16px', fontWeight: 700, color: '#38bdf8', fontFamily: 'JetBrains Mono, monospace', marginTop: '2px' }}>{analysis.stenosis.reference_diameter_px ?? 'Unavailable'}{analysis.stenosis.reference_diameter_px === null ? '' : ' px'}</div>
                  </div>
                  <div style={{ background: '#0b1120', border: '1px solid #1e293b', padding: '10px 12px', borderRadius: '10px' }}>
                    <span style={{ fontSize: '11px', color: '#94a3b8' }}>Minimum D_sten</span>
                    <div style={{ fontSize: '16px', fontWeight: 700, color: '#f43f5e', fontFamily: 'JetBrains Mono, monospace', marginTop: '2px' }}>{analysis.stenosis.stenosis_diameter_px ?? 'Unavailable'}{analysis.stenosis.stenosis_diameter_px === null ? '' : ' px'}</div>
                  </div>
                </div>

                <div style={{ background: '#0b1120', border: '1px solid #1e293b', borderRadius: '12px', padding: '12px' }}>
                  <span style={{ fontSize: '11px', color: '#94a3b8', display: 'block', marginBottom: '8px' }}>PER-REGION RESULTS</span>
                  <div style={{ fontSize: '12px', color: '#cbd5e1', marginBottom: '8px' }}>
                    Regions: {analysis.stenosis.detected_region_count} · Measurements available: {analysis.successful_measurements} · Unavailable: {analysis.unavailable_measurements}
                  </div>
                  {analysis.regions.map(region => (
                    <div key={region.region_id} style={{ borderTop: '1px solid #1e293b', padding: '7px 0', fontSize: '11px', display: 'flex', justifyContent: 'space-between', gap: '8px' }}>
                      <strong>Region {region.region_id}</strong>
                      <span>
                        {region.percentage_narrowing == null
                          ? 'Measurement unavailable'
                          : `${region.percentage_narrowing}% · ${region.severity} · Experimental`}
                      </span>
                    </div>
                  ))}
                  {analysis.completed_at && (
                    <div style={{ marginTop: '8px', color: '#94a3b8', fontSize: '11px' }}>
                      Completed: {new Date(analysis.completed_at).toLocaleString()}
                    </div>
                  )}
                </div>

              </>
            )}
          </div>
        </div>
      )}

      {activeTab === 'benchmarks' && (
        <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: '16px', padding: '18px' }}>
          <h2 style={{ marginTop: 0 }}>Saved analysis history</h2>
          {analysisHistory.length === 0 ? (
            <p style={{ color: '#94a3b8' }}>No saved analyses for this case.</p>
          ) : analysisHistory.map(run => (
            <div key={run.id} style={{ borderTop: '1px solid #1e293b', padding: '12px 0', display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap' }}>
              <span>{run.status} · {run.completed_at ? new Date(run.completed_at).toLocaleString() : 'In progress'}</span>
              <span>{run.stenosis_findings?.length || 0} stenosis region(s) · {run.id}</span>
              {run.reports?.[0]?.id && (
                <button type="button" className="overview-secondary-btn" onClick={() => downloadReport(run.reports[0].id)}>
                  Download persisted report
                </button>
              )}
            </div>
          ))}
        </div>
      )}

    </div>
  )
}