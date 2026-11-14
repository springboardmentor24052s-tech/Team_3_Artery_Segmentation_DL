import { useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Activity, ArrowRight, BellRing, CheckCircle2,
  Clock3, FilePlus2, Layers, ShieldCheck, Users
} from 'lucide-react'

const STORAGE_KEY = 'vascular_case_analyses_v1'

export default function Dashboard() {
  const navigate = useNavigate()

  const { cases, analyses } = useMemo(() => {
    try {
      const savedCases = JSON.parse(localStorage.getItem('cases') || '[]')
      const savedAnalyses = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}')
      return {
        cases: Array.isArray(savedCases) ? savedCases : [],
        analyses: savedAnalyses && typeof savedAnalyses === 'object' ? savedAnalyses : {}
      }
    } catch {
      return { cases: [], analyses: {} }
    }
  }, [])

  const stats = useMemo(() => {
    const analyzedEntries = Object.values(analyses)
    const criticalCount = analyzedEntries.filter(
      item => ['Severe', 'Critical'].includes(item?.analysis?.stenosis?.severity)
    ).length
    const pendingCount = cases.filter(c => c.status === 'Pending').length

    return {
      totalPatients: cases.length,
      analyzedCount: analyzedEntries.length,
      criticalCount,
      pendingCount
    }
  }, [cases, analyses])

  return (
    <div>
      {/* TOPBAR */}
      <div className="overview-topbar">
        <div>
          <span className="overview-eyebrow">CLINICAL COMMAND CENTER</span>
          <h1>VascularAI Diagnostic Overview</h1>
          <p>Real-time coronary artery segmentation, stenosis grading, and Grad-CAM++ explainability telemetry.</p>
        </div>

        <button
          type="button"
          className="overview-primary-btn"
          onClick={() => navigate('/cases')}
        >
          <FilePlus2 size={17} />
          Open Patient Cases
        </button>
      </div>

      {/* WELCOME BANNER */}
      <div className="overview-welcome">
        <div>
          <div className="overview-live">
            <span />
            AI INFERENCE ENGINE ONLINE · ATTENTION U-NET (v2.0)
          </div>
          <h2>Automated Coronary Stenosis & Morphometry Workstation</h2>
          <p>
            Select any patient record to run bilateral CLAHE preprocessing, Frangi + morphological vessel extraction, centerline tortuosity mapping, and Grad-CAM++ saliency verification.
          </p>
        </div>
        <Activity size={54} className="overview-welcome-icon" />
      </div>

      {/* LIVE METRIC CARDS */}
      <div className="overview-stat-grid">
        <div className="overview-stat-card">
          <div className="overview-stat-icon">
            <Users size={20} />
          </div>
          <span className="overview-stat-value">{stats.totalPatients}</span>
          <h3>Registered Cases</h3>
          <p>Active patient cohort in workspace</p>
        </div>

        <div className="overview-stat-card">
          <div className="overview-stat-icon" style={{ background: 'rgba(16, 185, 129, 0.14)', color: '#34d399' }}>
            <CheckCircle2 size={20} />
          </div>
          <span className="overview-stat-value" style={{ color: '#34d399' }}>{stats.analyzedCount}</span>
          <h3>AI Scans Completed</h3>
          <p>Saved with full 6-stage telemetry</p>
        </div>

        <div className="overview-stat-card">
          <div className="overview-stat-icon" style={{ background: 'rgba(239, 68, 68, 0.14)', color: '#f87171' }}>
            <BellRing size={20} />
          </div>
          <span className="overview-stat-value" style={{ color: '#f87171' }}>{stats.criticalCount}</span>
          <h3>High / Critical Alerts</h3>
          <p>Stenosis ≥ 70% requiring review</p>
        </div>

        <div className="overview-stat-card">
          <div className="overview-stat-icon" style={{ background: 'rgba(245, 158, 11, 0.14)', color: '#fbbf24' }}>
            <Clock3 size={20} />
          </div>
          <span className="overview-stat-value" style={{ color: '#fbbf24' }}>{stats.pendingCount}</span>
          <h3>Pending Angiograms</h3>
          <p>Awaiting scan acquisition</p>
        </div>
      </div>

      {/* RECENT CASES & QUICK MODULES */}
      <div className="overview-content-grid">
        <div className="overview-panel">
          <div className="overview-panel-heading">
            <div>
              <h2>Patient Case Telemetry</h2>
              <p>Live narrowing & risk status derived from saved angiograms</p>
            </div>
            <button type="button" className="overview-text-btn" onClick={() => navigate('/cases')}>
              View All <ArrowRight size={14} />
            </button>
          </div>

          <div className="overview-table-scroll">
            <table className="overview-table">
              <thead>
                <tr>
                  <th>Case / Patient</th>
                  <th>Narrowing</th>
                  <th>AI Risk Profile</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {cases.slice(0, 6).map((item) => {
                  const scan = analyses[item.caseId]?.analysis
                  const narrowing = scan?.stenosis?.narrowing_percentage
                  const severity = scan?.stenosis?.severity

                  return (
                    <tr
                      key={item.caseId}
                      onClick={() => navigate('/cases')}
                      style={{ cursor: 'pointer' }}
                    >
                      <td>
                        <strong>{item.patientId}</strong>
                        <div style={{ fontSize: '11px', color: '#64748b' }}>{item.caseId} · {item.age}y/{item.gender}</div>
                      </td>
                      <td style={{ fontFamily: 'JetBrains Mono, monospace', fontWeight: 700, color: narrowing !== undefined ? '#38bdf8' : '#64748b' }}>
                        {narrowing !== undefined ? `${narrowing}% (${severity})` : '—'}
                      </td>
                      <td style={{ fontSize: '12px', color: '#cbd5e1' }}>
                        {scan?.clinical_profile?.inferred_risk_profile || item.riskFactors || 'Awaiting Scan'}
                      </td>
                      <td>
                        <span className={`overview-status ${(item.status || 'pending').toLowerCase().replace(/\s+/g, '-')}`}>
                          {item.status || 'Pending'}
                        </span>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>

        {/* WORKSTATION MODULES */}
        <div className="overview-panel" style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          <div className="overview-panel-heading" style={{ marginBottom: '4px' }}>
            <div>
              <h2>Diagnostic Pipeline Modules</h2>
              <p>Direct access to Milestone 1–3 clinical tools</p>
            </div>
          </div>

          {[
            {
              title: '1. Patient Case & XAI Studio',
              desc: 'Upload ARCADE scans, inspect 6-stage pipeline, and export JSON/CSV/PDF reports.',
              path: '/cases',
              tag: 'Milestone 1–3'
            },
            {
              title: '2. Artery Segmentation Studio',
              desc: 'CLAHE contrast normalization, binary lumen extraction & centerline skeleton.',
              path: '/segmentation',
              tag: 'Milestone 2'
            },
            {
              title: '3. Dual-Model Stenosis & Grad-CAM++',
              desc: 'Compare Baseline U-Net vs Attention U-Net narrowing % and saliency heatmaps.',
              path: '/stenosis',
              tag: 'Milestone 3'
            },
            {
              title: '4. Quantitative Model Benchmarks',
              desc: 'Evaluate Dice, IoU, Precision, Recall, and vessel mask quality metrics.',
              path: '/analytics',
              tag: 'Validation'
            }
          ].map((mod) => (
            <div
              key={mod.title}
              onClick={() => navigate(mod.path)}
              style={{
                background: '#0b1220',
                border: '1px solid #1e293b',
                borderRadius: '12px',
                padding: '14px 16px',
                cursor: 'pointer',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                gap: '12px'
              }}
            >
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Layers size={15} color="#38bdf8" />
                  <strong style={{ fontSize: '13.5px' }}>{mod.title}</strong>
                </div>
                <p style={{ margin: '4px 0 0', fontSize: '12px', color: '#94a3b8' }}>{mod.desc}</p>
              </div>
              <span style={{ fontSize: '10.5px', fontWeight: 700, color: '#38bdf8', background: 'rgba(56,189,248,0.1)', padding: '4px 9px', borderRadius: '999px', whiteSpace: 'nowrap' }}>
                {mod.tag}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* CLINICAL DISCLAIMER */}
      <div className="overview-footer">
        <ShieldCheck size={16} style={{ flexShrink: 0 }} />
        <span>
          AI-assisted geometric lumen measurements for clinical decision support and research only; final stenosis grading requires cardiologist verification.
        </span>
      </div>
    </div>
  )
}