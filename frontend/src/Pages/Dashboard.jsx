import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Activity, ArrowRight, BellRing, CheckCircle2,
  Clock3, FilePlus2, Layers, ShieldCheck, Users
} from 'lucide-react'
import { apiGet } from '../Services/api'

function isStenosisRun(run) {
  return run.status === 'completed' &&
    (run.artifacts || []).some(
      artifact => artifact.artifact_type === 'stenosis_mask'
    )
}

export default function Dashboard() {
  const navigate = useNavigate()
  const [cases, setCases] = useState([])
  const [analyses, setAnalyses] = useState({})
  const [error, setError] = useState('')

  useEffect(() => {
    let active = true
    async function loadDashboard() {
      try {
        const [patients, caseRecords] = await Promise.all([
          apiGet('/patients'),
          apiGet('/cases')
        ])
        const patientById = new Map(patients.map(patient => [patient.id, patient]))
        const analysisByCase = {}
        const caseRows = await Promise.all(caseRecords.map(async record => {
          const patient = patientById.get(record.patient_id)
          const runs = await apiGet(`/cases/${encodeURIComponent(record.id)}/analysis-runs`)
          const latest = runs.find(isStenosisRun)
          if (latest) {
            const findings = latest.stenosis_findings || []
            const measured = findings.filter(finding => finding.stenosis_percentage != null)
            const worst = measured.reduce(
              (current, finding) => Number(finding.stenosis_percentage) > Number(current?.stenosis_percentage)
                ? finding
                : current,
              null
            )
            const mostSevere = measured.reduce((current, finding) => {
              const rank = { Mild: 1, Moderate: 2, Severe: 3 }
              return (rank[finding.severity] || 0) > (rank[current] || 0)
                ? finding.severity
                : current
            }, 'Unavailable')
            analysisByCase[record.id] = {
              analysis: {
                stenosis: {
                  narrowing_percentage: worst?.stenosis_percentage ?? null,
                  severity: mostSevere
                }
              }
            }
          }
          return {
            caseId: record.id,
            patientId: patient?.patient_code || 'Patient unavailable',
            age: patient?.age,
            gender: patient?.gender,
            riskFactors: patient?.risk_factors,
            status: record.analysis_status === 'completed' ? 'Completed' : 'Pending'
          }
        }))
        if (active) {
          setCases(caseRows)
          setAnalyses(analysisByCase)
        }
      } catch (loadError) {
        if (active) setError(loadError.message)
      }
    }
    loadDashboard()
    return () => { active = false }
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
          <p>Patient cases and stenosis analysis summaries from saved backend records.</p>
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
            STENOSIS DETECTION MODEL · RESEARCH OUTPUT
          </div>
          <h2>Case-scoped stenosis analysis</h2>
          <p>
            Upload an angiogram within a case to run the available stenosis model.             Baseline U-Net artery segmentation is available as a separate analysis workflow.
          </p>
        </div>
        <Activity size={54} className="overview-welcome-icon" />
      </div>

      {error && <div className="login-error" role="alert">{error}</div>}

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
          <p>Saved analysis runs</p>
        </div>

        <div className="overview-stat-card">
          <div className="overview-stat-icon" style={{ background: 'rgba(239, 68, 68, 0.14)', color: '#f87171' }}>
            <BellRing size={20} />
          </div>
          <span className="overview-stat-value" style={{ color: '#f87171' }}>{stats.criticalCount}</span>
          <h3>High / Critical Alerts</h3>
          <p>Measured severe regions; review required</p>
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
              desc: 'Create cases, upload images, and inspect persisted stenosis results.',
              path: '/cases',
              tag: 'Milestone 1–3'
            },
            {
              title: '2. Artery Segmentation Studio',
              desc: 'Baseline U-Net artery segmentation is available in its separate workflow; centerlines are unavailable.',
              path: '/segmentation',
              tag: 'Milestone 2'
            },
            {
              title: '3. Stenosis Detection',
              desc: 'Run the available stenosis model independently and review experimental measurements.',
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