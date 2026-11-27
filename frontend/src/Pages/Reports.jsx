import { useEffect, useState } from 'react'
import { Activity, Download, FileText, Printer, ShieldAlert } from 'lucide-react'

import { apiGet, downloadReport } from '../Services/api'

const severityRank = { Mild: 1, Moderate: 2, Severe: 3 }

function isStenosisRun(run) {
  return (run.artifacts || []).some(
    artifact => artifact.artifact_type === 'stenosis_mask'
  )
}

export default function Reports() {
  const [reports, setReports] = useState([])
  const [selectedId, setSelectedId] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    let active = true
    async function loadReports() {
      try {
        const cases = await apiGet('/cases')
        const entries = (await Promise.all(cases.map(async caseRecord => {
          const [runs, patient] = await Promise.all([
            apiGet(`/cases/${encodeURIComponent(caseRecord.id)}/analysis-runs`),
            apiGet(`/patients/${encodeURIComponent(caseRecord.patient_id)}`)
          ])
          return runs
            .filter(isStenosisRun)
            .flatMap(run => (run.reports || []).map(report => ({
              ...report,
              caseId: caseRecord.id,
              patientCode: patient.patient_code,
              model: caseRecord.model_version,
              run
            })))
        }))).flat()
        if (active) {
          setReports(entries)
          setSelectedId(entries[0]?.id || '')
        }
      } catch (loadError) {
        if (active) setError(loadError.message)
      } finally {
        if (active) setLoading(false)
      }
    }
    loadReports()
    return () => { active = false }
  }, [])

  const selected = reports.find(report => report.id === selectedId)
  const findings = selected?.run.stenosis_findings || []
  const measured = findings.filter(finding => finding.stenosis_percentage != null)
  const maximum = measured.reduce(
    (best, finding) => Number(finding.stenosis_percentage) > Number(best?.stenosis_percentage)
      ? finding
      : best,
    null
  )
  const highestSeverity = measured.reduce(
    (best, finding) => (severityRank[finding.severity] || 0) > (severityRank[best] || 0)
      ? finding.severity
      : best,
    'Unavailable'
  )

  const exportCSV = () => {
    if (!selected) return
    const header = ['Region', 'Narrowing (%)', 'Severity', 'Measurement available', 'Minimum diameter (px)', 'Reference diameter (px)']
    const rows = findings.map((finding, index) => {
      const details = finding.measurement_details || {}
      return [
        finding.segment_location || `region-${index + 1}`,
        finding.stenosis_percentage ?? '',
        finding.severity || 'Unavailable',
        details.measurement_available ? 'yes' : 'no',
        details.minimum_diameter_pixels ?? '',
        details.reference_diameter_pixels ?? ''
      ]
    })
    const csv = [header, ...rows].map(row =>
      row.map(value => `"${String(value).replace(/"/g, '""')}"`).join(',')
    ).join('\n')
    const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }))
    const link = document.createElement('a')
    link.href = url
    link.download = `stenosis-report-${selected.id}.csv`
    link.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '22px', color: '#f8fafc' }}>
      <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: '16px', padding: '22px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '14px' }}>
        <div>
          <span className="eyebrow">SAVED CASE ANALYSES</span>
          <h1 style={{ margin: '4px 0', fontSize: '22px' }}>Analysis Reports</h1>
          <p style={{ margin: 0, fontSize: '13px', color: '#94a3b8' }}>Reports are generated from persisted case analysis records.</p>
        </div>
        <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
          <button type="button" className="overview-secondary-btn" disabled={!selected} onClick={() => selected && downloadReport(selected.id)}>
            <Download size={15} /> Download JSON
          </button>
          <button type="button" className="overview-secondary-btn" disabled={!selected} onClick={exportCSV}>
            <FileText size={15} /> Export measurements CSV
          </button>
          <button type="button" className="overview-primary-btn" disabled={!selected} onClick={() => window.print()}>
            <Printer size={15} /> Print / Save PDF
          </button>
        </div>
      </div>

      {error && <div className="login-error" role="alert">{error}</div>}

      <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: '16px', padding: '22px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', alignItems: 'center', flexWrap: 'wrap', marginBottom: '18px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Activity size={18} color="#38bdf8" />
            <h2 style={{ margin: 0, fontSize: '16px' }}>Persisted stenosis findings</h2>
          </div>
          {reports.length > 0 && (
            <select value={selectedId} onChange={event => setSelectedId(event.target.value)} style={{ background: '#0b1120', color: '#f8fafc', border: '1px solid #334155', borderRadius: '8px', padding: '8px 12px' }}>
              {reports.map(report => (
                <option key={report.id} value={report.id}>
                  {report.patientCode} · {report.caseId} · {new Date(report.generated_at).toLocaleString()}
                </option>
              ))}
            </select>
          )}
        </div>

        {loading ? (
          <p style={{ color: '#94a3b8' }}>Loading reports from the authenticated API…</p>
        ) : !selected ? (
          <p style={{ color: '#94a3b8' }}>No saved analysis reports are available for your cases.</p>
        ) : (
          <>
            <p style={{ color: '#94a3b8', fontSize: '12px' }}>
              {selected.summary} · Completed {selected.run.completed_at ? new Date(selected.run.completed_at).toLocaleString() : 'time unavailable'} · Model: {selected.model || 'Stenosis detection model'}
            </p>
            <div className="overview-stat-grid">
              {[
                ['Detected regions', findings.length],
                ['Successful measurements', measured.length],
                ['Unavailable measurements', findings.length - measured.length],
                ['Maximum narrowing', maximum ? `${maximum.stenosis_percentage}%` : 'Unavailable'],
                ['Highest measured severity', highestSeverity]
              ].map(([label, value]) => (
                <div className="overview-stat-card" key={label}>
                  <span className="overview-stat-value">{value}</span>
                  <h3>{label}</h3>
                </div>
              ))}
            </div>

            <div style={{ overflowX: 'auto', marginTop: '18px' }}>
              <table className="overview-table">
                <thead>
                  <tr><th>Region</th><th>Narrowing (%)</th><th>Severity</th><th>Measurement</th><th>Minimum diameter</th><th>Reference diameter</th></tr>
                </thead>
                <tbody>
                  {findings.length === 0 ? (
                    <tr><td colSpan="6">No regions were detected.</td></tr>
                  ) : findings.map((finding, index) => {
                    const details = finding.measurement_details || {}
                    return (
                      <tr key={finding.id}>
                        <td>{finding.segment_location || `region-${index + 1}`}</td>
                        <td>{finding.stenosis_percentage == null ? 'Unavailable' : finding.stenosis_percentage}</td>
                        <td>{finding.severity || 'Unavailable'}</td>
                        <td>{details.measurement_available ? 'Experimental estimate' : 'Unavailable'}</td>
                        <td>{details.minimum_diameter_pixels == null ? '—' : `${details.minimum_diameter_pixels} px`}</td>
                        <td>{details.reference_diameter_pixels == null ? '—' : `${details.reference_diameter_pixels} px`}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </>
        )}
      </div>

      <div style={{ padding: '12px 16px', background: 'rgba(245, 158, 11, 0.08)', border: '1px solid rgba(245, 158, 11, 0.25)', borderRadius: '10px', fontSize: '12px', color: '#fcd34d', display: 'flex', gap: '8px' }}>
        <ShieldAlert size={16} />
        <span>Stenosis detection is reported independently from artery segmentation. Narrowing measurements are experimental pixel-based estimates, not clinical diagnoses. Missing measurements are reported as unavailable.</span>
      </div>
    </div>
  )
}
