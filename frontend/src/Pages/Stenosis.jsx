import { useEffect, useState } from 'react'
import { Activity, Info } from 'lucide-react'

import CaseDetails from './CaseDetails'
import { apiGet } from '../Services/api'

export default function Stenosis() {
  const [cases, setCases] = useState([])
  const [selectedCase, setSelectedCase] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    let active = true
    async function loadCases() {
      try {
        const [caseRecords, patients] = await Promise.all([
          apiGet('/cases'),
          apiGet('/patients')
        ])
        const patientById = new Map(patients.map(patient => [patient.id, patient]))
        const rows = caseRecords.map(record => ({
          caseId: record.id,
          backendCaseId: record.id,
          patientId: patientById.get(record.patient_id)?.patient_code || 'Patient unavailable',
          age: patientById.get(record.patient_id)?.age,
          gender: patientById.get(record.patient_id)?.gender,
          modelVersion: record.model_version
        }))
        if (active) setCases(rows)
      } catch (loadError) {
        if (active) setError(loadError.message)
      }
    }
    loadCases()
    return () => { active = false }
  }, [])

  if (selectedCase) {
    return (
      <CaseDetails
        caseData={selectedCase}
        onBack={() => setSelectedCase(null)}
      />
    )
  }

  return (
    <div style={{ color: '#f8fafc' }}>
      <div className="page-heading">
        <div>
          <span className="eyebrow">RESEARCH MODEL · CASE-SCOPED</span>
          <h1>Stenosis Detection</h1>
          <p>Choose an existing case to upload an image, run the available stenosis model, and review persisted results.</p>
        </div>
      </div>
      {error && <div className="login-error" role="alert">{error}</div>}
      <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: '16px', padding: '22px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '12px' }}>
          <Activity size={20} color="#38bdf8" />
          <strong>Available model: FrozenFineTuned MultiScaleResidualUNet (stenosis detection)</strong>
        </div>
        <p style={{ color: '#94a3b8', fontSize: '13px' }}>
          This model predicts stenosis regions only. It is not the Baseline U-Net artery segmentation model. Narrowing estimates are experimental pixel-based measurements.
        </p>
        {cases.length === 0 ? (
          <p style={{ color: '#94a3b8' }}>Create a patient and case record before running an analysis.</p>
        ) : (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px' }}>
            {cases.map(caseRecord => (
              <button
                type="button"
                key={caseRecord.caseId}
                className="overview-secondary-btn"
                onClick={() => setSelectedCase(caseRecord)}
              >
                {caseRecord.patientId} · {caseRecord.caseId.slice(0, 8)}
              </button>
            ))}
          </div>
        )}
        <div style={{ marginTop: '16px', color: '#fcd34d', fontSize: '12px', display: 'flex', gap: '8px' }}>
          <Info size={16} />
          <span>Baseline U-Net artery segmentation is available in the separate Segmentation workspace. Centerlines and Grad-CAM explanations are not part of this stenosis workflow.</span>
        </div>
      </div>
    </div>
  )
}
