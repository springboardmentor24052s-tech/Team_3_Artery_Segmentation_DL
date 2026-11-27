import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Activity,
  ArrowLeft,
  CalendarDays,
  CheckCircle2,
  ClipboardList,
  FilePlus2,
  Search,
  UserRound,
  X,
  Clock3,
  Users
} from 'lucide-react'

import CaseDetails from './CaseDetails'
import { apiGet, apiPost } from '../Services/api'

const initialCaseData = {
  patientId: '',
  age: '',
  gender: '',
  medicalHistory: '',
  riskFactors: '',
  clinicalNotes: ''
}

const filters = ['All', 'Pending', 'In Progress', 'Completed']

function isStenosisRun(run) {
  return (run.artifacts || []).some(
    artifact => artifact.artifact_type === 'stenosis_mask'
  )
}

async function loadWorkspaceCases() {
  const [patients, caseRecords] = await Promise.all([
    apiGet('/patients'),
    apiGet('/cases')
  ])
  const patientsById = new Map(patients.map(patient => [patient.id, patient]))
  const analyses = {}
  const mappedCases = await Promise.all(caseRecords.map(async record => {
    const patient = patientsById.get(record.patient_id)
    const runs = await apiGet(`/cases/${encodeURIComponent(record.id)}/analysis-runs`)
    const latestCompleted = runs.find(
      run => run.status === 'completed' && isStenosisRun(run)
    )
    if (latestCompleted) {
      const measurements = (latestCompleted.stenosis_findings || [])
        .filter(finding => finding.stenosis_percentage != null)
      const worst = measurements.reduce(
        (current, finding) => Number(finding.stenosis_percentage) > Number(current?.stenosis_percentage)
          ? finding
          : current,
        null
      )
      analyses[record.id] = {
        run: latestCompleted,
        analysis: {
          stenosis: {
            narrowing_percentage: worst?.stenosis_percentage ?? null,
            severity: worst?.severity || 'Unavailable'
          }
        }
      }
    }
    return {
      caseId: record.id,
      backendCaseId: record.id,
      patientId: patient?.patient_code || 'Patient unavailable',
      patientIdDb: record.patient_id,
      age: patient?.age,
      gender: patient?.gender,
      medicalHistory: patient?.medical_history,
      riskFactors: patient?.risk_factors,
      clinicalNotes: patient?.clinical_notes,
      modelVersion: record.model_version,
      status: record.analysis_status === 'completed'
        ? 'Completed'
        : record.analysis_status === 'running'
          ? 'In Progress'
          : 'Pending',
      createdAt: record.scan_date
    }
  }))
  return { cases: mappedCases, analyses }
}

function Cases() {
  const navigate = useNavigate()

  const [showForm, setShowForm] = useState(false)
  const [cases, setCases] = useState([])
  const [savedAnalyses, setSavedAnalyses] = useState({})
  const [selectedCase, setSelectedCase] = useState(null)
  const [searchTerm, setSearchTerm] = useState('')
  const [activeFilter, setActiveFilter] = useState('All')
  const [formError, setFormError] = useState('')
  const [successMessage, setSuccessMessage] = useState('')
  const [caseData, setCaseData] = useState(initialCaseData)
  const [loading, setLoading] = useState(true)

  const refreshCases = async () => {
    try {
      const workspace = await loadWorkspaceCases()
      setCases(workspace.cases)
      setSavedAnalyses(workspace.analyses)
      setFormError('')
    } catch (loadError) {
      setFormError(loadError.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    let active = true
    loadWorkspaceCases()
      .then(workspace => {
        if (!active) return
        setCases(workspace.cases)
        setSavedAnalyses(workspace.analyses)
      })
      .catch(loadError => {
        if (active) setFormError(loadError.message)
      })
      .finally(() => {
        if (active) setLoading(false)
      })
    return () => { active = false }
  }, [])

  const counts = useMemo(() => ({
    All: cases.length,
    Pending: cases.filter(item => item.status === 'Pending').length,
    'In Progress': cases.filter(item => item.status === 'In Progress').length,
    Completed: cases.filter(item => item.status === 'Completed').length
  }), [cases])

  const filteredCases = useMemo(() => {
    return cases.filter(item => {
      const search = searchTerm.trim().toLowerCase()

      const matchesSearch =
        (item.patientId || '').toLowerCase().includes(search) ||
        (item.caseId || '').toLowerCase().includes(search)

      const matchesFilter =
        activeFilter === 'All' || item.status === activeFilter

      return matchesSearch && matchesFilter
    })
  }, [cases, searchTerm, activeFilter])

  const openCaseForm = () => {
    setSelectedCase(null)
    setCaseData({ ...initialCaseData })
    setFormError('')
    setSuccessMessage('')
    setShowForm(true)
  }

  const closeCaseForm = () => {
    setShowForm(false)
    setFormError('')
    setCaseData({ ...initialCaseData })
  }

  const handleChange = event => {
    const { name, value } = event.target

    setCaseData(previous => ({
      ...previous,
      [name]: value
    }))

    setFormError('')
  }

  const handleSubmit = async event => {
    event.preventDefault()
    setFormError('')
    setSuccessMessage('')

    if (!caseData.patientId.trim()) {
      setFormError('Please enter a Patient ID.')
      return
    }

    if (!caseData.age || Number(caseData.age) < 0 || Number(caseData.age) > 120) {
      setFormError('Enter an age from 0 to 120.')
      return
    }

    if (!caseData.gender) {
      setFormError('Select a gender.')
      return
    }

    const duplicate = cases.some(
      item =>
        item.patientId.toLowerCase() ===
        caseData.patientId.trim().toLowerCase()
    )

    if (duplicate) {
      setFormError('This Patient ID already exists.')
      return
    }

    let createdPatient = null
    try {
      createdPatient = await apiPost('/patients', {
        patient_code: caseData.patientId.trim(),
        age: Number(caseData.age),
        gender: caseData.gender,
        clinical_notes: caseData.clinicalNotes.trim() || null
      })
      const createdCase = await apiPost('/cases', {
        patient_id: createdPatient.id,
        modality: 'Angiography',
        scan_date: new Date().toISOString().slice(0, 10),
        anatomical_region: 'Coronary arteries'
      })
      await refreshCases()
      setSelectedCase({
        ...caseData,
        caseId: createdCase.id,
        backendCaseId: createdCase.id,
        patientId: createdPatient.patient_code,
        patientIdDb: createdPatient.id,
        age: createdPatient.age,
        gender: createdPatient.gender,
        modelVersion: createdCase.model_version
      })
      setCaseData({ ...initialCaseData })
      setShowForm(false)
      setActiveFilter('All')
      setSearchTerm('')
      setSuccessMessage('Patient and imaging case created.')
    } catch (createError) {
      setFormError(createdPatient
        ? `Patient record saved, but case creation failed: ${createError.message}`
        : createError.message)
    }
  }

  const openCaseDetails = item => {
    setSelectedCase(item)
    setShowForm(false)
  }

  const closeCaseDetails = () => {
    setSelectedCase(null)
    refreshCases()
  }

  if (selectedCase) {
    return (
      <CaseDetails
        caseData={selectedCase}
        onBack={closeCaseDetails}
      />
    )
  }

  return (
    <div className="cases-page cases-redesign">
      {/* HEADER */}
      <div className="cases-header cases-redesign-header">
        <div>
          <span className="eyebrow">PATIENT WORKSPACE</span>
          <h1>Patients & Cases</h1>
          <p>
            Create patient and case records, then upload an angiogram for stenosis analysis.
          </p>
        </div>

        <div className="cases-header-actions">
          <button
            type="button"
            className="back-dashboard-button"
            onClick={() => navigate('/overview')}
          >
            <ArrowLeft size={16} />
            Overview
          </button>

          <button
            type="button"
            className="create-case-button"
            onClick={openCaseForm}
          >
            <FilePlus2 size={17} />
            Add New Patient
          </button>
        </div>
      </div>

      {successMessage && (
        <div className="cases-success-message" role="status" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '10px 14px', background: 'rgba(16,185,129,0.14)', border: '1px solid rgba(16,185,129,0.35)', borderRadius: '10px', color: '#34d399', marginBottom: '16px', fontSize: '13px' }}>
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: '8px' }}>
            <CheckCircle2 size={17} />
            {successMessage}
          </span>

          <button
            type="button"
            onClick={() => setSuccessMessage('')}
            style={{ background: 'none', border: 'none', color: '#34d399', cursor: 'pointer' }}
            aria-label="Dismiss message"
          >
            <X size={16} />
          </button>
        </div>
      )}

      {formError && !showForm && (
        <div className="login-error" role="alert">{formError}</div>
      )}

      {!showForm && (
        <>
          {/* SUMMARY */}
          <div className="case-summary-grid">
            <div className="case-summary-card">
              <div className="case-summary-icon">
                <Users size={20} />
              </div>
              <span>Cases</span>
              <strong>{counts.All}</strong>
              <small>Registered records</small>
            </div>

            <div className="case-summary-card">
              <div className="case-summary-icon pending-icon">
                <Clock3 size={20} />
              </div>
              <span>Pending Scan</span>
              <strong>{counts.Pending}</strong>
              <small>Awaiting angiogram</small>
            </div>

            <div className="case-summary-card">
              <div className="case-summary-icon">
                <Activity size={20} />
              </div>
              <span>In Progress</span>
              <strong>{counts['In Progress']}</strong>
              <small>Currently processing</small>
            </div>

            <div className="case-summary-card">
              <div className="case-summary-icon completed-icon">
                <CheckCircle2 size={20} />
              </div>
              <span>AI Analyzed</span>
              <strong>{counts.Completed}</strong>
              <small>Persisted analyses</small>
            </div>
          </div>

          {/* SEARCH */}
          <div className="cases-toolbar">
            <div>
              <h2>Patient Records</h2>
              <p>Click any patient card to upload an angiogram and extract clinical findings.</p>
            </div>

            <div className="case-search">
              <Search size={18} />

              <input
                type="search"
                placeholder="Search Patient ID or Case ID..."
                value={searchTerm}
                onChange={event => setSearchTerm(event.target.value)}
              />

              {searchTerm && (
                <button
                  type="button"
                  onClick={() => setSearchTerm('')}
                  style={{ background: 'none', border: 'none', color: '#94a3b8', cursor: 'pointer' }}
                  aria-label="Clear search"
                >
                  <X size={16} />
                </button>
              )}
            </div>
          </div>

          {/* FILTERS */}
          <div className="case-filter-row">
            {filters.map(filter => (
              <button
                type="button"
                key={filter}
                className={`case-filter ${
                  activeFilter === filter ? 'case-filter-active' : ''
                }`}
                onClick={() => setActiveFilter(filter)}
              >
                {filter}
                <span>{counts[filter]}</span>
              </button>
            ))}
          </div>

          {/* RECORD LIST */}
          {loading ? (
            <p style={{ color: '#94a3b8' }}>Loading cases from the authenticated API…</p>
          ) : filteredCases.length > 0 ? (
            <div className="case-list cases-redesign-list">
              {filteredCases.map((item, index) => {
                const record = savedAnalyses[item.caseId]
                const hasSavedScan = Boolean(record?.analysis)
                const derivedNarrowing = record?.analysis?.stenosis?.narrowing_percentage
                const derivedSeverity = record?.analysis?.stenosis?.severity

                return (
                  <div
                    className="case-card case-redesign-card"
                    key={item.caseId || `${item.patientId}-${index}`}
                    onClick={() => openCaseDetails(item)}
                    role="button"
                    tabIndex={0}
                    onKeyDown={event => {
                      if (event.key === 'Enter') openCaseDetails(item)
                    }}
                  >
                    <div className="case-card-heading">
                      <div className="case-card-icon">
                        <UserRound size={21} />
                      </div>

                      <div className="case-card-title">
                        <span>{item.caseId || `CASE-${index + 1}`}</span>
                        <h3>{item.patientId || 'Patient unavailable'}</h3>
                      </div>

                      <span
                        className={`case-status-badge status-${(
                          item.status || 'Pending'
                        ).toLowerCase().replace(/\s+/g, '-')}`}
                      >
                        {item.status || 'Pending'}
                      </span>
                    </div>

                    <div className="case-card-details">
                      <div>
                        <span>Age / Sex</span>
                        <strong>{item.age || '—'}y / {item.gender || '—'}</strong>
                      </div>

                      <div>
                        <span>Scan Finding</span>
                        {hasSavedScan ? (
                          <strong style={{ color: '#38bdf8', fontFamily: 'JetBrains Mono, monospace' }}>
                            {derivedNarrowing}% ({derivedSeverity})
                          </strong>
                        ) : (
                          <strong style={{ color: '#64748b' }}>Not scanned yet</strong>
                        )}
                      </div>

                      <div style={{ gridColumn: '1 / -1' }}>
                        <span>Medical history / risk factors</span>
                        <strong style={{ color: hasSavedScan ? '#e2e8f0' : '#64748b', fontSize: '12px' }}>
                          {[item.medicalHistory, item.riskFactors].filter(Boolean).join(' · ') || 'No patient notes recorded'}
                        </strong>
                      </div>
                    </div>

                    <div className="case-card-footer">
                      <span style={{ display: 'inline-flex', alignItems: 'center', gap: '5px' }}>
                        <CalendarDays size={14} />
                        {item.createdAt
                          ? new Date(item.createdAt).toLocaleDateString()
                          : 'Date unavailable'}
                      </span>

                      <div className="case-card-actions">
                        <span className="case-view-link">
                          {hasSavedScan ? 'View saved analysis →' : 'Upload & Analyze →'}
                        </span>
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          ) : (
            <div className="empty-cases cases-redesign-empty">
              <div className="empty-cases-icon">
                <ClipboardList size={27} />
              </div>

              <h2>
                {cases.length === 0
                  ? 'No patient records'
                  : 'No matching records'}
              </h2>

              <p>
                {cases.length === 0
                  ? 'Create a patient record to start the imaging workflow.'
                  : 'Try changing the search or status filter.'}
              </p>

              {cases.length === 0 && (
                <button
                  type="button"
                  className="create-case-button"
                  onClick={openCaseForm}
                >
                  <FilePlus2 size={17} />
                  Add New Patient
                </button>
              )}
            </div>
          )}
        </>
      )}

      {/* CREATE PATIENT FORM */}
      {showForm && (
        <div className="case-form-card cases-redesign-form">
          <div className="form-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '18px' }}>
            <div>
              <span className="eyebrow">NEW PATIENT RECORD</span>
              <h2 style={{ margin: '4px 0' }}>Register Patient & Imaging Case</h2>
              <p style={{ margin: 0, fontSize: '13px', color: '#94a3b8' }}>
                These records are stored in the authenticated backend database.
              </p>
            </div>

            <button
              type="button"
              className="cancel-button"
              onClick={closeCaseForm}
              aria-label="Close form"
            >
              <X size={18} />
            </button>
          </div>

          <form onSubmit={handleSubmit}>
            <div className="form-grid">
              <div className="form-group">
                <label htmlFor="patientId">Patient ID *</label>
                <input
                  id="patientId"
                  type="text"
                  name="patientId"
                  placeholder="e.g. ARCADE-ST-21"
                  value={caseData.patientId}
                  onChange={handleChange}
                  required
                />
              </div>

              <div className="form-group">
                <label htmlFor="age">Age *</label>
                <input
                  id="age"
                  type="number"
                  name="age"
                  placeholder="Enter age"
                  min="0"
                  max="120"
                  value={caseData.age}
                  onChange={handleChange}
                  required
                />
              </div>

              <div className="form-group">
                <label htmlFor="gender">Gender *</label>
                <select
                  id="gender"
                  name="gender"
                  value={caseData.gender}
                  onChange={handleChange}
                  required
                >
                  <option value="">Select gender</option>
                  <option value="Male">Male</option>
                  <option value="Female">Female</option>
                  <option value="Other">Other</option>
                </select>
              </div>

              <div className="form-group">
                <label htmlFor="clinicalNotes">Referring Note (Optional)</label>
                <input
                  id="clinicalNotes"
                  type="text"
                  name="clinicalNotes"
                  placeholder="e.g. Coronary angiography evaluation"
                  value={caseData.clinicalNotes}
                  onChange={handleChange}
                />
              </div>
            </div>

            {formError && (
              <div style={{ color: '#fca5a5', fontSize: '13px', marginBottom: '12px' }} role="alert">
                {formError}
              </div>
            )}

            <div style={{ display: 'flex', gap: '10px', justifyContent: 'flex-end', marginTop: '12px' }}>
              <button
                type="button"
                className="cancel-button"
                onClick={closeCaseForm}
              >
                Cancel
              </button>

              <button
                type="submit"
                className="create-case-button"
                disabled={loading}
              >
                <FilePlus2 size={17} />
                Save Patient Record
              </button>
            </div>
          </form>
        </div>
      )}
    </div>
  )
}

export default Cases