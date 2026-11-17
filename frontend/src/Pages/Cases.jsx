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
  Users,
  Trash2
} from 'lucide-react'

import CaseDetails from './CaseDetails'
import { getAllCaseAnalyses, deleteCaseAnalysis } from '../utils/scanStorage'

const initialCaseData = {
  patientId: '',
  age: '',
  gender: '',
  medicalHistory: '',
  riskFactors: '',
  clinicalNotes: ''
}

const filters = ['All', 'Pending', 'In Progress', 'Completed']

const demoCases = [
  {
    caseId: 'CASE-001',
    patientId: 'ARCADE-ST-01',
    age: '64',
    gender: 'Male',
    medicalHistory: 'Pending angiogram analysis',
    riskFactors: 'Awaiting AI Scan Assessment',
    clinicalNotes: 'Recommended scan: data/stenosis/test/images (Proximal LAD plaque test)',
    presetMode: 'critical_lad',
    status: 'Pending',
    createdAt: '2026-11-01T09:30:00.000Z'
  },
  {
    caseId: 'CASE-002',
    patientId: 'ARCADE-ST-02',
    age: '58',
    gender: 'Female',
    medicalHistory: 'Pending angiogram analysis',
    riskFactors: 'Awaiting AI Scan Assessment',
    clinicalNotes: 'Recommended scan: data/stenosis/test/images (Mid-RCA focal narrowing)',
    presetMode: 'severe_rca',
    status: 'Pending',
    createdAt: '2026-11-02T10:15:00.000Z'
  },
  {
    caseId: 'CASE-003',
    patientId: 'ARCADE-ST-03',
    age: '52',
    gender: 'Male',
    medicalHistory: 'Pending angiogram analysis',
    riskFactors: 'Awaiting AI Scan Assessment',
    clinicalNotes: 'Recommended scan: data/stenosis/test/images (LCX lesion evaluation)',
    presetMode: 'moderate_lcx',
    status: 'Pending',
    createdAt: '2026-11-03T11:20:00.000Z'
  },
  {
    caseId: 'CASE-004',
    patientId: 'ARCADE-ST-04',
    age: '69',
    gender: 'Male',
    medicalHistory: 'Pending angiogram analysis',
    riskFactors: 'Awaiting AI Scan Assessment',
    clinicalNotes: 'Recommended scan: data/stenosis/test/images (Left Main bifurcation)',
    presetMode: 'severe_lmca',
    status: 'Pending',
    createdAt: '2026-11-04T08:45:00.000Z'
  },
  {
    caseId: 'CASE-005',
    patientId: 'ARCADE-ST-05',
    age: '61',
    gender: 'Female',
    medicalHistory: 'Pending angiogram analysis',
    riskFactors: 'Awaiting AI Scan Assessment',
    clinicalNotes: 'Recommended scan: data/stenosis/test/images (Distal RCA tandem check)',
    presetMode: 'moderate_rca',
    status: 'Pending',
    createdAt: '2026-11-05T14:10:00.000Z'
  },
  {
    caseId: 'CASE-006',
    patientId: 'ARCADE-ST-06',
    age: '66',
    gender: 'Female',
    medicalHistory: 'Pending angiogram analysis',
    riskFactors: 'Awaiting AI Scan Assessment',
    clinicalNotes: 'Recommended scan: data/stenosis/test/images (Proximal LCX narrowing)',
    presetMode: 'critical_lcx',
    status: 'Pending',
    createdAt: '2026-11-06T09:50:00.000Z'
  },
  {
    caseId: 'CASE-007',
    patientId: 'ARCADE-ST-07',
    age: '55',
    gender: 'Male',
    medicalHistory: 'Pending angiogram analysis',
    riskFactors: 'Awaiting AI Scan Assessment',
    clinicalNotes: 'Recommended scan: data/stenosis/test/images (Mid-LAD plaque burden)',
    presetMode: 'mild_lad',
    status: 'Pending',
    createdAt: '2026-11-07T16:00:00.000Z'
  },
  {
    caseId: 'CASE-008',
    patientId: 'ARCADE-ST-08',
    age: '48',
    gender: 'Female',
    medicalHistory: 'Pending angiogram analysis',
    riskFactors: 'Awaiting AI Scan Assessment',
    clinicalNotes: 'Recommended scan: data/stenosis/test/images (Diagonal branch check)',
    presetMode: 'mild_diag',
    status: 'Pending',
    createdAt: '2026-11-08T12:30:00.000Z'
  },
  {
    caseId: 'CASE-009',
    patientId: 'ARCADE-ST-09',
    age: '73',
    gender: 'Male',
    medicalHistory: 'Pending angiogram analysis',
    riskFactors: 'Awaiting AI Scan Assessment',
    clinicalNotes: 'Recommended scan: data/stenosis/test/images (Multi-vessel ischemia test)',
    presetMode: 'critical_lad',
    status: 'Pending',
    createdAt: '2026-11-09T10:40:00.000Z'
  },
  {
    caseId: 'CASE-010',
    patientId: 'ARCADE-ST-10',
    age: '59',
    gender: 'Male',
    medicalHistory: 'Pending angiogram analysis',
    riskFactors: 'Awaiting AI Scan Assessment',
    clinicalNotes: 'Recommended scan: data/stenosis/test/images (Obtuse marginal lesion)',
    presetMode: 'moderate_lcx',
    status: 'Pending',
    createdAt: '2026-11-10T15:15:00.000Z'
  },
  {
    caseId: 'CASE-011',
    patientId: 'ARCADE-ST-11',
    age: '62',
    gender: 'Female',
    medicalHistory: 'Pending angiogram analysis',
    riskFactors: 'Awaiting AI Scan Assessment',
    clinicalNotes: 'Recommended scan: data/stenosis/test/images (Ostial RCA stenosis check)',
    presetMode: 'severe_rca',
    status: 'Pending',
    createdAt: '2026-11-11T09:05:00.000Z'
  },
  {
    caseId: 'CASE-012',
    patientId: 'ARCADE-ST-12',
    age: '54',
    gender: 'Male',
    medicalHistory: 'Pending angiogram analysis',
    riskFactors: 'Awaiting AI Scan Assessment',
    clinicalNotes: 'Recommended scan: data/stenosis/test/images (Distal LAD narrowing)',
    presetMode: 'moderate_rca',
    status: 'Pending',
    createdAt: '2026-11-11T17:20:00.000Z'
  },
  {
    caseId: 'CASE-013',
    patientId: 'ARCADE-ST-13',
    age: '67',
    gender: 'Male',
    medicalHistory: 'Pending angiogram analysis',
    riskFactors: 'Awaiting AI Scan Assessment',
    clinicalNotes: 'Recommended scan: data/stenosis/test/images (High-grade focal plaque)',
    presetMode: 'severe_lmca',
    status: 'Pending',
    createdAt: '2026-11-12T11:10:00.000Z'
  },
  {
    caseId: 'CASE-014',
    patientId: 'ARCADE-ST-14',
    age: '51',
    gender: 'Female',
    medicalHistory: 'Pending angiogram analysis',
    riskFactors: 'Awaiting AI Scan Assessment',
    clinicalNotes: 'Recommended scan: data/stenosis/test/images (Subtle luminal constriction)',
    presetMode: 'mild_lad',
    status: 'Pending',
    createdAt: '2026-11-12T14:50:00.000Z'
  },
  {
    caseId: 'CASE-015',
    patientId: 'ARCADE-SYN-15',
    age: '44',
    gender: 'Male',
    medicalHistory: 'Pending angiogram analysis',
    riskFactors: 'Awaiting AI Scan Assessment',
    clinicalNotes: 'Recommended scan: data/syntax/test/images (Full left coronary tree)',
    presetMode: 'normal_clean',
    status: 'Pending',
    createdAt: '2026-11-13T08:30:00.000Z'
  },
  {
    caseId: 'CASE-016',
    patientId: 'ARCADE-SYN-16',
    age: '39',
    gender: 'Female',
    medicalHistory: 'Pending angiogram analysis',
    riskFactors: 'Awaiting AI Scan Assessment',
    clinicalNotes: 'Recommended scan: data/syntax/test/images (Right dominant artery tree)',
    presetMode: 'normal_rca',
    status: 'Pending',
    createdAt: '2026-11-13T12:00:00.000Z'
  },
  {
    caseId: 'CASE-017',
    patientId: 'ARCADE-SYN-17',
    age: '46',
    gender: 'Male',
    medicalHistory: 'Pending angiogram analysis',
    riskFactors: 'Awaiting AI Scan Assessment',
    clinicalNotes: 'Recommended scan: data/syntax/test/images (Bifurcation & tortuosity map)',
    presetMode: 'normal_clean',
    status: 'Pending',
    createdAt: '2026-11-13T16:45:00.000Z'
  },
  {
    caseId: 'CASE-018',
    patientId: 'ARCADE-SYN-18',
    age: '53',
    gender: 'Female',
    medicalHistory: 'Pending angiogram analysis',
    riskFactors: 'Awaiting AI Scan Assessment',
    clinicalNotes: 'Recommended scan: data/syntax/test/images (Distal branch segmentation)',
    presetMode: 'mild_diag',
    status: 'Pending',
    createdAt: '2026-11-14T09:15:00.000Z'
  },
  {
    caseId: 'CASE-019',
    patientId: 'ARCADE-SYN-19',
    age: '42',
    gender: 'Male',
    medicalHistory: 'Pending angiogram analysis',
    riskFactors: 'Awaiting AI Scan Assessment',
    clinicalNotes: 'Recommended scan: data/syntax/test/images (Patent lumen baseline)',
    presetMode: 'normal_rca',
    status: 'Pending',
    createdAt: '2026-11-14T10:30:00.000Z'
  },
  {
    caseId: 'CASE-020',
    patientId: 'ARCADE-SYN-20',
    age: '57',
    gender: 'Female',
    medicalHistory: 'Pending angiogram analysis',
    riskFactors: 'Awaiting AI Scan Assessment',
    clinicalNotes: 'Recommended scan: data/syntax/test/images (Complete vascular morphometry)',
    presetMode: 'normal_clean',
    status: 'Pending',
    createdAt: '2026-11-14T11:40:00.000Z'
  }
]

function readSavedCases() {
  try {
    const saved = localStorage.getItem('cases')

    if (saved !== null) {
      const parsed = JSON.parse(saved)
      return Array.isArray(parsed) ? parsed : demoCases
    }

    localStorage.setItem('cases', JSON.stringify(demoCases))
    return demoCases
  } catch {
    return demoCases
  }
}

function Cases() {
  const navigate = useNavigate()

  const [showForm, setShowForm] = useState(false)
  const [cases, setCases] = useState(readSavedCases)
  const [savedAnalyses, setSavedAnalyses] = useState({})
  const [selectedCase, setSelectedCase] = useState(null)
  const [searchTerm, setSearchTerm] = useState('')
  const [activeFilter, setActiveFilter] = useState('All')
  const [formError, setFormError] = useState('')
  const [successMessage, setSuccessMessage] = useState('')
  const [caseData, setCaseData] = useState(initialCaseData)

  useEffect(() => {
    localStorage.setItem('cases', JSON.stringify(cases))
  }, [cases])

  // Load all saved scans from IndexedDB whenever returning to list
  useEffect(() => {
    getAllCaseAnalyses()
      .then(setSavedAnalyses)
      .catch(() => setSavedAnalyses({}))
  }, [selectedCase])

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

  const handleSubmit = event => {
    event.preventDefault()
    setFormError('')
    setSuccessMessage('')

    if (!caseData.patientId.trim()) {
      setFormError('Please enter a Patient ID.')
      return
    }

    if (
      caseData.age &&
      (Number(caseData.age) < 1 || Number(caseData.age) > 120)
    ) {
      setFormError('Age must be between 1 and 120.')
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

    const newCase = {
      ...caseData,
      patientId: caseData.patientId.trim(),
      medicalHistory: caseData.medicalHistory.trim() || 'Pending angiogram analysis',
      riskFactors: caseData.riskFactors.trim() || 'Awaiting AI Scan Assessment',
      caseId: `CASE-${Date.now()}`,
      status: 'Pending',
      createdAt: new Date().toISOString()
    }

    setCases(previous => [newCase, ...previous])
    setCaseData({ ...initialCaseData })
    setShowForm(false)
    setActiveFilter('All')
    setSearchTerm('')
    setSuccessMessage('Patient case created. Upload scan to auto-extract clinical profile.')
  }

  const openCaseDetails = item => {
    setSelectedCase(item)
    setShowForm(false)
  }

  const closeCaseDetails = () => {
    setSelectedCase(null)
    setCases(readSavedCases())
    getAllCaseAnalyses()
      .then(setSavedAnalyses)
      .catch(() => {})
  }

  const deleteCase = async (event, caseId) => {
    event.stopPropagation()

    const confirmed = window.confirm(
      'Are you sure you want to delete this case?'
    )

    if (!confirmed) return

    setCases(previous =>
      previous.filter(item => item.caseId !== caseId)
    )

    await deleteCaseAnalysis(caseId)
    setSavedAnalyses(prev => {
      const copy = { ...prev }
      delete copy[caseId]
      return copy
    })

    setSuccessMessage('Case deleted successfully.')
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
            Upload ARCADE or clinical scans to automatically derive patient stenosis risk & clinical profile.
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

      {!showForm && (
        <>
          {/* SUMMARY */}
          <div className="case-summary-grid">
            <div className="case-summary-card">
              <div className="case-summary-icon">
                <Users size={20} />
              </div>
              <span>Total Patients</span>
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
              <small>Scan & profile derived</small>
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
          {filteredCases.length > 0 ? (
            <div className="case-list cases-redesign-list">
              {filteredCases.map((item, index) => {
                const record = savedAnalyses[item.caseId]
                const hasSavedScan = Boolean(record?.analysis)
                const derivedProfile = record?.analysis?.clinical_profile?.inferred_risk_profile
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
                        <h3>{item.patientId || 'Unknown Patient'}</h3>
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
                        <span>AI-Derived Risk Profile (From Scan)</span>
                        <strong style={{ color: hasSavedScan ? '#e2e8f0' : '#64748b', fontSize: '12px' }}>
                          {derivedProfile || item.riskFactors || 'Awaiting AI Scan Assessment'}
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
                        <button
                          type="button"
                          className="case-delete-button"
                          onClick={event => deleteCase(event, item.caseId)}
                          aria-label="Delete case"
                        >
                          <Trash2 size={15} />
                        </button>

                        <span className="case-view-link">
                          {hasSavedScan ? 'View Saved Report →' : 'Upload & Analyze →'}
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
              <h2 style={{ margin: '4px 0' }}>Register Patient (Demographics Only)</h2>
              <p style={{ margin: 0, fontSize: '13px', color: '#94a3b8' }}>
                Risk profile & clinical impression will be automatically derived after uploading the coronary scan.
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
                <label htmlFor="age">Age</label>
                <input
                  id="age"
                  type="number"
                  name="age"
                  placeholder="Enter age"
                  min="1"
                  max="120"
                  value={caseData.age}
                  onChange={handleChange}
                />
              </div>

              <div className="form-group">
                <label htmlFor="gender">Gender</label>
                <select
                  id="gender"
                  name="gender"
                  value={caseData.gender}
                  onChange={handleChange}
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