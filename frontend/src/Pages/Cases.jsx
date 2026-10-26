import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import CaseDetails from './CaseDetails'

function Cases() {
  const navigate = useNavigate()

  const [showForm, setShowForm] = useState(false)

  const [cases, setCases] = useState(() => {
    const savedCases = localStorage.getItem('cases')
    return savedCases ? JSON.parse(savedCases) : []
  })

  const [selectedCase, setSelectedCase] = useState(null)

  const [caseData, setCaseData] = useState({
    patientId: '',
    age: '',
    gender: '',
    medicalHistory: '',
    riskFactors: '',
    clinicalNotes: ''
  })

  // Save cases in localStorage whenever cases change
  useEffect(() => {
    localStorage.setItem('cases', JSON.stringify(cases))
  }, [cases])

  const openCaseForm = () => {
    setShowForm(true)
  }

  const closeCaseForm = () => {
    setShowForm(false)
  }

  const handleChange = e => {
    const { name, value } = e.target

    setCaseData(prev => ({
      ...prev,
      [name]: value
    }))
  }

  const handleSubmit = e => {
    e.preventDefault()

    const newCase = {
      ...caseData,
      status: 'Pending'
    }

    setCases(prev => [...prev, newCase])

    setShowForm(false)

    setCaseData({
      patientId: '',
      age: '',
      gender: '',
      medicalHistory: '',
      riskFactors: '',
      clinicalNotes: ''
    })

    alert('Case created successfully!')
  }

  const openCaseDetails = item => {
    setSelectedCase(item)
  }

  const closeCaseDetails = () => {
    setSelectedCase(null)
  }

  // Show Case Details
  if (selectedCase) {
    return (
      <CaseDetails
        caseData={selectedCase}
        onBack={closeCaseDetails}
      />
    )
  }

  return (
    <div className='cases-page'>

      {/* HEADER */}
      <div className='cases-header'>

        <div>
          <h1>Cases</h1>
          <p>Manage your medical image cases.</p>
        </div>

        <div>

          <button
            type='button'
            onClick={() => navigate('/dashboard')}
            className='back-dashboard-button'
          >
            Back to Dashboard
          </button>

          <button
            type='button'
            onClick={openCaseForm}
          >
            + Create New Case
          </button>

        </div>

      </div>


      {/* NO CASES */}
      {cases.length === 0 && !showForm && (
        <div className='empty-cases'>

          <h2>No cases available</h2>

          <p>
            Create a new case to upload a medical image and start the
            processing workflow.
          </p>

          <button
            type='button'
            onClick={openCaseForm}
          >
            Create New Case
          </button>

        </div>
      )}


      {/* CASE LIST */}
      {cases.length > 0 && !showForm && (
        <div className='case-list'>

          {cases.map((item, index) => (
            <div
              className='case-card'
              key={index}
              onClick={() => openCaseDetails(item)}
            >

              <h2>Case {index + 1}</h2>

              <p>
                <strong>Patient ID:</strong>{' '}
                {item.patientId}
              </p>

              <p>
                <strong>Age:</strong>{' '}
                {item.age || 'Not provided'}
              </p>

              <p>
                <strong>Gender:</strong>{' '}
                {item.gender || 'Not provided'}
              </p>

              <p>
                <strong>Risk Factors:</strong>{' '}
                {item.riskFactors || 'None'}
              </p>

              <p>
                <strong>Medical History:</strong>{' '}
                {item.medicalHistory || 'Not provided'}
              </p>

              <p>
                <strong>Clinical Notes:</strong>{' '}
                {item.clinicalNotes || 'Not provided'}
              </p>

              <p>
                <strong>Status:</strong>{' '}

                <span className='case-status'>
                  {item.status}
                </span>
              </p>

              <p className='view-case-text'>
                Click to view case details →
              </p>

            </div>
          ))}

        </div>
      )}


      {/* CREATE CASE FORM */}
      {showForm && (
        <div className='case-form-card'>

          <div className='form-header'>

            <div>
              <h2>Create New Case</h2>

              <p>
                Enter the basic case information.
              </p>
            </div>

            <button
              type='button'
              className='cancel-button'
              onClick={closeCaseForm}
            >
              Cancel
            </button>

          </div>


          <form onSubmit={handleSubmit}>

            {/* PATIENT ID + AGE */}
            <div className='form-grid'>

              <div className='form-group'>

                <label>
                  Patient ID
                </label>

                <input
                  type='text'
                  name='patientId'
                  placeholder='Enter patient ID'
                  value={caseData.patientId}
                  onChange={handleChange}
                  required
                />

              </div>


              <div className='form-group'>

                <label>
                  Age
                </label>

                <input
                  type='number'
                  name='age'
                  placeholder='Enter age'
                  value={caseData.age}
                  onChange={handleChange}
                  min='1'
                  max='120'
                />

              </div>

            </div>


            {/* GENDER + RISK FACTORS */}
            <div className='form-grid'>

              <div className='form-group'>

                <label>
                  Gender
                </label>

                <select
                  name='gender'
                  value={caseData.gender}
                  onChange={handleChange}
                >

                  <option value=''>
                    Select gender
                  </option>

                  <option value='Male'>
                    Male
                  </option>

                  <option value='Female'>
                    Female
                  </option>

                  <option value='Other'>
                    Other
                  </option>

                </select>

              </div>


              <div className='form-group'>

                <label>
                  Risk Factors
                </label>

                <input
                  type='text'
                  name='riskFactors'
                  placeholder='Enter risk factors'
                  value={caseData.riskFactors}
                  onChange={handleChange}
                />

              </div>

            </div>


            {/* MEDICAL HISTORY */}
            <div className='form-group'>

              <label>
                Medical History
              </label>

              <textarea
                name='medicalHistory'
                placeholder='Enter relevant medical history'
                value={caseData.medicalHistory}
                onChange={handleChange}
                rows='4'
              />

            </div>


            {/* CLINICAL NOTES */}
            <div className='form-group'>

              <label>
                Clinical Notes
              </label>

              <textarea
                name='clinicalNotes'
                placeholder='Enter clinical notes'
                value={caseData.clinicalNotes}
                onChange={handleChange}
                rows='4'
              />

            </div>


            {/* FORM BUTTONS */}
            <div className='form-actions'>

              <button
                type='button'
                className='cancel-button'
                onClick={closeCaseForm}
              >
                Cancel
              </button>

              <button type='submit'>
                Create Case
              </button>

            </div>

          </form>

        </div>
      )}

    </div>
  )
}

export default Cases