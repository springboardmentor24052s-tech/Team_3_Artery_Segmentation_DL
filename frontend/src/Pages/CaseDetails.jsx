import { useEffect, useState } from 'react'

function CaseDetails({ caseData, onBack }) {
  const [selectedFile, setSelectedFile] = useState(null)
  const [imagePreview, setImagePreview] = useState(null)
  const [message, setMessage] = useState('')
  const [messageType, setMessageType] = useState('')
  const [isProcessing, setIsProcessing] = useState(false)
  const [showResult, setShowResult] = useState(false)

  const MAX_FILE_SIZE = 10 * 1024 * 1024

  useEffect(() => {
    return () => {
      if (imagePreview) {
        URL.revokeObjectURL(imagePreview)
      }
    }
  }, [imagePreview])

  const handleFileChange = (e) => {
    const file = e.target.files[0]

    setMessage('')
    setMessageType('')
    setShowResult(false)

    if (!file) {
      return
    }

    if (!file.type.startsWith('image/')) {
      setSelectedFile(null)
      setImagePreview(null)
      setMessage('Please select a valid image file.')
      setMessageType('error')
      return
    }

    if (file.size > MAX_FILE_SIZE) {
      setSelectedFile(null)
      setImagePreview(null)
      setMessage('Image size should be less than 10 MB.')
      setMessageType('error')
      return
    }

    setSelectedFile(file)
    setImagePreview(URL.createObjectURL(file))
  }

  const handleRemoveImage = () => {
    setSelectedFile(null)
    setImagePreview(null)
    setMessage('')
    setMessageType('')
    setShowResult(false)
  }

  const handleUpload = () => {
    if (!selectedFile) {
      setMessage('Please select an image first.')
      setMessageType('error')
      return
    }

    setIsProcessing(true)
    setShowResult(false)
    setMessage('')
    setMessageType('')

    // Temporary frontend simulation.
    // This will later be replaced with the FastAPI API call.
    setTimeout(() => {
      setIsProcessing(false)
      setShowResult(true)
      setMessage('Image processing completed successfully.')
      setMessageType('success')
    }, 3000)
  }

  return (
    <div className='case-details-page'>

      <div className='case-details-header'>
        <div>
          <h1>Case Details</h1>
          <p>
            View case information and upload a medical image.
          </p>
        </div>

        <button
          type='button'
          onClick={onBack}
        >
          Back to Cases
        </button>
      </div>

      <div className='case-info-card'>

        <h2>Patient Information</h2>

        <div className='case-info-grid'>

          <div>
            <span>Patient ID</span>
            <strong>{caseData.patientId}</strong>
          </div>

          <div>
            <span>Age</span>
            <strong>
              {caseData.age || 'Not provided'}
            </strong>
          </div>

          <div>
            <span>Gender</span>
            <strong>
              {caseData.gender || 'Not provided'}
            </strong>
          </div>

          <div>
            <span>Risk Factors</span>
            <strong>
              {caseData.riskFactors || 'None'}
            </strong>
          </div>

        </div>

        <div className='case-info-section'>
          <span>Medical History</span>
          <p>
            {caseData.medicalHistory || 'Not provided'}
          </p>
        </div>

        <div className='case-info-section'>
          <span>Clinical Notes</span>
          <p>
            {caseData.clinicalNotes || 'Not provided'}
          </p>
        </div>

      </div>

      <div className='upload-card'>

        <h2>Medical Image</h2>

        <p>
          Upload a medical image for processing and analysis.
        </p>

        <div className='upload-area'>

          {!selectedFile && (
            <>
              <p>Select a medical image to upload.</p>

              <input
                type='file'
                accept='image/*'
                onChange={handleFileChange}
                disabled={isProcessing}
              />
            </>
          )}

          {selectedFile && (
            <div className='selected-file-section'>

              <p>
                Selected file:
              </p>

              <strong>
                {selectedFile.name}
              </strong>

              <p>
                Size:{' '}
                {(selectedFile.size / (1024 * 1024)).toFixed(2)}
                {' '}MB
              </p>

            </div>
          )}

          {imagePreview && (
            <div className='image-preview'>

              <p>Image Preview</p>

              <img
                src={imagePreview}
                alt='Selected medical image'
              />

            </div>
          )}

          {selectedFile && !isProcessing && (
            <div className='upload-actions'>

              <button
                type='button'
                onClick={handleUpload}
              >
                Upload Image
              </button>

              <button
                type='button'
                className='cancel-button'
                onClick={handleRemoveImage}
              >
                Remove Image
              </button>

            </div>
          )}

          {isProcessing && (
            <div className='processing-status'>

              <div className='loader'></div>

              <p>
                Processing medical image...
              </p>

              <span>
                Please wait while the image is being processed.
              </span>

            </div>
          )}

          {message && (
            <p className={`upload-message ${messageType}`}>
              {message}
            </p>
          )}

          {showResult && (
            <div className='result-card'>

              <h2>Analysis Result</h2>

              <div className='result-item'>
                <span>Status</span>
                <strong>Completed</strong>
              </div>

              <div className='result-item'>
                <span>Processed Image</span>
                <strong>{selectedFile.name}</strong>
              </div>

              <div className='result-item'>
                <span>Result</span>
                <strong>Analysis available</strong>
              </div>

            </div>
          )}

        </div>

      </div>

    </div>
  )
}

export default CaseDetails