
import { useEffect, useState } from 'react'
import {
  Activity,
  LoaderCircle,
  Info,
  UploadCloud,
  Image as ImageIcon
} from 'lucide-react'

import { analyzeVesselWidth } from '../Services/stenosisApi'

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'

const models = [
  {
    name: 'Baseline Reference',
    mask: 'baseline_mask.png',
    centerline: 'baseline_centerline.png'
  },
  {
    name: 'Attention Reference',
    mask: 'attention_mask.png',
    centerline: 'attention_centerline.png'
  }
]

function ResultMetrics({ result }) {
  if (!result) return null

  const metrics = [
    ['Minimum width', `${result.minimum_width_px} px`],
    ['Median width', `${result.median_width_px} px`],
    ['Maximum width', `${result.maximum_width_px} px`],
    ['Centerline pixels', result.centerline_pixels]
  ]

  return (
    <div className="stenosis-width-results">
      <h4>Experimental Measurements</h4>

      <div className="stenosis-image-box">
        <h4>Vessel Width Heatmap</h4>
        <img
          src={result.width_heatmap}
          alt="Calculated vessel width heatmap"
        />
      </div>

      <div className="stenosis-metrics">
        {metrics.map(([label, value]) => (
          <div key={label}>
            <span>{label}</span>
            <strong>{value}</strong>
          </div>
        ))}
      </div>

      <p className="stenosis-result-note">
        {result.clinical_status}
      </p>
    </div>
  )
}

export default function StenosisVisualization() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [preview, setPreview] = useState('')
  const [uploadedResult, setUploadedResult] = useState(null)
  const [results, setResults] = useState({})
  const [loading, setLoading] = useState('')
  const [error, setError] = useState('')
  const [success, setSuccess] = useState('')

  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview)
    }
  }, [preview])

  const handleFileChange = (event) => {
    const file = event.target.files?.[0]

    setError('')
    setSuccess('')
    setUploadedResult(null)

    if (!file) return

    if (!file.type.startsWith('image/')) {
      setError('Please select a valid image file.')
      return
    }

    if (file.size > 10 * 1024 * 1024) {
      setError('File size must be below 10 MB.')
      return
    }

    setSelectedFile(file)
    setPreview(URL.createObjectURL(file))
  }

  const analyzeFile = async (file, key) => {
    setLoading(key)
    setError('')
    setSuccess('')

    try {
      const data = await analyzeVesselWidth(file)

      if (!data.width_heatmap) {
        throw new Error('Analysis did not return a heatmap.')
      }

      if (key === 'uploaded-mask') {
        setUploadedResult(data)
      } else {
        setResults(previous => ({
          ...previous,
          [key]: data
        }))
      }

      setSuccess('Vessel width analysis completed successfully.')
    } catch (err) {
      setError(err.message || 'Analysis failed.')
    } finally {
      setLoading('')
    }
  }

  const handleReferenceAnalysis = async (model) => {
    try {
      setLoading(model.name)
      setError('')
      setSuccess('')

      const response = await fetch(
        `${API_BASE_URL}/test-files/${model.mask}`
      )

      if (!response.ok) {
        throw new Error(`Unable to load ${model.mask}`)
      }

      const blob = await response.blob()

      const maskFile = new File(
        [blob],
        model.mask,
        { type: 'image/png' }
      )

      const data = await analyzeVesselWidth(maskFile)

      setResults(previous => ({
        ...previous,
        [model.name]: data
      }))

      setSuccess(`${model.name} analysis completed.`)
    } catch (err) {
      setError(err.message || 'Reference analysis failed.')
    } finally {
      setLoading('')
    }
  }

  return (
    <section className="stenosis-visualization">

      <div className="stenosis-heading">
        <span className="eyebrow">TASK 8 · VESSEL ANALYSIS</span>
        <h2>Stenosis Visualization</h2>
        <p>
          Upload a binary vessel segmentation mask or analyze
          the available reference masks.
        </p>
      </div>

      <div className="stenosis-notice">
        Width measurements are experimental geometric measurements.
        They are not clinically validated stenosis diagnoses.
        Original angiography images must first undergo vessel
        segmentation before this width analysis.
      </div>

      {error && (
        <div className="details-message error" role="alert">
          <Info size={17} />
          <span>{error}</span>
        </div>
      )}

      {success && (
        <div className="details-message success" role="status">
          {success}
        </div>
      )}

      {/* Upload binary segmentation mask */}
      <article className="stenosis-model-card">
        <h3>Analyze Your Segmentation Mask</h3>

        <label className="stenosis-upload-area">
          <UploadCloud size={30} />

          <strong>Select binary vessel mask</strong>

          <span>
            PNG, JPG or JPEG · Maximum 10 MB
          </span>

          <input
            type="file"
            accept="image/png,image/jpeg"
            onChange={handleFileChange}
            hidden
          />
        </label>

        {preview && (
          <div className="stenosis-image-box">
            <h4>
              <ImageIcon size={16} />
              Selected Mask
            </h4>

            <img src={preview} alt="Selected binary segmentation mask" />

            <p>{selectedFile?.name}</p>
          </div>
        )}

        <button
          type="button"
          className="details-analyze-button"
          disabled={!selectedFile || loading !== ''}
          onClick={() => analyzeFile(selectedFile, 'uploaded-mask')}
        >
          {loading === 'uploaded-mask' ? (
            <>
              <LoaderCircle size={17} />
              Analyzing...
            </>
          ) : (
            <>
              <Activity size={17} />
              Analyze Uploaded Mask
            </>
          )}
        </button>

        <ResultMetrics result={uploadedResult} />
      </article>

      {/* Reference mask comparison */}
      <div className="section-heading">
        <div>
          <h2>Reference Model Comparison</h2>
          <p>
            Analyze the existing reference masks separately.
          </p>
        </div>
      </div>

      <div className="stenosis-model-grid">
        {models.map(model => {
          const result = results[model.name]

          return (
            <article
              className="stenosis-model-card"
              key={model.name}
            >
              <h3>{model.name}</h3>

              <div className="stenosis-image-grid">
                <div className="stenosis-image-box">
                  <h4>Segmentation Mask</h4>

                  <img
                    src={`${API_BASE_URL}/test-files/${model.mask}`}
                    alt={`${model.name} segmentation mask`}
                  />
                </div>

                <div className="stenosis-image-box">
                  <h4>Centerline</h4>

                  <img
                    src={`${API_BASE_URL}/test-files/${model.centerline}`}
                    alt={`${model.name} centerline`}
                  />
                </div>
              </div>

              <button
                type="button"
                className="details-analyze-button"
                disabled={loading !== ''}
                onClick={() => handleReferenceAnalysis(model)}
              >
                {loading === model.name ? (
                  <>
                    <LoaderCircle size={17} />
                    Calculating...
                  </>
                ) : (
                  <>
                    <Activity size={17} />
                    Analyze Vessel Width
                  </>
                )}
              </button>

              <ResultMetrics result={result} />
            </article>
          )
        })}
      </div>
    </section>
  )
}
