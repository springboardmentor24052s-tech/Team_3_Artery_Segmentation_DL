import { useEffect, useState } from 'react'
import { Activity, Image as ImageIcon, LoaderCircle, UploadCloud } from 'lucide-react'

import { apiGet, apiPost, fetchImageUrl } from '../Services/api'
import './Segmentation.css'

const SEGMENT_LABEL = 'baseline_unet_artery_mask'
const ARTIFACT_TYPES = [
  'artery_input_preview',
  'artery_segmentation_mask',
  'artery_probability_map',
  'artery_segmentation_overlay'
]

export default function Segmentation() {
  const [cases, setCases] = useState([])
  const [caseId, setCaseId] = useState('')
  const [images, setImages] = useState([])
  const [existingImageId, setExistingImageId] = useState('')
  const [selectedFile, setSelectedFile] = useState(null)
  const [runs, setRuns] = useState([])
  const [activeRunId, setActiveRunId] = useState('')
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)
  const [processing, setProcessing] = useState(false)

  useEffect(() => {
    let active = true
    async function loadCases() {
      try {
        const caseRecords = await apiGet('/cases')
        if (active) {
          setCases(caseRecords)
          setCaseId(current => current || caseRecords[0]?.id || '')
        }
      } catch (loadError) {
        if (active) setError(loadError.message)
      } finally {
        if (active) setLoading(false)
      }
    }
    loadCases()
    return () => { active = false }
  }, [])

  useEffect(() => {
    let active = true
    async function loadCaseData() {
      setError('')
      setImages([])
      setRuns([])
      setSelectedFile(null)
      setExistingImageId('')
      setActiveRunId('')
      setResult(null)
      if (!caseId) return

      try {
        const [caseImages, caseRuns] = await Promise.all([
          apiGet(`/images?case_id=${encodeURIComponent(caseId)}`),
          apiGet(`/cases/${encodeURIComponent(caseId)}/analysis-runs`)
        ])
        if (!active) return
        setImages(caseImages)
        const savedRuns = caseRuns.filter(run =>
          run.status === 'completed' &&
          (run.segments || []).some(segment => segment.label === SEGMENT_LABEL)
        )
        setRuns(savedRuns)
        setActiveRunId(savedRuns[0]?.id || '')
      } catch (loadError) {
        if (active) setError(loadError.message)
      }
    }
    loadCaseData()
    return () => { active = false }
  }, [caseId])

  useEffect(() => {
    let active = true
    const objectUrls = []
    const selectedRun = runs.find(run => run.id === activeRunId)
    if (!selectedRun) {
      return () => { active = false }
    }

    async function loadResult() {
      try {
        const imageRecord = await apiGet(
          `/images/${encodeURIComponent(selectedRun.image_id)}`
        )
        const artifacts = new Set(
          (selectedRun.artifacts || []).map(artifact => artifact.artifact_type)
        )
        const imagesByType = {}
        for (const artifactType of ARTIFACT_TYPES) {
          if (!artifacts.has(artifactType)) continue
          const url = await fetchImageUrl(
            `/analysis-runs/${encodeURIComponent(selectedRun.id)}/artifacts/${encodeURIComponent(artifactType)}`
          )
          objectUrls.push(url)
          imagesByType[artifactType] = url
        }

        if (!imagesByType.artery_segmentation_mask ||
            !imagesByType.artery_segmentation_overlay) {
          throw new Error('Saved segmentation images are incomplete.')
        }

        const originalUrl = imageRecord.file_format === 'dcm'
          ? imagesByType.artery_input_preview
          : await fetchImageUrl(
              `/images/${encodeURIComponent(selectedRun.image_id)}/content`
            )
        if (originalUrl) objectUrls.push(originalUrl)

        if (active) {
          setResult({
            run: selectedRun,
            images: imagesByType,
            originalUrl
          })
        }
      } catch (loadError) {
        objectUrls.forEach(URL.revokeObjectURL)
        if (active) {
          setResult(null)
          setError(loadError.message)
        }
      }
    }
    loadResult()
    return () => {
      active = false
      objectUrls.forEach(URL.revokeObjectURL)
    }
  }, [activeRunId, runs])

  const runSegmentation = async event => {
    event.preventDefault()
    if (!caseId || (!selectedFile && !existingImageId)) return

    setError('')
    setProcessing(true)
    setResult(null)
    try {
      let imageId = existingImageId
      if (selectedFile) {
        const formData = new FormData()
        formData.append('file', selectedFile)
        formData.append('case_id', caseId)
        const uploaded = await apiPost('/images', formData)
        imageId = uploaded.id
        setImages(current => current.some(item => item.id === imageId)
          ? current
          : [uploaded, ...current])
        setExistingImageId(imageId)
        setSelectedFile(null)
      }

      const prediction = await apiPost(
        `/images/${encodeURIComponent(imageId)}/segment`,
        {}
      )
      const run = await apiGet(
        `/analysis-runs/${encodeURIComponent(prediction.analysis_run_id)}`
      )
      setRuns(current => [run, ...current.filter(item => item.id !== run.id)])
      setActiveRunId(run.id)
      setImages(current => current.some(item => item.id === imageId)
        ? current
        : [{ id: imageId, original_filename: selectedFile?.name || imageId }, ...current])
      setSelectedFile(null)
      setExistingImageId('')
    } catch (runError) {
      setError(runError.message)
    } finally {
      setProcessing(false)
    }
  }

  const selectedRun = runs.find(run => run.id === activeRunId)
  const report = selectedRun?.reports?.[0]

  return (
    <div className="segmentation-workspace">
      <div className="page-heading">
        <div>
          <span className="eyebrow">BASELINE U-NET · CASE-SCOPED</span>
          <h1>Artery Segmentation</h1>
          <p>Run the selected Baseline U-Net checkpoint and review saved mask outputs.</p>
        </div>
      </div>

      {error && <div className="login-error" role="alert">{error}</div>}

      <section className="segmentation-control-card">
        <div className="segmentation-card-heading">
          <Activity size={19} />
          <div>
            <h2>Run artery segmentation</h2>
            <p>Baseline U-Net · round 2 best checkpoint · threshold 0.50</p>
          </div>
        </div>
        <form className="segmentation-controls" onSubmit={runSegmentation}>
          <label>
            <span>Authorized case</span>
            <select
              value={caseId}
              onChange={event => setCaseId(event.target.value)}
              disabled={loading || cases.length === 0 || processing}
              required
            >
              <option value="">Select a case</option>
              {cases.map(item => (
                <option key={item.id} value={item.id}>
                  {item.id.slice(0, 8)} · {item.anatomical_region}
                </option>
              ))}
            </select>
          </label>

          <label>
            <span>Previously uploaded image (optional)</span>
            <select
              value={existingImageId}
              onChange={event => {
                setExistingImageId(event.target.value)
                setSelectedFile(null)
              }}
              disabled={!caseId || images.length === 0 || processing}
            >
              <option value="">Upload a new image instead</option>
              {images.map(image => (
                <option key={image.id} value={image.id}>
                  {image.original_filename || image.id}
                </option>
              ))}
            </select>
          </label>

          <label className="segmentation-file-control">
            <span>Upload image</span>
            <input
              type="file"
              accept=".png,.jpg,.jpeg,.dcm"
              onChange={event => {
                setSelectedFile(event.target.files?.[0] || null)
                if (event.target.files?.[0]) setExistingImageId('')
              }}
              disabled={!caseId || processing}
            />
          </label>

          <button
            className="overview-primary-btn"
            type="submit"
            disabled={!caseId || processing || (!selectedFile && !existingImageId)}
          >
            {processing
              ? <><LoaderCircle size={16} className="segmentation-spinner" /> Running model…</>
              : <><UploadCloud size={16} /> Upload and segment</>}
          </button>
        </form>
        {!loading && cases.length === 0 && (
          <p className="segmentation-empty">Create a patient and case before uploading an image.</p>
        )}
      </section>

      <section className="segmentation-results-card">
        <div className="segmentation-results-heading">
          <div>
            <h2>Saved segmentation results</h2>
            <p>Choose an earlier Baseline U-Net run for this case.</p>
          </div>
          <label>
            <span className="visually-hidden">Saved segmentation run</span>
            <select
              value={activeRunId}
              onChange={event => setActiveRunId(event.target.value)}
              disabled={runs.length === 0}
            >
              {runs.length === 0 && <option value="">No saved runs</option>}
              {runs.map(run => (
                <option key={run.id} value={run.id}>
                  {run.started_at
                    ? new Date(run.started_at).toLocaleString()
                    : run.id.slice(0, 8)}
                </option>
              ))}
            </select>
          </label>
        </div>

        {result ? (
          <>
            <div className="segmentation-metadata">
              <span>Model: Baseline U-Net · round2-best</span>
              <span>Threshold: 0.50</span>
              {report?.summary && <span>{report.summary}</span>}
            </div>
            <div className="segmentation-image-grid">
              <figure>
                <figcaption>Original image</figcaption>
                {result.originalUrl
                  ? <img src={result.originalUrl} alt="Original uploaded angiogram" />
                  : <div className="segmentation-image-missing">Original preview unavailable</div>}
              </figure>
              <figure>
                <figcaption>Predicted artery mask</figcaption>
                <img
                  src={result.images.artery_segmentation_mask}
                  alt="Baseline U-Net binary artery segmentation mask"
                />
              </figure>
              <figure>
                <figcaption>Segmentation overlay</figcaption>
                <img
                  src={result.images.artery_segmentation_overlay}
                  alt="Artery segmentation overlay on the input image"
                />
              </figure>
              <figure>
                <figcaption>Probability map</figcaption>
                {result.images.artery_probability_map
                  ? <img src={result.images.artery_probability_map} alt="Baseline U-Net pixel probability map" />
                  : <div className="segmentation-image-missing">Probability map unavailable</div>}
              </figure>
            </div>
          </>
        ) : (
          <div className="segmentation-no-results">
            <ImageIcon size={28} />
            <p>{runs.length ? 'Select a saved result to view its images.' : 'No saved Baseline U-Net results for this case yet.'}</p>
          </div>
        )}
      </section>
      <p className="segmentation-disclaimer">
        Model outputs are research results, not a clinical diagnosis. Dice and other
        ground-truth metrics are not calculated for individual uploads.
      </p>
    </div>
  )
}
