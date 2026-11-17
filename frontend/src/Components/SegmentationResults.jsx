
import {
  Image,
  ScanLine,
  Activity,
  GitBranch,
  Ruler,
  Info,
  CircleDashed
} from 'lucide-react'

function SegmentationResults({ imagePreview }) {
  const resultSections = [
    {
      title: 'Baseline U-Net Mask',
      subtitle: 'Baseline segmentation output',
      description: 'The predicted artery mask from the baseline U-Net model will be displayed here.',
      status: 'Waiting for model output',
      icon: ScanLine
    },
    {
      title: 'Attention U-Net Mask',
      subtitle: 'Attention-based segmentation output',
      description: 'The predicted artery mask from the Attention U-Net model will be displayed here.',
      status: 'Waiting for model output',
      icon: Activity
    },
    {
      title: 'Post-processed Mask',
      subtitle: 'Refined segmentation output',
      description: 'The refined artery mask after segmentation post-processing will be displayed here.',
      status: 'Waiting for post-processing output',
      icon: Image
    },
    {
      title: 'Vessel Structure',
      subtitle: 'Extracted vessel regions',
      description: 'The extracted vessel structure and connected vessel regions will be displayed here.',
      status: 'Waiting for vessel extraction output',
      icon: GitBranch
    },
    {
      title: 'Vessel Centerline',
      subtitle: 'Centerline extraction',
      description: 'The centerline generated from the processed vessel mask will be displayed here.',
      status: 'Centerline API integration ready',
      icon: Activity
    }
  ]

  return (
    <div className="segmentation-results segmentation-redesign">

      {/* SECTION HEADER */}
      <div className="segmentation-page-heading">
        <div>
          <span className="eyebrow">ANALYSIS OUTPUT</span>
          <h2>Segmentation Results</h2>
          <p>
            Review artery segmentation outputs and vessel analysis.
          </p>
        </div>

        <span className="segmentation-preview-status">
          <CircleDashed size={14} />
          Preview
        </span>
      </div>

      {/* ORIGINAL IMAGE */}
      <section className="analysis-section segmentation-original-card">
        <div className="segmentation-section-heading">
          <div className="segmentation-icon">
            <Image size={19} />
          </div>

          <div>
            <h3>Original Medical Image</h3>
            <p>Uploaded source image</p>
          </div>
        </div>

        {imagePreview ? (
          <div className="segmentation-original-preview">
            <img
              src={imagePreview}
              alt="Original medical image"
              className="analysis-image"
            />
          </div>
        ) : (
          <div className="result-placeholder">
            Original image not available.
          </div>
        )}
      </section>

      {/* SEGMENTATION OUTPUTS */}
      <div className="segmentation-output-heading">
        <div>
          <h3>Model Outputs</h3>
          <p>Segmentation pipeline visualization</p>
        </div>
      </div>

      <div className="segmentation-output-grid">
        {resultSections.map((section) => {
          const Icon = section.icon

          return (
            <section
              className="analysis-section segmentation-output-card"
              key={section.title}
            >
              <div className="segmentation-section-heading">
                <div className="segmentation-icon">
                  <Icon size={19} />
                </div>

                <div>
                  <h3>{section.title}</h3>
                  <p>{section.subtitle}</p>
                </div>
              </div>

              <div className="result-placeholder segmentation-placeholder">
                <div className="segmentation-placeholder-icon">
                  <CircleDashed size={27} />
                </div>

                <strong>{section.title}</strong>

                <p>{section.description}</p>

                <span className="segmentation-pending-label">
                  {section.status}
                </span>
              </div>
            </section>
          )
        })}
      </div>

      {/* VESSEL MEASUREMENTS */}
      <section className="vessel-measurements segmentation-measurements">
        <div className="segmentation-section-heading">
          <div className="segmentation-icon">
            <Ruler size={19} />
          </div>

          <div>
            <h3>Vessel Measurements</h3>
            <p>Quantitative vessel analysis</p>
          </div>
        </div>

        <div className="segmentation-measurement-grid">
          <div className="measurement-item">
            <span>Estimated Vessel Diameter</span>
            <strong>Pending backend analysis</strong>
          </div>

          <div className="measurement-item">
            <span>Measurement Unit</span>
            <strong>Pixels / physical unit after calibration</strong>
          </div>

          <div className="measurement-item">
            <span>Measurement Status</span>
            <strong>Waiting for vessel analysis</strong>
          </div>
        </div>
      </section>

      {/* DISCLAIMER */}
      <div className="segmentation-disclaimer">
        <Info size={17} />
        <p>
          Segmentation outputs are currently awaiting model integration.
          No clinical measurements or predictions are generated by these
          preview placeholders.
        </p>
      </div>

    </div>
  )
}

export default SegmentationResults
