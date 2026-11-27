import { AlertTriangle, Activity } from 'lucide-react'

import './ModelAnalytics.css'

export default function ModelAnalytics() {
  return (
    <div className="analytics-page">
      <div className="analytics-heading">
        <div>
          <span className="analytics-label">MODEL STATUS</span>
          <h1>Model Analytics</h1>
          <p>Performance metrics are not reported without a verified evaluation record.</p>
        </div>
        <span className="analytics-status">Research</span>
      </div>

      <div className="analytics-model-card">
        <div>
          <span>AVAILABLE CHECKPOINT</span>
          <h2>FrozenFineTuned MultiScaleResidualUNet</h2>
          <p>Stenosis detection and localization only; not an artery segmentation model.</p>
        </div>
        <Activity size={32} color="#58a6e7" />
      </div>

      <section className="analytics-chart-card radar-card">
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', color: '#fbbf24' }}>
          <AlertTriangle size={20} />
          <h3 style={{ margin: 0 }}>Unverified evaluation metrics omitted</h3>
        </div>
        <p>
          The current application does not load a persisted evaluation report for this view. Hardcoded Dice, IoU, accuracy, model comparisons, and sample mask images are intentionally not presented as live results.
        </p>
        <p>
          Baseline U-Net artery segmentation is available in its separate workflow. Segmentation-derived centerlines and measurements are not currently provided.
        </p>
      </section>
    </div>
  )
}
