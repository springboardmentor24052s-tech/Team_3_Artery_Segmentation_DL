import { Trophy, Target, Activity, Ruler, Layers, Info, Download } from 'lucide-react'

const summaryMetrics = [
  {
    label: 'Pixel Accuracy',
    value: '0.963578',
    winner: 'Attention U-Net',
    icon: Target,
    accent: '#38bdf8'
  },
  {
    label: 'Specificity',
    value: '0.966584',
    winner: 'Attention U-Net',
    icon: Layers,
    accent: '#10b981'
  },
  {
    label: 'Mean Image Dice',
    value: '0.211671',
    winner: 'Attention U-Net',
    icon: Activity,
    accent: '#a78bfa'
  },
  {
    label: 'Hausdorff Distance',
    value: '234.993331 px',
    winner: 'Attention U-Net',
    icon: Ruler,
    accent: '#f59e0b'
  },
  {
    label: 'Surface Distance',
    value: '44.928504 px',
    winner: 'Attention U-Net',
    icon: Ruler,
    accent: '#f43f5e'
  }
]

export default function PerformanceReport() {
  const handleDownloadSummary = () => {
    const reportPayload = {
      title: 'Preliminary Segmentation Model Evaluation Report',
      models: {
        baseline: 'U-Net (Baseline architecture)',
        advanced: 'Attention U-Net (Attention-based architecture)'
      },
      best_metrics: summaryMetrics.map(m => ({
        metric: m.label,
        value: m.value,
        best_model: m.winner
      })),
      overview:
        'Attention U-Net has better reported pixel accuracy, specificity, mean image Dice, Hausdorff distance and surface distance in the supplied evaluation results. However, its precision and recall values should also be considered when interpreting performance.',
      disclaimer:
        'These are preliminary evaluation results. Further model training, tuning and validation may change the metrics. They should not be interpreted as clinically validated results.'
    }

    const blob = new Blob([JSON.stringify(reportPayload, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'vascular_performance_report.json'
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div style={{
      background: '#0f172a',
      border: '1px solid #1e293b',
      borderRadius: '16px',
      padding: '24px',
      color: '#f8fafc',
      display: 'flex',
      flexDirection: 'column',
      gap: '20px'
    }}>
      {/* HEADER + ARCHITECTURES */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'flex-start',
        flexWrap: 'wrap',
        gap: '16px',
        borderBottom: '1px solid #1e293b',
        paddingBottom: '16px'
      }}>
        <div>
          <span className="eyebrow">MODEL EVALUATION</span>
          <h2 style={{ margin: '4px 0', fontSize: '20px', color: '#f8fafc' }}>Performance Report</h2>
          <p style={{ margin: 0, fontSize: '13px', color: '#94a3b8' }}>
            Preliminary evaluation of segmentation models across supplied validation metrics.
          </p>
        </div>

        <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
          <div style={{
            background: '#0b1120',
            border: '1px solid #1e293b',
            borderRadius: '10px',
            padding: '10px 14px'
          }}>
            <span style={{ fontSize: '10.5px', color: '#94a3b8', display: 'block', textTransform: 'uppercase' }}>
              Baseline Model
            </span>
            <strong style={{ fontSize: '13px', color: '#e2e8f0', display: 'block' }}>U-Net</strong>
            <span style={{ fontSize: '11px', color: '#64748b' }}>Baseline architecture</span>
          </div>

          <div style={{
            background: '#0b1120',
            border: '1px solid rgba(56, 189, 248, 0.35)',
            borderRadius: '10px',
            padding: '10px 14px'
          }}>
            <span style={{ fontSize: '10.5px', color: '#38bdf8', display: 'block', textTransform: 'uppercase' }}>
              Advanced Model
            </span>
            <strong style={{ fontSize: '13px', color: '#f8fafc', display: 'block' }}>Attention U-Net</strong>
            <span style={{ fontSize: '11px', color: '#94a3b8' }}>Attention-based architecture</span>
          </div>
        </div>
      </div>

      {/* EVALUATION SUMMARY GRID */}
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
          <Trophy size={17} color="#fbbf24" />
          <div>
            <h3 style={{ margin: 0, fontSize: '15.5px', color: '#f8fafc' }}>Evaluation Summary</h3>
            <span style={{ fontSize: '12px', color: '#94a3b8' }}>Best reported values across the supplied metrics</span>
          </div>
        </div>

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(195px, 1fr))',
          gap: '12px'
        }}>
          {summaryMetrics.map((item) => {
            const IconComponent = item.icon
            return (
              <div
                key={item.label}
                style={{
                  background: '#0b1120',
                  border: '1px solid #1e293b',
                  borderRadius: '12px',
                  padding: '14px',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '6px'
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: '12px', color: '#94a3b8', fontWeight: 500 }}>{item.label}</span>
                  <IconComponent size={15} color={item.accent} />
                </div>

                <strong style={{
                  fontSize: '18px',
                  fontFamily: 'JetBrains Mono, monospace',
                  color: item.accent,
                  letterSpacing: '-0.3px'
                }}>
                  {item.value}
                </strong>

                <span style={{
                  alignSelf: 'flex-start',
                  fontSize: '11px',
                  fontWeight: 600,
                  color: '#34d399',
                  background: 'rgba(16, 185, 129, 0.12)',
                  border: '1px solid rgba(16, 185, 129, 0.25)',
                  padding: '2px 8px',
                  borderRadius: '999px'
                }}>
                  {item.winner}
                </span>
              </div>
            )
          })}
        </div>
      </div>

      {/* EVALUATION OVERVIEW */}
      <div style={{
        background: '#0b1120',
        border: '1px solid #1e293b',
        borderRadius: '12px',
        padding: '16px',
        display: 'flex',
        gap: '12px',
        alignItems: 'flex-start'
      }}>
        <Info size={18} color="#38bdf8" style={{ flexShrink: 0, marginTop: '2px' }} />
        <div>
          <h4 style={{ margin: '0 0 4px', fontSize: '14px', color: '#f8fafc' }}>Evaluation Overview</h4>
          <p style={{ margin: 0, fontSize: '13px', color: '#cbd5e1', lineHeight: 1.55 }}>
            Attention U-Net has better reported pixel accuracy, specificity, mean image Dice, Hausdorff distance and surface distance in the supplied evaluation results. However, its precision and recall values should also be considered when interpreting performance.
          </p>
        </div>
      </div>

      {/* FOOTER DISCLAIMER + DOWNLOAD */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        flexWrap: 'wrap',
        gap: '14px',
        paddingTop: '14px',
        borderTop: '1px solid #1e293b'
      }}>
        <div style={{
          display: 'flex',
          alignItems: 'flex-start',
          gap: '8px',
          maxWidth: '640px',
          fontSize: '11.5px',
          color: '#fbbf24',
          background: 'rgba(245, 158, 11, 0.08)',
          border: '1px solid rgba(245, 158, 11, 0.25)',
          padding: '10px 12px',
          borderRadius: '10px'
        }}>
          <Info size={15} style={{ flexShrink: 0, marginTop: '1px' }} />
          <span>
            These are preliminary evaluation results. Further model training, tuning and validation may change the metrics. They should not be interpreted as clinically validated results.
          </span>
        </div>

        <button
          type="button"
          onClick={handleDownloadSummary}
          className="overview-primary-btn"
          style={{ whiteSpace: 'nowrap' }}
        >
          <Download size={15} />
          Download Full Performance Report
        </button>
      </div>
    </div>
  )
}