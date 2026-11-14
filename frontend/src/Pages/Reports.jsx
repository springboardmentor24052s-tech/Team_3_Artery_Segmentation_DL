
import { FileText, Printer } from 'lucide-react'

export default function Reports() {
  const handlePrint = () => window.print()

  return (
    <div className="dashboard-page">
      <div className="page-heading">
        <div>
          <span className="eyebrow">PROJECT DOCUMENTATION</span>
          <h1>Reports</h1>
          <p>
            Review the available model evaluation information.
          </p>
        </div>

        <button className="primary-button" onClick={handlePrint}>
          <Printer size={17} />
          Print Report
        </button>
      </div>

      <div className="metric-card">
        <FileText size={28} />
        <h2>Model Evaluation Summary</h2>
        <p>
          Selected model: MultiScale Residual U-Net
        </p>
        <p>Test dataset: 300 angiography images</p>
        <p>Dice Coefficient: 0.4709</p>
        <p>IoU: 0.3080</p>
        <p>Precision: 0.4731</p>
        <p>Recall: 0.4688</p>
      </div>

      <div className="metric-card">
        <h3>Report Note</h3>
        <p>
          These are experimental research results and should
          not be interpreted as a standalone clinical diagnosis.
        </p>
      </div>
    </div>
  )
}
