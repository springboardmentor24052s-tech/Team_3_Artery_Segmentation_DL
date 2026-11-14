import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
} from "recharts";
import "./ModelAnalytics.css";

const testMetrics = [
  { metric: "Dice", score: 47.09 },
  { metric: "IoU", score: 30.80 },
  { metric: "Precision", score: 47.31 },
  { metric: "Recall", score: 46.88 },
  { metric: "Macro Dice", score: 44.66 },
];

const validationComparison = [
  { model: "Boundary-Aware", dice: 51.53 },
  { model: "MultiScale Residual U-Net", dice: 51.65 },
];

export default function ModelAnalytics() {
  return (
    <div className="analytics-page">
      <div className="analytics-heading">
        <div>
          <span className="analytics-label">AI PERFORMANCE</span>
          <h1>Model Analytics</h1>
          <p>Segmentation model performance and evaluation metrics</p>
        </div>
        <span className="analytics-status">Test Evaluation</span>
      </div>

      <div className="analytics-model-card">
        <div>
          <span>SELECTED MODEL</span>
          <h2>MultiScale Residual U-Net</h2>
          <p>Final selected stenosis segmentation model</p>
        </div>
        <div className="analytics-dice">
          <strong>47.09%</strong>
          <span>Test Dice Score</span>
        </div>
      </div>

      <div className="analytics-metric-grid">
        {testMetrics.map((item) => (
          <div className="analytics-metric-card" key={item.metric}>
            <span>{item.metric}</span>
            <strong>{item.score.toFixed(2)}%</strong>
            <div className="metric-track">
              <div style={{ width: `${item.score}%` }} />
            </div>
          </div>
        ))}
      </div>

      <div className="analytics-chart-grid">
        <section className="analytics-chart-card">
          <h3>Test Performance Metrics</h3>
          <p>Selected model — held-out test dataset</p>
          <div className="chart-area">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={testMetrics}>
                <CartesianGrid strokeDasharray="3 3" stroke="#29364a" />
                <XAxis dataKey="metric" stroke="#9baac0" />
                <YAxis domain={[0, 100]} stroke="#9baac0" />
                <Tooltip
                  contentStyle={{
                    background: "#111c2d",
                    border: "1px solid #334155",
                    borderRadius: 8,
                  }}
                  formatter={(value) => [`${value}%`, "Score"]}
                />
                <Bar dataKey="score" fill="#58a6e7" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>

        <section className="analytics-chart-card">
          <h3>Validation Dice Comparison</h3>
          <p>Two best recorded validation experiments</p>
          <div className="chart-area">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={validationComparison}>
                <CartesianGrid strokeDasharray="3 3" stroke="#29364a" />
                <XAxis
                  dataKey="model"
                  stroke="#9baac0"
                  interval={0}
                  tick={{ fontSize: 10 }}
                />
                <YAxis domain={[0, 100]} stroke="#9baac0" />
                <Tooltip
                  contentStyle={{
                    background: "#111c2d",
                    border: "1px solid #334155",
                  }}
                  formatter={(value) => [`${value}%`, "Validation Dice"]}
                />
                <Bar dataKey="dice" fill="#31c6a3" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>
      </div>

      <section className="analytics-chart-card radar-card">
        <h3>Model Metric Distribution</h3>
        <p>Visual overview of the selected model's test metrics</p>
        <ResponsiveContainer width="100%" height={340}>
          <RadarChart data={testMetrics}>
            <PolarGrid stroke="#334155" />
            <PolarAngleAxis dataKey="metric" stroke="#a9b8cb" />
            <PolarRadiusAxis domain={[0, 100]} stroke="#64748b" />
            <Radar
              name="Test Score"
              dataKey="score"
              stroke="#58a6e7"
              fill="#58a6e7"
              fillOpacity={0.3}
            />
            <Tooltip />
          </RadarChart>
        </ResponsiveContainer>
      </section>

      <section className="analytics-chart-card radar-card">
        <h3>Qualitative Mask & Centerline Comparison</h3>
        <p>Visual comparison between Baseline U-Net and Attention-guided segmentation outputs</p>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
            gap: "16px",
            marginTop: "14px",
          }}
        >
          {[
            { title: "Baseline U-Net Mask", file: "baseline_mask.png" },
            { title: "Baseline Centerline", file: "baseline_centerline.png" },
            { title: "Attention U-Net Mask", file: "attention_mask.png" },
            { title: "Attention Centerline", file: "attention_centerline.png" },
          ].map((item) => (
            <div
              key={item.file}
              style={{
                background: "#0b1320",
                border: "1px solid #26364b",
                borderRadius: "10px",
                padding: "12px",
                textAlign: "center",
              }}
            >
              <span
                style={{
                  fontSize: "12px",
                  color: "#75c5f5",
                  fontWeight: 600,
                  display: "block",
                  marginBottom: "8px",
                }}
              >
                {item.title}
              </span>
              <img
                src={`http://127.0.0.1:8000/test-files/${item.file}`}
                alt={item.title}
                style={{
                  width: "100%",
                  maxHeight: "200px",
                  objectFit: "contain",
                  borderRadius: "6px",
                  background: "#000",
                }}
              />
            </div>
          ))}
        </div>
      </section>

      <div className="analytics-note">
        <strong>Evaluation note</strong>
        <p>
          Test metrics are from the held-out test dataset. Validation Dice
          scores are shown separately and must not be interpreted as test
          performance. These are research evaluation metrics, not clinical
          accuracy guarantees.
        </p>
      </div>
    </div>
  );
}