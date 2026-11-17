import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Legend, CartesianGrid
} from 'recharts'
import { Cpu, CheckCircle2, TrendingUp } from 'lucide-react'

const benchmarkData = [
  { metric: 'Dice Score', Baseline: 78.4, AttentionUNet: 86.9 },
  { metric: 'IoU (Jaccard)', Baseline: 65.2, AttentionUNet: 77.1 },
  { metric: 'Precision', Baseline: 81.0, AttentionUNet: 88.4 },
  { metric: 'Sensitivity', Baseline: 76.5, AttentionUNet: 85.7 },
  { metric: 'Specificity', Baseline: 97.8, AttentionUNet: 99.1 }
]

export default function ModelComparison() {
  return (
    <div style={{
      background: '#0f172a',
      border: '1px solid #1e293b',
      borderRadius: '16px',
      padding: '22px',
      color: '#f8fafc'
    }}>
      {/* HEADER */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'flex-start',
        flexWrap: 'wrap',
        gap: '12px',
        marginBottom: '20px',
        borderBottom: '1px solid #1e293b',
        paddingBottom: '14px'
      }}>
        <div>
          <span className="eyebrow">ARCHITECTURE BENCHMARK · ARCADE TEST SPLIT</span>
          <h2 style={{ margin: '4px 0 4px', fontSize: '19px', color: '#f8fafc' }}>
            Baseline U-Net (v1.0) vs. Attention U-Net with scSE (v2.0)
          </h2>
          <p style={{ margin: 0, fontSize: '13px', color: '#94a3b8' }}>
            Quantitative segmentation & stenosis localization comparison across 300 unseen coronary angiograms.
          </p>
        </div>

        <span style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: '6px',
          padding: '6px 12px',
          borderRadius: '999px',
          background: 'rgba(16, 185, 129, 0.14)',
          border: '1px solid rgba(16, 185, 129, 0.35)',
          color: '#34d399',
          fontSize: '12px',
          fontWeight: 700
        }}>
          <TrendingUp size={14} /> +8.5% Dice Gain (v2.0)
        </span>
      </div>

      {/* KPI SUMMARY STRIP */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
        gap: '12px',
        marginBottom: '22px'
      }}>
        {[
          { label: 'Attention U-Net Dice', val: '86.9%', delta: '+8.5% vs Baseline', color: '#38bdf8' },
          { label: 'Mean IoU (Jaccard)', val: '77.1%', delta: '+11.9% distal vessel gain', color: '#10b981' },
          { label: 'Stenosis Detection F1', val: '85.2%', delta: 'Reduced false positives', color: '#a78bfa' },
          { label: 'Avg Inference Latency', val: '~145 ms', delta: 'Real-time PACS ready', color: '#f59e0b' }
        ].map((kpi) => (
          <div key={kpi.label} style={{
            background: '#0b1120',
            border: '1px solid #1e293b',
            borderRadius: '12px',
            padding: '14px'
          }}>
            <span style={{ fontSize: '11.5px', color: '#94a3b8', display: 'block' }}>{kpi.label}</span>
            <strong style={{
              fontSize: '24px',
              fontFamily: 'JetBrains Mono, monospace',
              color: kpi.color,
              display: 'block',
              margin: '4px 0'
            }}>
              {kpi.val}
            </strong>
            <span style={{ fontSize: '11px', color: '#64748b' }}>{kpi.delta}</span>
          </div>
        ))}
      </div>

      {/* CHART + ARCHITECTURE SPECS GRID */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))',
        gap: '18px',
        alignItems: 'stretch'
      }}>
        {/* BAR CHART */}
        <div style={{
          background: '#0b1120',
          border: '1px solid #1e293b',
          borderRadius: '12px',
          padding: '16px'
        }}>
          <span style={{ fontSize: '12px', fontWeight: 700, color: '#cbd5e1', display: 'block', marginBottom: '12px' }}>
            SEGMENTATION & CLASSIFICATION METRICS (%)
          </span>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={benchmarkData} barGap={6}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
              <XAxis dataKey="metric" stroke="#94a3b8" fontSize={11.5} />
              <YAxis stroke="#64748b" fontSize={11} domain={[50, 100]} unit="%" />
              <Tooltip
                contentStyle={{
                  background: '#0f172a',
                  border: '1px solid #334155',
                  borderRadius: '8px',
                  color: '#f8fafc',
                  fontSize: '12px'
                }}
              />
              <Legend wrapperStyle={{ fontSize: '12px' }} />
              <Bar dataKey="Baseline" name="Baseline U-Net (v1.0)" fill="#475569" radius={[6, 6, 0, 0]} />
              <Bar dataKey="AttentionUNet" name="Attention U-Net (v2.0)" fill="#38bdf8" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* ARCHITECTURAL ADVANTAGES TABLE */}
        <div style={{
          background: '#0b1120',
          border: '1px solid #1e293b',
          borderRadius: '12px',
          padding: '16px',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'space-between',
          gap: '10px'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Cpu size={16} color="#38bdf8" />
            <span style={{ fontSize: '12.5px', fontWeight: 700, color: '#e2e8f0' }}>
              WHY ATTENTION U-NET (v2.0) OUTPERFORMS BASELINE
            </span>
          </div>

          {[
            {
              title: 'Multi-Scale Frangi + Top-Hat Vesselness Fusion',
              desc: 'Captures thin distal coronary branches (1–3px width) lost in standard encoder pooling.'
            },
            {
              title: 'Spatial & Channel Squeeze-Excitation (scSE)',
              desc: 'Suppresses catheter & rib-cage background artifacts in low-contrast X-ray frames.'
            },
            {
              title: 'Focal Saliency Alignment (Grad-CAM++)',
              desc: 'Concentrates gradient activation directly over high-grade stenotic constrictions.'
            }
          ].map((item) => (
            <div key={item.title} style={{
              background: '#0f172a',
              border: '1px solid #1e293b',
              borderRadius: '10px',
              padding: '10px 12px',
              display: 'flex',
              gap: '10px',
              alignItems: 'flex-start'
            }}>
              <CheckCircle2 size={16} color="#10b981" style={{ flexShrink: 0, marginTop: '2px' }} />
              <div>
                <strong style={{ fontSize: '12.5px', color: '#f8fafc', display: 'block' }}>{item.title}</strong>
                <span style={{ fontSize: '11.5px', color: '#94a3b8', lineHeight: 1.4 }}>{item.desc}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}