import { useState, useEffect } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, Cell,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  LineChart, Line,
} from 'recharts'
import {
  BarChart3, Brain, GitBranch, Layers, Target, TrendingUp,
  AlertTriangle, RefreshCw, Search, Fingerprint, Network,
  FlaskConical, Award, Lightbulb, Boxes, Activity, Zap, Shield
} from 'lucide-react'
import { getModelPerformance, getMiningResults } from '../services/api'

// ── CSS variable palette (matches index.css) ──────────────────────
const C = {
  primary:   '#0f1422',
  secondary: '#5a6275',
  label:     '#8792a8',
  border:    '#e2e6ee',
  surface:   '#ffffff',
  page:      '#f0f2f5',
  accent:    '#4f6ef7',
  green:     '#059669',
  red:       '#dc2626',
  amber:     '#d97706',
  purple:    '#7c3aed',
}

// ── Recharts axis / grid common props ────────────────────────────
const AX = { fill: C.label, fontSize: 11 }
const GRID = '#eef0f5'
const TT = {
  contentStyle: { background: '#fff', border: `1px solid ${C.border}`, borderRadius: 12, boxShadow: '0 8px 24px rgba(15,20,40,.10)' },
  labelStyle:   { color: C.primary,   fontWeight: 600 },
  itemStyle:    { color: C.secondary },
}

// ── Fallback data (rich – used when backend is offline) ───────────
const DEMO_PERF = {
  models: [
    { name: 'Logistic Regression', accuracy: 0.999,  precision: 1.000, recall: 0.9412, f1: 0.9697, auc: 0.9996,
      true_positives: 16, false_positives: 0, true_negatives: 983, false_negatives: 1 },
    { name: 'Random Forest',       accuracy: 0.998,  precision: 1.000, recall: 0.8824, f1: 0.9375, auc: 0.9998,
      true_positives: 15, false_positives: 0, true_negatives: 983, false_negatives: 2 },
    { name: 'XGBoost',             accuracy: 0.991,  precision: 0.6818,recall: 0.8824, f1: 0.7692, auc: 0.9986,
      true_positives: 15, false_positives: 7, true_negatives: 976, false_negatives: 2 },
  ],
  best_model: 'logistic_regression',
  confusion_matrix: { true_positives: 16, false_positives: 0, true_negatives: 983, false_negatives: 1 },
}

const DEMO_MINING = {
  anomaly_detection: {
    isolation_forest: {
      f1_score: 0.908, precision: 0.84, recall: 0.988,
      detection_rate: 0.988, false_positive_rate: 0.032,
      anomalies_found: 100, true_positives: 84, false_positives: 16, true_negatives: 4899, false_negatives: 1,
    },
    local_outlier_factor: {
      f1_score: 0.953, precision: 0.94, recall: 0.965,
      detection_rate: 0.965, false_positive_rate: 0.012,
      anomalies_found: 87, true_positives: 82, false_positives: 5, true_negatives: 4910, false_negatives: 3,
    },
    if_sweep: [
      { contamination: 0.001, f1_score: 0.111, precision: 1.0, recall: 0.059 },
      { contamination: 0.002, f1_score: 0.211, precision: 1.0, recall: 0.118 },
      { contamination: 0.005, f1_score: 0.455, precision: 1.0, recall: 0.294 },
      { contamination: 0.010, f1_score: 0.741, precision: 1.0, recall: 0.588 },
      { contamination: 0.020, f1_score: 0.908, precision: 0.84, recall: 0.988 },
      { contamination: 0.050, f1_score: 0.508, precision: 0.34, recall: 1.000 },
    ],
  },
  clustering: {
    kmeans: {
      n_clusters: 5, silhouette_score: 0.412,
      cluster_distribution: [
        { cluster: 0, label: 'Low-value Normal',      count: 1960, fraud_count: 0,  fraud_pct: 0.00 },
        { cluster: 1, label: 'Medium Transactions',   count: 1510, fraud_count: 4,  fraud_pct: 0.26 },
        { cluster: 2, label: 'Standard Purchases',    count: 1250, fraud_count: 12, fraud_pct: 0.96 },
        { cluster: 3, label: 'High-value Suspicious', count:  620, fraud_count: 38, fraud_pct: 6.13 },
        { cluster: 4, label: 'Extreme Anomalies',     count:  160, fraud_count: 31, fraud_pct: 19.4 },
      ],
    },
    dbscan: {
      n_clusters_found: 8, noise_points: 247,
      noise_points_fraud_pct: 28.5, silhouette_score: 0.378,
    },
  },
  pattern_discovery: {
    association_rules: [
      { rule: 'High V14 deviation → Fraud',          confidence: 0.89, support: 0.15, lift: 5.2 },
      { rule: 'Negative V12 & V10 → Fraud',          confidence: 0.82, support: 0.11, lift: 4.8 },
      { rule: 'V4 > 2σ + Amount > $500 → Fraud',     confidence: 0.78, support: 0.08, lift: 4.5 },
      { rule: 'V17 outlier + V14 outlier → Fraud',   confidence: 0.91, support: 0.12, lift: 5.6 },
      { rule: 'Low V3 + High V7 deviation → Fraud',  confidence: 0.74, support: 0.07, lift: 4.1 },
    ],
    temporal_patterns: { peak_fraud_hours: [1, 2, 3, 23], avg_fraud_amount: 122.21, avg_normal_amount: 88.29 },
    key_insights: [
      'V14 is the strongest fraud indicator with highest SHAP impact',
      'DBSCAN noise points contain 28.5% fraud — a strong anomaly signal',
      'Cluster 4 (K-Means) captures 19.4% fraud concentration',
      'LOF outperforms Isolation Forest with 95.3% F1 vs 90.8% F1',
      'Association rules reveal multi-feature fraud signatures with high lift',
      'Fraud transactions peak during midnight–3 AM window',
    ],
  },
}

const DEMO_FEATURE_IMP = [
  { feature: 'V14', importance: 19.8 }, { feature: 'V17', importance: 14.2 },
  { feature: 'V12', importance: 12.8 }, { feature: 'V10', importance: 11.2 },
  { feature: 'V16', importance:  9.8 }, { feature: 'V11', importance:  8.7 },
  { feature: 'V4',  importance:  7.6 }, { feature: 'V3',  importance:  6.2 },
  { feature: 'V7',  importance:  5.1 }, { feature: 'V9',  importance:  4.6 },
]

const MODEL_COLORS = ['#4f6ef7', '#059669', '#d97706']
const CLUSTER_COLORS = ['#4f6ef7', '#059669', '#a855f7', '#d97706', '#dc2626']

// ── Small reusable components ─────────────────────────────────────

function Card({ children, className = '' }) {
  return (
    <div className={`glass-card p-6 ${className}`} style={{ animationFillMode: 'both' }}>
      {children}
    </div>
  )
}

function SecHeader({ icon: Icon, title, subtitle, iconBg, iconColor }) {
  return (
    <div className="flex items-start gap-3 mb-5">
      <div className="w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0"
           style={{ background: iconBg }}>
        <Icon className="w-[18px] h-[18px]" style={{ color: iconColor }} />
      </div>
      <div>
        <h2 className="text-base font-bold" style={{ color: C.primary }}>{title}</h2>
        {subtitle && <p className="text-xs mt-0.5" style={{ color: C.secondary }}>{subtitle}</p>}
      </div>
    </div>
  )
}

function Pill({ label, value, bg, color }) {
  return (
    <div className="rounded-xl p-3 text-center" style={{ background: bg, border: `1px solid ${color}22` }}>
      <p className="text-[10px] font-semibold uppercase tracking-wider mb-1" style={{ color: C.label }}>{label}</p>
      <p className="text-lg font-bold" style={{ color }}>{value}</p>
    </div>
  )
}

function ProgressBar({ pct, color }) {
  return (
    <div className="w-full h-2 rounded-full overflow-hidden" style={{ background: C.page }}>
      <div className="h-full rounded-full transition-all duration-700" style={{ width: `${pct}%`, background: color }} />
    </div>
  )
}

// ── Main ─────────────────────────────────────────────────────────

function Analytics() {
  const [perf,   setPerf]   = useState(null)
  const [mining, setMining] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error,   setError]   = useState(null)
  const [modelTab, setModelTab] = useState('bar')

  const fetchData = async () => {
    setLoading(true); setError(null)
    try {
      const [pr, mr] = await Promise.allSettled([getModelPerformance(), getMiningResults()])
      setPerf  (pr.status === 'fulfilled' ? pr.value : DEMO_PERF)
      setMining(mr.status === 'fulfilled' ? mr.value : DEMO_MINING)
    } catch {
      setPerf(DEMO_PERF); setMining(DEMO_MINING)
      setError('Using demo data — backend unavailable')
    } finally { setLoading(false) }
  }

  useEffect(() => { fetchData() }, [])

  const p = perf   || DEMO_PERF
  const m = mining || DEMO_MINING

  // ── normalise model name display
  const displayName = (n = '') =>
    n.replace('logistic_regression','Logistic Reg.').replace('random_forest','Random Forest').replace('xgboost','XGBoost')

  // ── Model comparison bar-chart data
  const barData = (p.models || []).map(mo => ({
    name: displayName(mo.name),
    Precision: +((mo.precision||0)*100).toFixed(1),
    Recall:    +((mo.recall   ||0)*100).toFixed(1),
    'F1 Score':+((mo.f1      ||0)*100).toFixed(1),
    AUC:       +((mo.auc     ||0)*100).toFixed(1),
  }))

  // ── Radar data — keys must match model display names
  const radarData = ['Precision','Recall','F1 Score','AUC','Accuracy'].map(metric => {
    const row = { metric }
    ;(p.models || []).forEach(mo => {
      const key = displayName(mo.name)
      const v = metric === 'Precision' ? mo.precision
              : metric === 'Recall'    ? mo.recall
              : metric === 'F1 Score'  ? mo.f1
              : metric === 'AUC'       ? mo.auc
              :                         mo.accuracy
      row[key] = +((v||0)*100).toFixed(1)
    })
    return row
  })

  // ── Feature importance
  const featData = (p.feature_importance
    ? p.feature_importance.map(f => ({ feature: f.feature, importance: +(f.importance*100).toFixed(1) }))
    : DEMO_FEATURE_IMP
  ).slice(0, 10)

  // ── Anomaly data
  const ad  = m.anomaly_detection || {}
  const ifd = ad.isolation_forest || {}
  const lfd = ad.local_outlier_factor || {}

  const anomalyBar = [
    { name: 'Isolation Forest', F1: +((ifd.f1_score||0)*100).toFixed(1), Precision: +((ifd.precision||0)*100).toFixed(1), Recall: +((ifd.recall||0)*100).toFixed(1) },
    { name: 'LOF',              F1: +((lfd.f1_score||0)*100).toFixed(1), Precision: +((lfd.precision||0)*100).toFixed(1), Recall: +((lfd.recall||0)*100).toFixed(1) },
  ]

  // ── IF sweep line chart
  const sweepData = (ad.if_sweep || DEMO_MINING.anomaly_detection.if_sweep).map(r => ({
    contamination: +(r.contamination*100).toFixed(1) + '%',
    F1:        +((r.f1_score||0)*100).toFixed(1),
    Precision: +((r.precision||0)*100).toFixed(1),
    Recall:    +((r.recall   ||0)*100).toFixed(1),
  }))

  // ── Clustering
  const cl       = m.clustering  || {}
  const kmeans   = cl.kmeans     || {}
  const dbscan   = cl.dbscan     || {}
  const clusters = kmeans.cluster_distribution || DEMO_MINING.clustering.kmeans.cluster_distribution

  // ── Pattern
  const pd   = m.pattern_discovery || DEMO_MINING.pattern_discovery
  const rules = pd.association_rules || []

  // ── Confusion matrix (from best model or perf endpoint)
  const cm = p.confusion_matrix || (() => {
    const best = (p.models||[]).find(x => displayName(x.name).toLowerCase().includes((p.best_model||'').replace('_',' '))) || (p.models||[])[0] || {}
    return { true_positives: best.true_positives, false_positives: best.false_positives,
             true_negatives: best.true_negatives,  false_negatives: best.false_negatives }
  })()

  if (loading) {
    return (
      <div className="space-y-6 animate-fade-in">
        <div>
          <h1 className="text-2xl font-bold" style={{ color: C.primary }}>Analytics & Data Mining</h1>
          <p className="text-sm mt-1" style={{ color: C.secondary }}>Loading results…</p>
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {[1,2,3,4].map(i => <div key={i} className="glass-card p-6 h-64 loading-shimmer rounded-2xl" />)}
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6 animate-fade-in">

      {/* ── Header ── */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold" style={{ color: C.primary }}>
            Analytics & <span className="gradient-text">Data Mining</span>
          </h1>
          <p className="text-sm mt-0.5" style={{ color: C.secondary }}>
            Model evaluation · Anomaly detection · Clustering · Pattern discovery
          </p>
        </div>
        <button onClick={fetchData} className="btn-secondary flex items-center gap-2 text-sm">
          <RefreshCw className="w-4 h-4" /> Refresh
        </button>
      </div>

      {error && (
        <div className="rounded-xl px-4 py-3 flex items-center gap-3"
             style={{ background: 'rgba(245,158,11,.08)', border: '1px solid rgba(245,158,11,.2)' }}>
          <AlertTriangle className="w-4 h-4 flex-shrink-0" style={{ color: C.amber }} />
          <p className="text-sm" style={{ color: '#92400e' }}>{error}</p>
        </div>
      )}

      {/* ══════════════════════════════════════════════════
          SECTION 1 — MODEL PERFORMANCE
          ══════════════════════════════════════════════════ */}
      <Card>
        <SecHeader icon={TrendingUp} title="Model Performance Comparison"
          subtitle={`Best model: ${displayName(p.best_model||'')} — Logistic Regression · Random Forest · XGBoost`}
          iconBg="rgba(79,110,247,.10)" iconColor={C.accent} />

        {/* Tab switcher */}
        <div className="flex gap-2 mb-5">
          {[['bar','Bar Chart'],['radar','Radar'],['details','Details']].map(([id,label]) => (
            <button key={id} onClick={() => setModelTab(id)}
              className="px-4 py-1.5 rounded-lg text-sm font-medium transition-all duration-200"
              style={modelTab === id
                ? { background: 'rgba(79,110,247,.12)', color: C.accent, border: `1px solid rgba(79,110,247,.25)` }
                : { background: 'transparent', color: C.secondary, border: `1px solid ${C.border}` }}>
              {label}
            </button>
          ))}
        </div>

        {/* Bar Chart */}
        {modelTab === 'bar' && (
          <ResponsiveContainer width="100%" height={340}>
            <BarChart data={barData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={GRID} />
              <XAxis dataKey="name" tick={AX} axisLine={{ stroke: C.border }} tickLine={false} />
              <YAxis domain={[60, 100]} tick={AX} axisLine={{ stroke: C.border }} tickLine={false} />
              <Tooltip {...TT} formatter={v => [`${v}%`, '']} />
              <Legend formatter={v => <span style={{ color: C.secondary, fontSize: 12 }}>{v}</span>} />
              <Bar dataKey="Precision" fill="#4f6ef7" radius={[4,4,0,0]} />
              <Bar dataKey="Recall"    fill="#059669" radius={[4,4,0,0]} />
              <Bar dataKey="F1 Score"  fill="#d97706" radius={[4,4,0,0]} />
              <Bar dataKey="AUC"       fill="#a855f7" radius={[4,4,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        )}

        {/* Radar */}
        {modelTab === 'radar' && (
          <ResponsiveContainer width="100%" height={380}>
            <RadarChart data={radarData}>
              <PolarGrid stroke={GRID} />
              <PolarAngleAxis dataKey="metric" tick={{ fill: C.secondary, fontSize: 11 }} />
              <PolarRadiusAxis domain={[60, 100]} tick={{ fill: C.label, fontSize: 10 }} />
              {(p.models || []).map((mo, i) => (
                <Radar key={mo.name} name={displayName(mo.name)} dataKey={displayName(mo.name)}
                  stroke={MODEL_COLORS[i]} fill={MODEL_COLORS[i]} fillOpacity={0.12} strokeWidth={2} />
              ))}
              <Legend formatter={v => <span style={{ color: C.secondary, fontSize: 12 }}>{v}</span>} />
              <Tooltip {...TT} formatter={v => [`${v}%`, '']} />
            </RadarChart>
          </ResponsiveContainer>
        )}

        {/* Details table */}
        {modelTab === 'details' && (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr style={{ borderBottom: `1px solid ${C.border}` }}>
                  {['Model','Accuracy','Precision','Recall','F1 Score','AUC-ROC','MCC'].map(h => (
                    <th key={h} className="text-left text-[10px] font-semibold uppercase tracking-wider py-2.5 px-4"
                        style={{ color: C.label }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(p.models || []).map((mo, i) => {
                  const isBest = displayName(mo.name).toLowerCase().replace('.','').replace(' ','') ===
                                 displayName(p.best_model||'').toLowerCase().replace('.','').replace(' ','')
                  return (
                    <tr key={mo.name} style={{ borderBottom: `1px solid ${C.border}`,
                                               background: isBest ? 'rgba(79,110,247,.04)' : 'transparent' }}>
                      <td className="py-3 px-4">
                        <div className="flex items-center gap-2">
                          {isBest && <Award className="w-3.5 h-3.5" style={{ color: C.amber }} />}
                          <span className="text-sm font-semibold" style={{ color: isBest ? C.accent : C.primary }}>
                            {displayName(mo.name)}
                          </span>
                        </div>
                      </td>
                      <td className="py-3 px-4 text-sm text-center" style={{ color: C.secondary }}>{((mo.accuracy||0)*100).toFixed(2)}%</td>
                      <td className="py-3 px-4 text-sm text-center font-medium" style={{ color: C.accent }}>{((mo.precision||0)*100).toFixed(1)}%</td>
                      <td className="py-3 px-4 text-sm text-center font-medium" style={{ color: C.green }}>{((mo.recall||0)*100).toFixed(1)}%</td>
                      <td className="py-3 px-4 text-sm text-center font-medium" style={{ color: C.amber }}>{((mo.f1||0)*100).toFixed(1)}%</td>
                      <td className="py-3 px-4 text-sm text-center font-medium" style={{ color: C.purple }}>{((mo.auc||0)*100).toFixed(1)}%</td>
                      <td className="py-3 px-4 text-sm text-center" style={{ color: C.secondary }}>{mo.mcc ? mo.mcc.toFixed(4) : '—'}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      {/* ══════════════════════════════════════════════════
          SECTION 2 — ANOMALY DETECTION (3 cols)
          ══════════════════════════════════════════════════ */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

        {/* Anomaly Method Comparison */}
        <Card>
          <SecHeader icon={Search} title="Anomaly Detection" subtitle="Isolation Forest vs LOF"
            iconBg="rgba(220,38,38,.10)" iconColor={C.red} />
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={anomalyBar} margin={{ top: 0, right: 10, left: -10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={GRID} />
              <XAxis dataKey="name" tick={{ ...AX, fontSize: 10 }} axisLine={{ stroke: C.border }} tickLine={false} />
              <YAxis domain={[0, 100]} tick={AX} axisLine={{ stroke: C.border }} tickLine={false} />
              <Tooltip {...TT} formatter={v => [`${v}%`, '']} />
              <Legend formatter={v => <span style={{ color: C.secondary, fontSize: 11 }}>{v}</span>} />
              <Bar dataKey="F1"        fill="#4f6ef7" radius={[4,4,0,0]} />
              <Bar dataKey="Precision" fill="#059669" radius={[4,4,0,0]} />
              <Bar dataKey="Recall"    fill="#d97706" radius={[4,4,0,0]} />
            </BarChart>
          </ResponsiveContainer>

          <div className="grid grid-cols-2 gap-3 mt-4">
            <Pill label="IF F1" value={`${((ifd.f1_score||0)*100).toFixed(1)}%`} bg="rgba(79,110,247,.06)" color={C.accent} />
            <Pill label="LOF F1" value={`${((lfd.f1_score||0)*100).toFixed(1)}%`} bg="rgba(5,150,105,.06)" color={C.green} />
          </div>

          <div className="mt-3 rounded-xl p-3 flex items-start gap-2"
               style={{ background: 'rgba(79,110,247,.06)', border: `1px solid rgba(79,110,247,.12)` }}>
            <Award className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" style={{ color: C.accent }} />
            <div>
              <p className="text-xs font-semibold" style={{ color: C.accent }}>
                {(lfd.f1_score||0) >= (ifd.f1_score||0) ? 'LOF wins' : 'Isolation Forest wins'}
              </p>
              <p className="text-[10px] mt-0.5" style={{ color: C.secondary }}>
                Best F1: {Math.max((ifd.f1_score||0),(lfd.f1_score||0)*100 >= (ifd.f1_score||0)*100 ? lfd.f1_score : ifd.f1_score).toFixed(3)*100 >= 0 ?
                  `${(Math.max(ifd.f1_score||0, lfd.f1_score||0)*100).toFixed(1)}%` : '—'}
              </p>
            </div>
          </div>
        </Card>

        {/* IF Contamination Sweep */}
        <Card>
          <SecHeader icon={FlaskConical} title="IF Parameter Sweep" subtitle="F1 / Precision / Recall vs contamination"
            iconBg="rgba(124,58,237,.10)" iconColor={C.purple} />
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={sweepData} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={GRID} />
              <XAxis dataKey="contamination" tick={{ ...AX, fontSize: 10 }} axisLine={{ stroke: C.border }} tickLine={false} />
              <YAxis domain={[0, 100]} tick={AX} axisLine={{ stroke: C.border }} tickLine={false} />
              <Tooltip {...TT} formatter={v => [`${v}%`, '']} />
              <Legend formatter={v => <span style={{ color: C.secondary, fontSize: 11 }}>{v}</span>} />
              <Line type="monotone" dataKey="F1"        stroke={C.accent}  strokeWidth={2} dot={{ r: 3, fill: C.accent }} />
              <Line type="monotone" dataKey="Precision" stroke={C.green}   strokeWidth={2} dot={{ r: 3, fill: C.green  }} />
              <Line type="monotone" dataKey="Recall"    stroke={C.amber}   strokeWidth={2} dot={{ r: 3, fill: C.amber  }} />
            </LineChart>
          </ResponsiveContainer>
          <p className="text-xs mt-2 text-center" style={{ color: C.label }}>
            Peak F1 at contamination = {(ifd.contamination||0.02)*100}%
          </p>
        </Card>

        {/* Clustering */}
        <Card>
          <SecHeader icon={Boxes} title="Clustering Analysis" subtitle="K-Means (5 clusters) · DBSCAN"
            iconBg="rgba(5,150,105,.10)" iconColor={C.green} />

          <div className="space-y-2 mb-4">
            {clusters.map((c, i) => {
              const pct = c.fraud_pct || ((c.fraud_count / Math.max(c.count, 1)) * 100)
              const maxCount = clusters[0]?.count || 1
              return (
                <div key={i}>
                  <div className="flex items-center justify-between text-[10px] mb-0.5">
                    <span className="font-medium" style={{ color: C.secondary }}>{c.label || `Cluster ${c.cluster}`}</span>
                    <span className="font-bold" style={{ color: pct > 5 ? C.red : pct > 1 ? C.amber : C.green }}>
                      {pct.toFixed(1)}% fraud
                    </span>
                  </div>
                  <ProgressBar pct={(c.count / maxCount) * 100} color={CLUSTER_COLORS[i]} />
                  <p className="text-[9px] mt-0.5" style={{ color: C.label }}>{(c.count||0).toLocaleString()} txns</p>
                </div>
              )
            })}
          </div>

          <div style={{ borderTop: `1px solid ${C.border}` }} className="pt-3 mt-1">
            <div className="flex items-center gap-2 mb-2">
              <Network className="w-3.5 h-3.5" style={{ color: C.purple }} />
              <span className="text-xs font-semibold" style={{ color: C.primary }}>DBSCAN</span>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <Pill label="Clusters" value={dbscan.n_clusters_found || 8}    bg="rgba(124,58,237,.06)" color={C.purple} />
              <Pill label="Noise pts" value={(dbscan.noise_points||247).toLocaleString()} bg="rgba(217,119,6,.06)"  color={C.amber} />
            </div>
            <div className="mt-2 rounded-xl p-2.5"
                 style={{ background: 'rgba(220,38,38,.06)', border: '1px solid rgba(220,38,38,.12)' }}>
              <p className="text-xs font-semibold" style={{ color: C.red }}>
                {dbscan.noise_points_fraud_pct || 28.5}% of noise = fraud
              </p>
              <p className="text-[10px] mt-0.5" style={{ color: C.secondary }}>Noise points = strong anomaly signal</p>
            </div>
          </div>
        </Card>
      </div>

      {/* ══════════════════════════════════════════════════
          SECTION 3 — FEATURE IMPORTANCE + CONFUSION MATRIX
          ══════════════════════════════════════════════════ */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

        {/* Feature Importance (2/3 width) */}
        <div className="lg:col-span-2">
          <Card>
            <SecHeader icon={Brain} title="Feature Importance (SHAP)" subtitle="Top 10 most influential PCA features for fraud detection"
              iconBg="rgba(124,58,237,.10)" iconColor={C.purple} />
            <ResponsiveContainer width="100%" height={340}>
              <BarChart data={featData} layout="vertical" margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={GRID} horizontal={false} />
                <XAxis type="number" tick={AX} axisLine={{ stroke: C.border }} tickLine={false}
                  label={{ value: 'Importance (%)', position: 'insideBottom', offset: -2, fill: C.label, fontSize: 10 }} />
                <YAxis dataKey="feature" type="category" tick={{ fill: C.primary, fontSize: 12, fontWeight: 600 }}
                  axisLine={{ stroke: C.border }} tickLine={false} width={40} />
                <Tooltip {...TT} formatter={v => [`${v}%`, 'Importance']} />
                <Bar dataKey="importance" radius={[0, 6, 6, 0]} barSize={20}>
                  {featData.map((_, i) => (
                    <Cell key={i} fill={i < 3 ? C.accent : i < 6 ? C.green : C.purple} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>

        {/* Confusion Matrix (1/3 width) */}
        <Card>
          <SecHeader icon={Target} title="Confusion Matrix" subtitle={`Best model: ${displayName(p.best_model||'')}`}
            iconBg="rgba(5,150,105,.10)" iconColor={C.green} />

          {cm && (
            <>
              <div className="grid grid-cols-2 gap-2 mb-4">
                {[
                  { label: 'True Neg',  value: cm.true_negatives,  bg: 'rgba(5,150,105,.08)',   color: C.green,  border: 'rgba(5,150,105,.2)' },
                  { label: 'False Pos', value: cm.false_positives, bg: 'rgba(217,119,6,.08)',   color: C.amber,  border: 'rgba(217,119,6,.2)' },
                  { label: 'False Neg', value: cm.false_negatives, bg: 'rgba(220,38,38,.08)',   color: C.red,    border: 'rgba(220,38,38,.2)' },
                  { label: 'True Pos',  value: cm.true_positives,  bg: 'rgba(79,110,247,.08)',  color: C.accent, border: 'rgba(79,110,247,.2)' },
                ].map(cell => (
                  <div key={cell.label} className="rounded-xl p-3 text-center transition-transform hover:scale-[1.03] duration-200"
                       style={{ background: cell.bg, border: `1px solid ${cell.border}` }}>
                    <p className="text-xl font-bold" style={{ color: cell.color }}>{(cell.value||0).toLocaleString()}</p>
                    <p className="text-[10px] mt-0.5" style={{ color: C.label }}>{cell.label}</p>
                  </div>
                ))}
              </div>

              {(() => {
                const tp = cm.true_positives||0, fp = cm.false_positives||0
                const tn = cm.true_negatives||0, fn = cm.false_negatives||0
                const total = tp+fp+tn+fn || 1
                const prec = tp/(tp+fp||1), rec = tp/(tp+fn||1), acc = (tp+tn)/total
                return (
                  <div className="space-y-2">
                    {[
                      { label: 'Accuracy',  val: acc,  color: C.accent },
                      { label: 'Precision', val: prec, color: C.green  },
                      { label: 'Recall',    val: rec,  color: C.amber  },
                    ].map(r => (
                      <div key={r.label}>
                        <div className="flex justify-between text-xs mb-0.5">
                          <span style={{ color: C.secondary }}>{r.label}</span>
                          <span className="font-semibold" style={{ color: r.color }}>{(r.val*100).toFixed(2)}%</span>
                        </div>
                        <ProgressBar pct={r.val*100} color={r.color} />
                      </div>
                    ))}
                  </div>
                )
              })()}
            </>
          )}
        </Card>
      </div>

      {/* ══════════════════════════════════════════════════
          SECTION 4 — ASSOCIATION RULES + PATTERN INSIGHTS
          ══════════════════════════════════════════════════ */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

        {/* Association Rules */}
        <Card>
          <SecHeader icon={GitBranch} title="Association Rule Mining" subtitle="Apriori-derived rules linking PCA features to fraud"
            iconBg="rgba(217,119,6,.10)" iconColor={C.amber} />

          <div className="space-y-2">
            {rules.map((r, i) => (
              <div key={i} className="rounded-xl p-3 transition-all duration-200"
                   style={{ background: C.page, border: `1px solid ${C.border}` }}
                   onMouseEnter={e => e.currentTarget.style.borderColor = '#c5cef8'}
                   onMouseLeave={e => e.currentTarget.style.borderColor = C.border}>
                <p className="text-xs font-semibold mb-2" style={{ color: C.primary }}>{r.rule}</p>
                <div className="flex items-center gap-3">
                  {[
                    { label: 'Conf', val: `${(r.confidence*100).toFixed(0)}%`, color: C.accent },
                    { label: 'Sup',  val: `${(r.support*100).toFixed(0)}%`,    color: C.green  },
                    { label: 'Lift', val: r.lift?.toFixed(1),                  color: C.amber  },
                  ].map(x => (
                    <span key={x.label} className="text-[10px] font-semibold px-2 py-0.5 rounded-md"
                          style={{ background: `${x.color}11`, color: x.color }}>
                      {x.label}: {x.val}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>

          {pd.temporal_patterns && (
            <div className="mt-4 rounded-xl p-3" style={{ background: 'rgba(79,110,247,.06)', border: `1px solid rgba(79,110,247,.12)` }}>
              <p className="text-xs font-semibold mb-1" style={{ color: C.primary }}>⏱ Temporal Patterns</p>
              <p className="text-[11px]" style={{ color: C.secondary }}>
                Peak fraud hours: <span className="font-bold" style={{ color: C.red }}>
                  {(pd.temporal_patterns.peak_fraud_hours||[]).join(':00, ')}:00
                </span>
              </p>
              <p className="text-[11px] mt-0.5" style={{ color: C.secondary }}>
                Avg fraud amount: <span className="font-bold" style={{ color: C.amber }}>
                  ${pd.temporal_patterns.avg_fraud_amount}
                </span>
                {' '}vs normal <span className="font-bold" style={{ color: C.green }}>
                  ${pd.temporal_patterns.avg_normal_amount}
                </span>
              </p>
            </div>
          )}
        </Card>

        {/* Key Insights */}
        <Card>
          <SecHeader icon={Lightbulb} title="Key Data Mining Insights" subtitle="Most significant discoveries from the full pipeline"
            iconBg="rgba(217,119,6,.10)" iconColor={C.amber} />

          <div className="space-y-2">
            {(pd.key_insights || DEMO_MINING.pattern_discovery.key_insights).map((ins, i) => (
              <div key={i} className="flex items-start gap-3 rounded-xl p-3 transition-all duration-200"
                   style={{ background: C.page, border: `1px solid ${C.border}` }}
                   onMouseEnter={e => e.currentTarget.style.background = '#fff'}
                   onMouseLeave={e => e.currentTarget.style.background = C.page}>
                <div className="w-6 h-6 rounded-lg flex items-center justify-center flex-shrink-0 mt-0.5 text-[10px] font-bold"
                     style={{ background: 'rgba(217,119,6,.10)', color: C.amber }}>
                  {i + 1}
                </div>
                <p className="text-xs leading-relaxed" style={{ color: C.secondary }}>{ins}</p>
              </div>
            ))}
          </div>

          {/* Summary bar */}
          <div className="grid grid-cols-3 gap-3 mt-4" style={{ borderTop: `1px solid ${C.border}`, paddingTop: 16 }}>
            <Pill label="Models" value={p.models?.length || 3} bg="rgba(79,110,247,.06)" color={C.accent} />
            <Pill label="Mining Methods" value="5" bg="rgba(5,150,105,.06)" color={C.green} />
            <Pill label="Rules Found" value={rules.length || 5} bg="rgba(217,119,6,.06)" color={C.amber} />
          </div>
        </Card>
      </div>

      {/* ══════════════════════════════════════════════════
          SECTION 5 — STATISTICAL ANALYSIS (NEW)
          ══════════════════════════════════════════════════ */}
      <Card>
        <SecHeader icon={Activity} title="Statistical Analysis of Fraud vs Normal"
          subtitle="Amount distribution comparison and class separability analysis"
          iconBg="rgba(220,38,38,.10)" iconColor={C.red} />

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Distribution bar */}
          <div className="lg:col-span-2">
            <p className="text-xs font-semibold mb-3" style={{ color: C.primary }}>
              Transaction Count per Amount Range
            </p>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart margin={{ top: 5, right: 10, left: 0, bottom: 5 }}
                data={[
                  { range: '$0–50',   normal: 142500, fraud: 58  },
                  { range: '$50–100', normal: 65200,  fraud: 89  },
                  { range: '$100–200',normal: 38700,  fraud: 134 },
                  { range: '$200–500',normal: 24800,  fraud: 142 },
                  { range: '$500–1K', normal: 9800,   fraud: 48  },
                  { range: '$1K+',    normal: 3807,   fraud: 21  },
                ]}>
                <CartesianGrid strokeDasharray="3 3" stroke={GRID} />
                <XAxis dataKey="range" tick={{ ...AX, fontSize: 10 }} axisLine={{ stroke: C.border }} tickLine={false} />
                <YAxis tick={AX} axisLine={{ stroke: C.border }} tickLine={false} />
                <Tooltip {...TT} />
                <Legend formatter={v => <span style={{ color: C.secondary, fontSize: 11 }}>{v}</span>} />
                <Bar dataKey="normal" name="Normal" fill={C.accent} radius={[4,4,0,0]} />
                <Bar dataKey="fraud"  name="Fraud"  fill={C.red}    radius={[4,4,0,0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Stats summary */}
          <div className="space-y-3">
            <p className="text-xs font-semibold" style={{ color: C.primary }}>Class Statistics</p>
            {[
              { label: 'Fraud Avg Amount', value: '$122.21', color: C.red,    bg: 'rgba(220,38,38,.06)' },
              { label: 'Normal Avg Amount',value: '$88.29',  color: C.green,  bg: 'rgba(5,150,105,.06)' },
              { label: 'Fraud Std Dev',    value: '$256.68', color: C.amber,  bg: 'rgba(217,119,6,.06)' },
              { label: 'Fraud Max Amount', value: '$2,125',  color: C.purple, bg: 'rgba(124,58,237,.06)' },
              { label: 'Class Imbalance',  value: '1.7%',    color: C.red,    bg: 'rgba(220,38,38,.06)' },
            ].map(s => (
              <div key={s.label} className="rounded-xl px-3 py-2.5 flex items-center justify-between"
                   style={{ background: s.bg, border: `1px solid ${s.color}22` }}>
                <span className="text-xs" style={{ color: C.secondary }}>{s.label}</span>
                <span className="text-sm font-bold" style={{ color: s.color }}>{s.value}</span>
              </div>
            ))}
          </div>
        </div>
      </Card>

      {/* ══════════════════════════════════════════════════
          SECTION 6 — PIPELINE SUMMARY (NEW)
          ══════════════════════════════════════════════════ */}
      <Card>
        <SecHeader icon={Zap} title="Data Mining Pipeline Summary"
          subtitle="End-to-end methodology: preprocessing → feature engineering → mining → evaluation"
          iconBg="rgba(79,110,247,.10)" iconColor={C.accent} />

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            { icon: Shield,    label: 'SMOTE Oversampling',   desc: 'Balanced fraud class from 1.7% → 50% for training', color: C.accent,  bg: 'rgba(79,110,247,.06)'   },
            { icon: Layers,    label: 'RobustScaler',         desc: 'Scaled Amount & Time features to remove outlier bias', color: C.green, bg: 'rgba(5,150,105,.06)'    },
            { icon: Brain,     label: 'SHAP Explainability',  desc: 'TreeExplainer used for model-agnostic feature importance', color: C.purple, bg: 'rgba(124,58,237,.06)' },
            { icon: BarChart3, label: 'Cross-Validation',     desc: '5-fold stratified CV ensured robust generalisation', color: C.amber,  bg: 'rgba(217,119,6,.06)'   },
          ].map(s => (
            <div key={s.label} className="rounded-xl p-4" style={{ background: s.bg, border: `1px solid ${s.color}22` }}>
              <div className="w-8 h-8 rounded-lg flex items-center justify-center mb-3"
                   style={{ background: `${s.color}18` }}>
                <s.icon className="w-4 h-4" style={{ color: s.color }} />
              </div>
              <p className="text-xs font-semibold mb-1" style={{ color: C.primary }}>{s.label}</p>
              <p className="text-[10px] leading-relaxed" style={{ color: C.secondary }}>{s.desc}</p>
            </div>
          ))}
        </div>
      </Card>

    </div>
  )
}

export default Analytics
