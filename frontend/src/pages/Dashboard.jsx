import { useState, useEffect } from 'react'
import { 
  CreditCard, AlertTriangle, Percent, DollarSign, 
  Activity, RefreshCw, Clock
} from 'lucide-react'
import { 
  PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend,
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
} from 'recharts'
import StatCard from '../components/StatCard'
import RiskBadge from '../components/RiskBadge'
import LoadingSpinner from '../components/LoadingSpinner'
import { getStats, getPredictionHistory } from '../services/api'

// ── Fallback data ──────────────────────────────
const FALLBACK_STATS = {
  total_transactions: 284807,
  fraud_count: 492,
  fraud_percentage: 0.173,
  avg_amount: 88.35,
  total_amount: 25166237.82,
  normal_count: 284315,
}

const FALLBACK_HISTORY = [
  { id: 1, amount: 149.62, prediction: 'Legit', probability: 0.02, risk_level: 'low',    timestamp: '2026-05-21T15:30:00' },
  { id: 2, amount: 2.69,   prediction: 'Legit', probability: 0.05, risk_level: 'low',    timestamp: '2026-05-21T15:28:00' },
  { id: 3, amount: 378.66, prediction: 'Fraud', probability: 0.94, risk_level: 'high',   timestamp: '2026-05-21T15:25:00' },
  { id: 4, amount: 123.50, prediction: 'Legit', probability: 0.12, risk_level: 'low',    timestamp: '2026-05-21T15:20:00' },
  { id: 5, amount: 69.99,  prediction: 'Legit', probability: 0.38, risk_level: 'medium', timestamp: '2026-05-21T15:15:00' },
  { id: 6, amount: 999.99, prediction: 'Fraud', probability: 0.87, risk_level: 'high',   timestamp: '2026-05-21T15:10:00' },
  { id: 7, amount: 45.20,  prediction: 'Legit', probability: 0.04, risk_level: 'low',    timestamp: '2026-05-21T15:05:00' },
  { id: 8, amount: 520.00, prediction: 'Fraud', probability: 0.91, risk_level: 'high',   timestamp: '2026-05-21T15:00:00' },
]

const AMOUNT_DIST = [
  { range: '$0-50',   count: 142500, fill: '#4f6ef7' },
  { range: '$50-100', count: 65200,  fill: '#6a5ef5' },
  { range: '$100-200',count: 38700,  fill: '#a855f7' },
  { range: '$200-500',count: 24800,  fill: '#059669' },
  { range: '$500-1K', count: 9800,   fill: '#d97706' },
  { range: '$1K-5K',  count: 3200,   fill: '#ea580c' },
  { range: '$5K+',    count: 607,    fill: '#dc2626' },
]

const PIE_COLORS = ['#4f6ef7', '#ef4444']

function CustomPieTooltip({ active, payload }) {
  if (active && payload && payload.length) {
    return (
      <div className="glass-card px-4 py-3">
        <p className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>{payload[0].name}</p>
        <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>{payload[0].value.toLocaleString()} transactions</p>
        <p className="text-xs" style={{ color: 'var(--text-label)' }}>
          {((payload[0].value / (FALLBACK_STATS.total_transactions)) * 100).toFixed(3)}%
        </p>
      </div>
    )
  }
  return null
}

// Chart axis / grid styles
const AXIS_TICK = { fill: '#8792a8', fontSize: 11 }
const AXIS_LINE = { stroke: '#e2e6ee' }
const GRID_STROKE = '#eef0f5'

function Dashboard() {
  const [stats, setStats] = useState(null)
  const [history, setHistory] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const fetchData = async () => {
    setLoading(true)
    setError(null)
    try {
      const [statsData, historyData] = await Promise.allSettled([
        getStats(),
        getPredictionHistory(),
      ])
      setStats(statsData.status === 'fulfilled' ? statsData.value : FALLBACK_STATS)
      setHistory(
        historyData.status === 'fulfilled'
          ? (Array.isArray(historyData.value) ? historyData.value : historyData.value?.predictions || FALLBACK_HISTORY)
          : FALLBACK_HISTORY
      )
    } catch {
      setStats(FALLBACK_STATS)
      setHistory(FALLBACK_HISTORY)
      setError('Using demo data — backend unavailable')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetchData() }, [])

  const s = stats || FALLBACK_STATS
  const h = history || FALLBACK_HISTORY

  const pieData = [
    { name: 'Normal', value: s.normal_count || (s.total_transactions - s.fraud_count) },
    { name: 'Fraud',  value: s.fraud_count },
  ]

  return (
    <div className="space-y-7 animate-fade-in">
      {/* ── Header ── */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight" style={{ color: 'var(--text-primary)' }}>
            Dashboard
          </h1>
          <p className="text-sm mt-0.5" style={{ color: 'var(--text-secondary)' }}>
            Credit card transaction monitoring overview
          </p>
        </div>
        <button onClick={fetchData} className="btn-secondary flex items-center gap-2 text-sm">
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {/* ── Error banner ── */}
      {error && (
        <div className="rounded-xl px-4 py-3 flex items-center gap-3 animate-slide-up"
             style={{ background: 'rgba(245,158,11,0.08)', border: '1px solid rgba(245,158,11,0.20)' }}>
          <AlertTriangle className="w-4 h-4 flex-shrink-0" style={{ color: '#d97706' }} />
          <p className="text-sm" style={{ color: '#92400e' }}>{error}</p>
        </div>
      )}

      {/* ── Stat Cards ── */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-5">
        <StatCard icon={CreditCard}    title="Total Transactions" value={loading ? '...' : s.total_transactions?.toLocaleString()} color="blue"    loading={loading} />
        <StatCard icon={AlertTriangle} title="Fraud Detected"     value={loading ? '...' : s.fraud_count?.toLocaleString()}        color="red"     loading={loading} />
        <StatCard icon={Percent}       title="Fraud Rate"         value={loading ? '...' : `${(s.fraud_percentage || ((s.fraud_count / s.total_transactions) * 100))?.toFixed(3)}%`} color="amber" loading={loading} />
        <StatCard icon={DollarSign}    title="Avg Transaction"    value={loading ? '...' : `$${s.avg_amount?.toFixed(2)}`}         color="emerald" loading={loading} />
      </div>

      {/* ── Charts Row ── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Pie Chart */}
        <div className="glass-card p-6 animate-slide-up">
          <div className="flex items-center gap-2 mb-5">
            <div className="w-7 h-7 rounded-lg flex items-center justify-center"
                 style={{ background: 'rgba(79,110,247,0.10)' }}>
              <Activity className="w-4 h-4" style={{ color: '#4f6ef7' }} />
            </div>
            <h2 className="text-base font-semibold" style={{ color: 'var(--text-primary)' }}>
              Transaction Distribution
            </h2>
          </div>
          {loading ? (
            <div className="h-64 loading-shimmer rounded-xl" />
          ) : (
            <ResponsiveContainer width="100%" height={260}>
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%" cy="50%"
                  innerRadius={70} outerRadius={105}
                  paddingAngle={4}
                  dataKey="value"
                  stroke="none"
                >
                  {pieData.map((_, index) => (
                    <Cell key={`cell-${index}`} fill={PIE_COLORS[index]} />
                  ))}
                </Pie>
                <Tooltip content={<CustomPieTooltip />} />
                <Legend
                  verticalAlign="bottom"
                  formatter={(value) => (
                    <span style={{ color: 'var(--text-secondary)', fontSize: 12 }}>{value}</span>
                  )}
                />
              </PieChart>
            </ResponsiveContainer>
          )}
          <div className="grid grid-cols-2 gap-3 mt-3">
            <div className="rounded-xl p-3 text-center"
                 style={{ background: 'rgba(79,110,247,0.06)', border: '1px solid rgba(79,110,247,0.12)' }}>
              <p className="text-xs mb-1" style={{ color: 'var(--text-label)' }}>Normal</p>
              <p className="text-lg font-bold" style={{ color: '#4f6ef7' }}>
                {((1 - (s.fraud_count / s.total_transactions)) * 100).toFixed(2)}%
              </p>
            </div>
            <div className="rounded-xl p-3 text-center"
                 style={{ background: 'rgba(239,68,68,0.06)', border: '1px solid rgba(239,68,68,0.12)' }}>
              <p className="text-xs mb-1" style={{ color: 'var(--text-label)' }}>Fraud</p>
              <p className="text-lg font-bold" style={{ color: '#dc2626' }}>
                {((s.fraud_count / s.total_transactions) * 100).toFixed(3)}%
              </p>
            </div>
          </div>
        </div>

        {/* Bar Chart */}
        <div className="glass-card p-6 animate-slide-up">
          <div className="flex items-center gap-2 mb-5">
            <div className="w-7 h-7 rounded-lg flex items-center justify-center"
                 style={{ background: 'rgba(5,150,105,0.10)' }}>
              <DollarSign className="w-4 h-4" style={{ color: '#059669' }} />
            </div>
            <h2 className="text-base font-semibold" style={{ color: 'var(--text-primary)' }}>
              Amount Distribution
            </h2>
          </div>
          {loading ? (
            <div className="h-64 loading-shimmer rounded-xl" />
          ) : (
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={AMOUNT_DIST} margin={{ top: 5, right: 16, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={GRID_STROKE} />
                <XAxis dataKey="range" tick={AXIS_TICK} axisLine={AXIS_LINE} tickLine={false} />
                <YAxis tick={AXIS_TICK} axisLine={AXIS_LINE} tickLine={false} />
                <Tooltip
                  contentStyle={{
                    background: '#ffffff',
                    border: '1px solid #e2e6ee',
                    borderRadius: '12px',
                    boxShadow: '0 8px 24px rgba(15,20,40,0.10)',
                  }}
                  labelStyle={{ color: 'var(--text-primary)', fontWeight: 600 }}
                  itemStyle={{ color: 'var(--text-secondary)' }}
                />
                <Bar dataKey="count" radius={[6, 6, 0, 0]}>
                  {AMOUNT_DIST.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>

      {/* ── Recent Predictions Table ── */}
      <div className="glass-card p-6 animate-slide-up">
        <div className="flex items-center justify-between mb-5">
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-lg flex items-center justify-center"
                 style={{ background: 'rgba(79,110,247,0.10)' }}>
              <Clock className="w-4 h-4" style={{ color: '#4f6ef7' }} />
            </div>
            <h2 className="text-base font-semibold" style={{ color: 'var(--text-primary)' }}>
              Recent Predictions
            </h2>
          </div>
          <span className="text-xs px-2.5 py-1 rounded-full font-medium"
                style={{ background: 'rgba(79,110,247,0.08)', color: '#4f6ef7' }}>
            {h.length} records
          </span>
        </div>

        {loading ? (
          <div className="space-y-3">
            {[1,2,3,4].map(i => <div key={i} className="loading-shimmer rounded-xl h-12" />)}
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border)' }}>
                  {['#', 'Amount', 'Prediction', 'Probability', 'Risk Level', 'Time'].map(col => (
                    <th key={col} className="text-left text-[10px] font-semibold uppercase tracking-wider py-2.5 px-4"
                        style={{ color: 'var(--text-label)' }}>
                      {col}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {h.slice(0, 8).map((item, idx) => (
                  <tr key={item.id || idx}
                      className="transition-colors duration-150"
                      style={{ borderBottom: '1px solid var(--border)' }}
                      onMouseEnter={e => e.currentTarget.style.background = 'rgba(79,110,247,0.03)'}
                      onMouseLeave={e => e.currentTarget.style.background = 'transparent'}>
                    <td className="py-3 px-4 text-sm" style={{ color: 'var(--text-label)' }}>{idx + 1}</td>
                    <td className="py-3 px-4">
                      <span className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
                        ${Number(item.amount).toFixed(2)}
                      </span>
                    </td>
                    <td className="py-3 px-4">
                      <span className="inline-flex items-center gap-1.5 text-sm font-medium"
                            style={{ color: item.prediction === 'Fraud' ? '#dc2626' : '#059669' }}>
                        <span className="w-1.5 h-1.5 rounded-full"
                              style={{ background: item.prediction === 'Fraud' ? '#dc2626' : '#059669' }} />
                        {item.prediction}
                      </span>
                    </td>
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-2">
                        <div className="w-16 h-1.5 rounded-full overflow-hidden"
                             style={{ background: '#e2e6ee' }}>
                          <div
                            className="h-full rounded-full transition-all duration-500"
                            style={{
                              width: `${item.probability * 100}%`,
                              background: item.probability > 0.7 ? '#dc2626' : item.probability > 0.4 ? '#d97706' : '#059669',
                            }}
                          />
                        </div>
                        <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                          {(item.probability * 100).toFixed(1)}%
                        </span>
                      </div>
                    </td>
                    <td className="py-3 px-4"><RiskBadge level={item.risk_level} /></td>
                    <td className="py-3 px-4 text-xs" style={{ color: 'var(--text-label)' }}>
                      {item.timestamp ? new Date(item.timestamp).toLocaleTimeString() : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}

export default Dashboard
