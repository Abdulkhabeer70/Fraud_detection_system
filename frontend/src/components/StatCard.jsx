import { TrendingUp, TrendingDown, Minus } from 'lucide-react'

const colorMap = {
  emerald: {
    icon: { background: 'rgba(16,185,129,0.10)', color: '#059669' },
    glow: 'glow-emerald',
    border: 'rgba(16,185,129,0.15)',
    value: '#059669',
  },
  red: {
    icon: { background: 'rgba(239,68,68,0.10)', color: '#dc2626' },
    glow: 'glow-red',
    border: 'rgba(239,68,68,0.15)',
    value: '#dc2626',
  },
  amber: {
    icon: { background: 'rgba(245,158,11,0.10)', color: '#d97706' },
    glow: 'glow-amber',
    border: 'rgba(245,158,11,0.15)',
    value: '#d97706',
  },
  blue: {
    icon: { background: 'rgba(79,110,247,0.10)', color: '#4f6ef7' },
    glow: 'glow-blue',
    border: 'rgba(79,110,247,0.15)',
    value: '#4f6ef7',
  },
}

function StatCard({ icon: Icon, title, value, change, color = 'emerald', loading = false }) {
  const scheme = colorMap[color] || colorMap.emerald

  if (loading) {
    return (
      <div className="glass-card p-6 animate-pulse">
        <div className="flex items-center justify-between mb-4">
          <div className="w-11 h-11 rounded-xl loading-shimmer" />
          <div className="w-14 h-5 rounded-lg loading-shimmer" />
        </div>
        <div className="w-24 h-3.5 rounded loading-shimmer mb-2.5" />
        <div className="w-32 h-7 rounded loading-shimmer" />
      </div>
    )
  }

  return (
    <div className={`glass-card-hover p-6 ${scheme.glow} animate-fade-in`}
         style={{ borderLeft: `3px solid ${scheme.border}` }}>
      <div className="flex items-center justify-between mb-4">
        <div className="w-11 h-11 rounded-xl flex items-center justify-center"
             style={scheme.icon}>
          <Icon className="w-5 h-5" />
        </div>
        {change !== undefined && change !== null && (
          <div className="flex items-center gap-1 text-xs font-medium px-2 py-1 rounded-lg"
               style={{
                 background: change > 0 ? 'rgba(16,185,129,0.08)' : change < 0 ? 'rgba(239,68,68,0.08)' : 'rgba(100,116,139,0.08)',
                 color: change > 0 ? '#059669' : change < 0 ? '#dc2626' : '#64748b',
               }}>
            {change > 0 ? <TrendingUp className="w-3 h-3" /> : change < 0 ? <TrendingDown className="w-3 h-3" /> : <Minus className="w-3 h-3" />}
            {Math.abs(change)}%
          </div>
        )}
      </div>
      <p className="text-xs font-semibold uppercase tracking-wider mb-1.5" style={{ color: '#8792a8' }}>
        {title}
      </p>
      <p className="text-2xl font-bold" style={{ color: scheme.value }}>{value}</p>
    </div>
  )
}

export default StatCard
