import { ShieldCheck, ShieldAlert, AlertTriangle } from 'lucide-react'

function RiskBadge({ level }) {
  const normalized = (level || 'low').toLowerCase()

  const config = {
    low: {
      label: 'Low Risk',
      icon: ShieldCheck,
      bg: 'rgba(5,150,105,0.08)',
      border: 'rgba(5,150,105,0.20)',
      color: '#059669',
      dot: '#059669',
    },
    medium: {
      label: 'Medium Risk',
      icon: AlertTriangle,
      bg: 'rgba(217,119,6,0.08)',
      border: 'rgba(217,119,6,0.20)',
      color: '#d97706',
      dot: '#d97706',
    },
    high: {
      label: 'High Risk',
      icon: ShieldAlert,
      bg: 'rgba(220,38,38,0.08)',
      border: 'rgba(220,38,38,0.20)',
      color: '#dc2626',
      dot: '#dc2626',
    },
  }

  const { label, icon: Icon, bg, border, color, dot } = config[normalized] || config.low

  return (
    <span
      className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold"
      style={{ background: bg, border: `1px solid ${border}`, color }}
    >
      <span className="relative flex h-1.5 w-1.5">
        <span className="relative inline-flex rounded-full h-1.5 w-1.5" style={{ background: dot }} />
      </span>
      <Icon className="w-3 h-3" />
      {label}
    </span>
  )
}

export default RiskBadge
