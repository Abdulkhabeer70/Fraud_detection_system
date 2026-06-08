import { NavLink } from 'react-router-dom'
import { 
  LayoutDashboard, 
  ScanSearch, 
  BarChart3, 
  Shield, 
  Zap,
  ChevronRight 
} from 'lucide-react'

const navItems = [
  { to: '/', label: 'Dashboard', icon: LayoutDashboard, desc: 'Overview & Stats' },
  { to: '/predict', label: 'Predict', icon: ScanSearch, desc: 'Analyze Transaction' },
  { to: '/analytics', label: 'Analytics', icon: BarChart3, desc: 'Data Mining Results' },
]

function Sidebar() {
  return (
    <aside className="fixed left-0 top-0 h-screen w-64 glass-sidebar z-50 flex flex-col">
      {/* Logo */}
      <div className="px-6 py-6" style={{ borderBottom: '1px solid rgba(255,255,255,0.07)' }}>
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl flex items-center justify-center shadow-lg"
               style={{ background: 'linear-gradient(135deg, #4f6ef7 0%, #a855f7 100%)' }}>
            <Shield className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-base font-bold tracking-tight" style={{ color: '#ffffff' }}>FraudGuard</h1>
            <p className="text-[10px] font-semibold uppercase tracking-widest" style={{ color: '#7c8ef7' }}>AI Detection</p>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-3 py-6 space-y-1">
        <p className="px-3 mb-3 text-[10px] font-semibold uppercase tracking-widest" style={{ color: '#4a5270' }}>
          Navigation
        </p>
        {navItems.map(({ to, label, icon: Icon, desc }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              `group flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all duration-200 ${
                isActive ? 'nav-active' : 'nav-inactive'
              }`
            }
            style={({ isActive }) => isActive
              ? { background: 'rgba(79,110,247,0.15)', border: '1px solid rgba(79,110,247,0.25)' }
              : { border: '1px solid transparent' }
            }
          >
            {({ isActive }) => (
              <>
                <div className="w-8 h-8 rounded-lg flex items-center justify-center transition-all duration-200"
                     style={isActive
                       ? { background: 'rgba(79,110,247,0.2)', color: '#7c8ef7' }
                       : { background: 'rgba(255,255,255,0.05)', color: '#6b7590' }
                     }>
                  <Icon className="w-[17px] h-[17px]" />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-semibold"
                     style={{ color: isActive ? '#a5b4fc' : '#c8d0e8' }}>
                    {label}
                  </p>
                  <p className="text-[10px] truncate" style={{ color: '#4a5270' }}>{desc}</p>
                </div>
                <ChevronRight className="w-3.5 h-3.5 transition-all duration-200"
                              style={{ color: isActive ? '#7c8ef7' : 'transparent', opacity: isActive ? 1 : 0 }} />
              </>
            )}
          </NavLink>
        ))}
      </nav>

      {/* Footer */}
      <div className="px-4 py-4" style={{ borderTop: '1px solid rgba(255,255,255,0.07)' }}>
        <div className="rounded-xl p-3" style={{ background: 'rgba(79,110,247,0.10)', border: '1px solid rgba(79,110,247,0.15)' }}>
          <div className="flex items-center gap-2 mb-1.5">
            <Zap className="w-3.5 h-3.5" style={{ color: '#f59e0b' }} />
            <span className="text-xs font-semibold" style={{ color: '#c8d0e8' }}>Data Mining</span>
          </div>
          <p className="text-[10px] leading-relaxed" style={{ color: '#4a5270' }}>
            Powered by XGBoost, Random Forest & advanced anomaly detection
          </p>
        </div>
      </div>
    </aside>
  )
}

export default Sidebar
