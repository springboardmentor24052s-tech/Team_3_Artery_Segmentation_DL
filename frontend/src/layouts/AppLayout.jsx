import { useState } from 'react'
import { NavLink, Outlet, useNavigate } from 'react-router-dom'
import {
  BrainCircuit,
  ChartNoAxesCombined,
  FileText,
  HeartPulse,
  LayoutDashboard,
  LogOut,
  Menu,
  ShieldCheck,
  Users,
  Activity,
  X
} from 'lucide-react'

const navigation = [
  { label: 'Overview', path: '/overview', icon: LayoutDashboard },
  { label: 'Patients & Cases', path: '/cases', icon: Users },
  { label: 'Segmentation', path: '/segmentation', icon: Activity },
  { label: 'Stenosis Analysis', path: '/stenosis', icon: BrainCircuit },
  { label: 'Model Analytics', path: '/model-analytics', icon: ChartNoAxesCombined },
  { label: 'Reports', path: '/reports', icon: FileText }
]

export default function AppLayout() {
  const [mobileOpen, setMobileOpen] = useState(false)
  const navigate = useNavigate()

  const closeMobileMenu = () => {
    setMobileOpen(false)
  }

  const handleLogout = () => {
    localStorage.removeItem('access_token')
    localStorage.removeItem('demoUser')
    closeMobileMenu()
    navigate('/login', { replace: true })
  }

  return (
    <div className="app-shell">
      {mobileOpen && (
        <button
          type="button"
          className="mobile-overlay"
          aria-label="Close navigation menu"
          onClick={closeMobileMenu}
        />
      )}

      <aside
        id="app-sidebar"
        className={`sidebar ${mobileOpen ? 'sidebar-open' : ''}`}
        aria-label="Main navigation"
      >
        <div className="brand">
          <div className="brand-icon">
            <HeartPulse size={23} />
          </div>

          <div>
            <h2>
              Artery<span>AI</span>
            </h2>
            <p>Medical Intelligence</p>
          </div>

          <button
            type="button"
            className="sidebar-close"
            onClick={closeMobileMenu}
            aria-label="Close menu"
          >
            <X size={19} />
          </button>
        </div>

        <div className="sidebar-section-label">WORKSPACE</div>

        <nav className="sidebar-nav">
          {navigation.map(({ label, path, icon: Icon }) => (
            <NavLink
              key={path}
              to={path}
              end
              onClick={closeMobileMenu}
              className={({ isActive }) =>
                `nav-link ${isActive ? 'nav-link-active' : ''}`
              }
            >
              <Icon size={19} />
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="sidebar-status">
            <div className="status-indicator" />
            <div>
              <strong>AI Engine</strong>
              <span>Milestone 3 Active</span>
            </div>
          </div>

          <button
            type="button"
            className="logout-link"
            onClick={handleLogout}
          >
            <LogOut size={18} />
            <span>Sign out</span>
          </button>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div className="topbar-left">
            <button
              type="button"
              className="mobile-menu"
              onClick={() => setMobileOpen(true)}
              aria-label="Open navigation menu"
              aria-expanded={mobileOpen}
              aria-controls="app-sidebar"
            >
              <Menu size={21} />
            </button>

            <div>
              <span className="topbar-eyebrow">MEDICAL AI PLATFORM</span>
              <p>Artery Segmentation & Stenosis Detection</p>
            </div>
          </div>

          <div className="topbar-right">
            <div className="secure-label">
              <ShieldCheck size={16} />
              <span>Clinical Decision Support</span>
            </div>

            <div className="user-avatar" title="User profile">
              KP
            </div>
          </div>
        </header>

        <section className="page-content">
          <Outlet />
        </section>
      </main>
    </div>
  )
}