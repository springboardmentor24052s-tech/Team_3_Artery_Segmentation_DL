
import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Activity,
  ArrowRight,
  Eye,
  EyeOff,
  LockKeyhole,
  Mail,
  ShieldCheck
} from 'lucide-react'

export default function Login() {
  const navigate = useNavigate()

  const [email, setEmail] = useState(
    localStorage.getItem('remember_email') || ''
  )
  const [password, setPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [rememberMe, setRememberMe] = useState(
    Boolean(localStorage.getItem('remember_email'))
  )
  const [error, setError] = useState('')

  const handleLogin = (e) => {
    e.preventDefault()
    setError('')

    // Validate email and password
    if (!email.trim() || !password.trim()) {
      setError('Please enter both email and password.')
      return
    }

    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email.trim())) {
      setError('Please enter a valid email address.')
      return
    }

    // Remember email if selected
    if (rememberMe) {
      localStorage.setItem('remember_email', email.trim())
    } else {
      localStorage.removeItem('remember_email')
    }

    // Temporary frontend-only demo authentication
    localStorage.setItem(
      'demoUser',
      JSON.stringify({
        email: email.trim()
      })
    )

    // Navigate to dashboard
    navigate('/dashboard', { replace: true })
  }

  return (
    <div className="login-page">

      <div className="login-brand">
        <div className="login-brand-icon">
          <Activity size={25} />
        </div>

        <span>
          ARTERY<span className="brand-accent">AI</span>
        </span>
      </div>

      <div className="login-card">

        <div className="login-icon">
          <Activity size={27} />
        </div>

        <span className="eyebrow">
          RESEARCH WORKSPACE
        </span>

        <h1>Welcome back</h1>

        <p className="login-subtitle">
          Sign in to access your artery imaging workspace.
        </p>

        <form onSubmit={handleLogin}>

          {/* Email */}
          <div className="form-group">
            <label htmlFor="email">
              Email address
            </label>

            <div className="login-input-wrap">
              <Mail size={18} />

              <input
                id="email"
                type="email"
                placeholder="you@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                autoComplete="email"
              />
            </div>
          </div>

          {/* Password */}
          <div className="form-group">

            <div className="password-label-row">
              <label htmlFor="password">
                Password
              </label>

              <button
                type="button"
                className="text-button"
                onClick={() =>
                  setError(
                    'Password recovery will be available after backend integration.'
                  )
                }
              >
                Forgot password?
              </button>
            </div>

            <div className="login-input-wrap">
              <LockKeyhole size={18} />

              <input
                id="password"
                type={showPassword ? 'text' : 'password'}
                placeholder="Enter your password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                autoComplete="current-password"
              />

              <button
                type="button"
                className="password-toggle"
                onClick={() => setShowPassword(!showPassword)}
                aria-label={
                  showPassword ? 'Hide password' : 'Show password'
                }
              >
                {showPassword
                  ? <EyeOff size={18} />
                  : <Eye size={18} />
                }
              </button>
            </div>
          </div>

          {/* Remember email */}
          <label className="remember-row">
            <input
              type="checkbox"
              checked={rememberMe}
              onChange={(e) => setRememberMe(e.target.checked)}
            />

            <span>Remember my email</span>
          </label>

          {/* Error message */}
          {error && (
            <div className="login-error" role="alert">
              {error}
            </div>
          )}

          {/* Login button */}
          <button
            type="submit"
            className="login-submit"
          >
            Sign in
            <ArrowRight size={18} />
          </button>

        </form>

        <div className="login-security">
          <ShieldCheck size={16} />
          <span>Secure research workspace</span>
        </div>

      </div>

      <p className="login-disclaimer">
        Artery Segmentation & Stenosis Detection · Research Project
      </p>

    </div>
  )
}
