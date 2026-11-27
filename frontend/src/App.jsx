import { useEffect, useState } from 'react'
import { Navigate, Route, Routes } from 'react-router-dom'

import AppLayout from './layouts/AppLayout'
import Login from './Pages/Login'
import Dashboard from './Pages/Dashboard'
import Cases from './Pages/Cases'
import Stenosis from './Pages/Stenosis'
import Segmentation from './Pages/Segmentation'
import ModelAnalytics from './Pages/ModelAnalytics'
import Reports from './Pages/Reports'

function ProtectedLayout() {
  const [authenticated, setAuthenticated] = useState(
    Boolean(localStorage.getItem('access_token'))
  )

  useEffect(() => {
    const expireSession = () => setAuthenticated(false)
    window.addEventListener('auth-expired', expireSession)
    return () => window.removeEventListener('auth-expired', expireSession)
  }, [])

  if (!authenticated) {
    return <Navigate to="/login" replace />
  }

  return <AppLayout />
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />

      <Route element={<ProtectedLayout />}>
        <Route path="/" element={<Navigate to="/overview" replace />} />

        <Route path="/overview" element={<Dashboard />} />
        <Route path="/dashboard" element={<Navigate to="/overview" replace />} />

        <Route path="/cases" element={<Cases />} />
        <Route path="/segmentation" element={<Segmentation />} />
        <Route path="/stenosis" element={<Stenosis />} />

        <Route path="/model-analytics" element={<ModelAnalytics />} />
        <Route path="/analytics" element={<Navigate to="/model-analytics" replace />} />

        <Route path="/reports" element={<Reports />} />
      </Route>

      <Route path="*" element={<Navigate to="/overview" replace />} />
    </Routes>
  )
}