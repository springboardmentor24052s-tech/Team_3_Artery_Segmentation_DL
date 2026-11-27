export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'

async function request(path, options = {}) {
  const headers = new Headers(options.headers || {})
  const token = localStorage.getItem('access_token')
  if (token) headers.set('Authorization', `Bearer ${token}`)

  if (options.body && !(options.body instanceof FormData)) {
    headers.set('Content-Type', 'application/json')
  }

  let response
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...options,
      headers
    })
  } catch {
    throw new Error('Could not connect to the API. Check that the backend is running.')
  }

  if (response.status === 401 && token) {
    localStorage.removeItem('access_token')
    localStorage.removeItem('current_user')
    window.dispatchEvent(new Event('auth-expired'))
  }

  if (!response.ok) {
    const responseText = await response.text()
    let data = responseText
    if (responseText) {
      try {
        data = JSON.parse(responseText)
      } catch {
        data = responseText
      }
    }
    const detail = typeof data === 'object' && data !== null
      ? data.detail || data.message || `Request failed (${response.status})`
      : data || `Request failed (${response.status})`
    throw new Error(detail)
  }

  if (options.responseType === 'blob') return response.blob()
  if (response.status === 204) return null

  const responseText = await response.text()
  if (!responseText) return null
  let data
  try {
    data = JSON.parse(responseText)
  } catch {
    data = responseText
  }

  if (typeof data === 'object' && data?.error) {
    throw new Error(data.error)
  }

  return data
}

export function apiGet(path) {
  return request(path)
}

export function apiPost(path, body) {
  return request(path, {
    method: 'POST',
    body: body instanceof FormData ? body : JSON.stringify(body)
  })
}

export async function fetchImageUrl(path) {
  const blob = await request(path, { responseType: 'blob' })
  return URL.createObjectURL(blob)
}

export async function downloadReport(reportId) {
  const blob = await request(
    `/reports/${encodeURIComponent(reportId)}/download`,
    { responseType: 'blob' }
  )
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = `stenosis-report-${reportId}.json`
  link.click()
  URL.revokeObjectURL(url)
}
