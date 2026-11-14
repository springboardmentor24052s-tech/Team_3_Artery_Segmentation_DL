
const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'

// Handle FastAPI responses
async function handleResponse(response) {
  const contentType = response.headers.get('content-type') || ''

  let data

  if (contentType.includes('application/json')) {
    data = await response.json()
  } else {
    data = await response.text()
  }

  if (!response.ok) {
    const detail =
      typeof data === 'object'
        ? data?.detail || data?.message || data?.error
        : data

    throw new Error(
      detail || `Request failed with status ${response.status}`
    )
  }

  if (typeof data === 'object' && data?.error) {
    throw new Error(data.error)
  }

  return data
}

// 1. Upload medical image
export async function processSegmentation(file) {
  if (!file) {
    throw new Error('Please select an image first.')
  }

  const formData = new FormData()
  formData.append('file', file)

  const response = await fetch(
    `${API_BASE_URL}/upload-image`,
    {
      method: 'POST',
      body: formData
    }
  )

  return handleResponse(response)
}

// 2. Extract vessel centerline
export async function extractCenterline(maskFile) {
  if (!maskFile) {
    throw new Error('Please provide a segmentation mask.')
  }

  const formData = new FormData()
  formData.append('file', maskFile)

  const response = await fetch(
    `${API_BASE_URL}/extract-centerline`,
    {
      method: 'POST',
      body: formData
    }
  )

  return handleResponse(response)
}

// 3. Analyze vessel width
export async function analyzeVesselWidth(maskFile) {
  if (!maskFile) {
    throw new Error('Please provide a binary vessel mask.')
  }

  const formData = new FormData()
  formData.append('file', maskFile)

  const response = await fetch(
    `${API_BASE_URL}/analyze-vessel-width`,
    {
      method: 'POST',
      body: formData
    }
  )

  return handleResponse(response)
}
