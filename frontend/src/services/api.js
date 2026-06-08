import axios from 'axios'

const api = axios.create({
  baseURL: 'http://localhost:8000',
  timeout: 15000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// ── Stats ──────────────────────────────────────
export async function getStats() {
  const response = await api.get('/stats')
  return response.data
}

// ── Predict ────────────────────────────────────
export async function predict(data) {
  const response = await api.post('/predict', data)
  return response.data
}

// ── Model Performance ──────────────────────────
export async function getModelPerformance() {
  const response = await api.get('/model-performance')
  return response.data
}

// ── Mining Results ─────────────────────────────
export async function getMiningResults() {
  const response = await api.get('/mining-results')
  return response.data
}

// ── Prediction History ─────────────────────────
export async function getPredictionHistory() {
  const response = await api.get('/predictions/history')
  return response.data
}

export default api
