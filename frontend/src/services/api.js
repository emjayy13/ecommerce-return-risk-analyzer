const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000/api'
).replace(/\/$/, '')

async function request(path) {
  const response = await fetch(`${API_BASE_URL}${path}`)
  const payload = await response.json()

  if (!response.ok || !payload.success) {
    throw new Error(payload.message || 'Unable to load data from the API')
  }

  return payload.data
}

export function getCustomers(filters = {}) {
  const searchParams = new URLSearchParams()

  if (filters.risk) {
    searchParams.set('risk', filters.risk)
  }

  if (filters.category) {
    searchParams.set('category', filters.category)
  }

  if (filters.returnRatio !== '' && filters.returnRatio !== undefined) {
    searchParams.set('returnRatio', filters.returnRatio)
  }

  const queryString = searchParams.toString()
  const path = queryString ? `/customers?${queryString}` : '/customers'

  return request(path)
}

export function getDashboardSummary() {
  return request('/dashboard/summary')
}

export function getCategoryStats() {
  return request('/dashboard/categories')
}

export function getReturnTrends() {
  return request('/dashboard/trends')
}

export function getCustomerById(customerId) {
  return request(`/customers/${encodeURIComponent(customerId)}`)
}
