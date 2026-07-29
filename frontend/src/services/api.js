const API_BASE_URL = 'http://localhost:5000/api'

async function request(path) {
  const response = await fetch(`${API_BASE_URL}${path}`)
  const payload = await response.json()

  if (!response.ok || !payload.success) {
    throw new Error(payload.message || 'Unable to load data from the API')
  }

  return payload.data
}

export function getCustomers() {
  return request('/customers')
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