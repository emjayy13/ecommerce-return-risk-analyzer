import { useEffect, useState } from 'react'
import './App.css'
import CategoryReturnsChart from './components/CategoryReturnsChart.jsx'
import CustomerDetailModal from './components/CustomerDetailModal.jsx'
import CustomerTable from './components/CustomerTable.jsx'
import ReturnTrendChart from './components/ReturnTrendChart.jsx'
import RiskSegmentsChart from './components/RiskSegmentsChart.jsx'
import SummaryCard from './components/SummaryCard.jsx'
import {
  getCategoryStats,
  getCustomerById,
  getCustomers,
  getDashboardSummary,
  getReturnTrends,
} from './services/api.js'

function normalizeCustomer(customer) {
  const totalOrders = customer.totalOrders ?? 0
  const totalReturns = customer.totalReturns ?? 0

  return {
    id: customer._id,
    name: customer.name,
    email: customer.email,
    totalOrders,
    totalReturns,
    returnRatio: totalOrders === 0 ? 0 : totalReturns / totalOrders,
    riskScore: customer.riskScore,
    riskLevel: customer.riskLevel,
    flaggedBefore: customer.flaggedBefore,
  }
}

function normalizeCustomerDetail(detail) {
  const customer = normalizeCustomer(detail.customer)
  const returnRecords = detail.returns ?? []
  const categories = [
    ...new Set(returnRecords.map((returnItem) => returnItem.category)),
  ]
  const flags = []

  if (customer.flaggedBefore) {
    flags.push('Customer was previously flagged')
  }

  if (returnRecords.some((returnItem) => returnItem.mismatchFlag)) {
    flags.push('Item mismatch recorded')
  }

  return {
    ...customer,
    category:
      categories.length > 0 ? categories.join(', ') : 'No category recorded',
    flags,
    returnHistory: returnRecords.map((returnItem) => ({
      id: returnItem._id,
      date: returnItem.returnedAt.slice(0, 10),
      reason: returnItem.reason,
    })),
  }
}

function buildRiskLevelData(customerList) {
  const countsByRiskLevel = {}

  customerList.forEach((customer) => {
    if (countsByRiskLevel[customer.riskLevel] === undefined) {
      countsByRiskLevel[customer.riskLevel] = 0
    }

    countsByRiskLevel[customer.riskLevel] += 1
  })

  return Object.entries(countsByRiskLevel).map(([name, value]) => ({
    name,
    value,
  }))
}

function App() {
  const [customers, setCustomers] = useState([])
  const [allCustomers, setAllCustomers] = useState([])
  const [summary, setSummary] = useState({
    totalReturns: 0,
    avgReturnRatio: 0,
    highRiskCustomers: 0,
  })
  const [categoryReturnData, setCategoryReturnData] = useState([])
  const [returnTrendData, setReturnTrendData] = useState([])
  const [showHighRiskOnly, setShowHighRiskOnly] = useState(false)
  const [selectedCategory, setSelectedCategory] = useState('All')
  const [minimumReturnRatio, setMinimumReturnRatio] = useState(0)
  const [selectedCustomer, setSelectedCustomer] = useState(null)
  const [isLoadingDashboard, setIsLoadingDashboard] = useState(true)
  const [dashboardError, setDashboardError] = useState('')
  const [isLoadingCustomers, setIsLoadingCustomers] = useState(false)
  const [isLoadingProfile, setIsLoadingProfile] = useState(false)
  const [interactionError, setInteractionError] = useState('')

  useEffect(() => {
    async function loadOverallDashboardData() {
      try {
        setIsLoadingDashboard(true)
        setDashboardError('')

        const [customerData, summaryData, categoryData, trendData] =
          await Promise.all([
            getCustomers(),
            getDashboardSummary(),
            getCategoryStats(),
            getReturnTrends(),
          ])

        setAllCustomers(customerData.map(normalizeCustomer))
        setSummary(summaryData)
        setCategoryReturnData(
          categoryData.map((item) => ({
            category: item.category,
            returns: item.count,
          })),
        )
        setReturnTrendData(trendData)
      } catch (error) {
        setDashboardError(error.message || 'Unable to load dashboard data.')
      } finally {
        setIsLoadingDashboard(false)
      }
    }

    loadOverallDashboardData()
  }, [])

  useEffect(() => {
    async function loadFilteredCustomers() {
      try {
        setIsLoadingCustomers(true)
        setInteractionError('')

        const customerData = await getCustomers({
          risk: showHighRiskOnly ? 'High' : '',
          category: selectedCategory === 'All' ? '' : selectedCategory,
          returnRatio:
            minimumReturnRatio > 0 ? minimumReturnRatio / 100 : '',
        })

        setCustomers(customerData.map(normalizeCustomer))
      } catch (error) {
        setCustomers([])
        setInteractionError(
          error.message || 'Unable to load filtered customers.',
        )
      } finally {
        setIsLoadingCustomers(false)
      }
    }

    loadFilteredCustomers()
  }, [showHighRiskOnly, selectedCategory, minimumReturnRatio])

  async function handleViewProfile(customer) {
    try {
      setIsLoadingProfile(true)
      setInteractionError('')

      const detail = await getCustomerById(customer.id)
      setSelectedCustomer(normalizeCustomerDetail(detail))
    } catch (error) {
      setInteractionError(
        error.message || 'Unable to load the customer profile.',
      )
    } finally {
      setIsLoadingProfile(false)
    }
  }

  const riskSegmentData = buildRiskLevelData(allCustomers)
  const summaryMetrics = [
    {
      label: 'Total Returns',
      value: summary.totalReturns.toLocaleString(),
      helperText: 'Across all customers',
    },
    {
      label: 'Average Return Ratio',
      value: `${(summary.avgReturnRatio * 100).toFixed(1)}%`,
      helperText: 'Across all customers',
    },
    {
      label: 'High-Risk Customers',
      value: summary.highRiskCustomers,
      helperText: 'Risk level marked High',
    },
  ]

  return (
    <main>
      <header>
        <p>Operations Dashboard</p>
        <h1>Customer Return Risk Analyzer</h1>
        <p>Monitor customer return behaviour and identify potential risk.</p>
      </header>

      {(isLoadingDashboard ||
        isLoadingCustomers ||
        isLoadingProfile ||
        dashboardError ||
        interactionError) && (
        <div className="dashboard-status" aria-live="polite">
          {isLoadingDashboard && <p>Loading dashboard data…</p>}
          {!isLoadingDashboard && isLoadingCustomers && (
            <p>Updating customer list…</p>
          )}
          {isLoadingProfile && <p>Loading customer profile…</p>}
          {dashboardError && (
            <p className="dashboard-status__error" role="alert">
              {dashboardError}
            </p>
          )}
          {interactionError && (
            <p className="dashboard-status__error" role="alert">
              {interactionError}
            </p>
          )}
        </div>
      )}

      <section aria-labelledby="overview-heading">
        <h2 id="overview-heading">Return Overview</h2>

        <div className="summary-grid">
          {summaryMetrics.map((metric) => (
            <SummaryCard
              key={metric.label}
              label={metric.label}
              value={metric.value}
              helperText={metric.helperText}
            />
          ))}
        </div>
      </section>

      <section aria-labelledby="filters-heading">
        <h2 id="filters-heading">Filters</h2>

        <div className="filter-controls">
          <label className="filter-checkbox">
            <input
              type="checkbox"
              checked={showHighRiskOnly}
              onChange={(event) => setShowHighRiskOnly(event.target.checked)}
            />
            Show high-risk customers only
          </label>

          <label className="filter-field">
            Category
            <select
              value={selectedCategory}
              onChange={(event) => setSelectedCategory(event.target.value)}
            >
              <option value="All">All categories</option>
              <option value="Electronics">Electronics</option>
              <option value="Fashion">Fashion</option>
              <option value="Shoes">Shoes</option>
              <option value="Home">Home</option>
              <option value="Beauty">Beauty</option>
              <option value="Sports">Sports</option>
              <option value="Others">Others</option>
            </select>
          </label>

          <label className="filter-field">
            Minimum return ratio (%)
            <input
              type="number"
              min="0"
              max="100"
              value={minimumReturnRatio}
              onChange={(event) =>
                setMinimumReturnRatio(Number(event.target.value))
              }
            />
          </label>
        </div>
      </section>

      <CustomerTable
        customers={customers}
        totalCustomers={allCustomers.length}
        onViewProfile={handleViewProfile}
      />

      <section className="charts-section" aria-labelledby="charts-heading">
        <h2 id="charts-heading">Charts & Insights</h2>

        <div className="chart-grid">
          <CategoryReturnsChart data={categoryReturnData} />
          <RiskSegmentsChart data={riskSegmentData} />

          <div className="chart-grid__full">
            <ReturnTrendChart data={returnTrendData} />
          </div>
        </div>
      </section>

      <CustomerDetailModal
        customer={selectedCustomer}
        onClose={() => setSelectedCustomer(null)}
      />
    </main>
  )
}

export default App
