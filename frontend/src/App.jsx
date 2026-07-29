import { useState } from 'react'
import './App.css'
import CustomerTable from './components/CustomerTable.jsx'
import SummaryCard from './components/SummaryCard.jsx'
import CustomerDetailModal from './components/CustomerDetailModal.jsx'
import { customers} from './data/mockDashboardData.js'
import CategoryReturnsChart from './components/CategoryReturnsChart.jsx'
import RiskSegmentsChart from './components/RiskSegmentsChart.jsx'
import ReturnTrendChart from './components/ReturnTrendChart.jsx'

function buildCategoryReturnData(customerList) {
  const totalsByCategory = {}

  customerList.forEach((customer) => {
    if (totalsByCategory[customer.category] === undefined) {
      totalsByCategory[customer.category] = 0
    }

    totalsByCategory[customer.category] += customer.totalReturns
  })

  return Object.entries(totalsByCategory).map(([category, returns]) => ({
    category,
    returns,
  }))
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

function buildReturnTrendData(customerList) {
  const returnsByDate = {}

  customerList.forEach((customer) => {
    customer.returnHistory.forEach((returnItem) => {
      if (returnsByDate[returnItem.date] === undefined) {
        returnsByDate[returnItem.date] = 0
      }

      returnsByDate[returnItem.date] += 1
    })
  })

  return Object.entries(returnsByDate)
    .map(([date, returns]) => ({
      date,
      returns,
    }))
    .sort((firstItem, secondItem) =>
      firstItem.date.localeCompare(secondItem.date),
    )
}
 

function App() {

  const [showHighRiskOnly, setShowHighRiskOnly] = useState(false)
  const [selectedCategory, setSelectedCategory] = useState('All')
  const [minimumReturnRatio, setMinimumReturnRatio] = useState(0)
  const [selectedCustomer, setSelectedCustomer] = useState(null)

  const highRiskCustomers = showHighRiskOnly
    ? customers.filter((customer) => customer.riskScore > 70)
    : customers

  const categoryCustomers =
    selectedCategory === 'All'
      ? highRiskCustomers
      : highRiskCustomers.filter(
          (customer) => customer.category === selectedCategory,
        )

  const visibleCustomers = categoryCustomers.filter(
    (customer) => customer.returnRatio >= minimumReturnRatio / 100,
  )
    const sortedCustomers = [...visibleCustomers].sort(
    (firstCustomer, secondCustomer) =>
      secondCustomer.riskScore - firstCustomer.riskScore,
  )
    const categoryReturnData = buildCategoryReturnData(customers)
    const riskSegmentData = buildRiskLevelData(customers)
    const returnTrendData = buildReturnTrendData(customers)

      const totalReturns = customers.reduce(
    (total, customer) => total + customer.totalReturns,
    0,
  )

  const averageReturnRatio =
    customers.length === 0
      ? 0
      : customers.reduce(
          (total, customer) => total + customer.totalReturns / customer.totalOrders,
          0,
        ) / customers.length

  const highRiskCustomerCount = customers.filter(
    (customer) => customer.riskLevel === 'High',
  ).length

  const summaryMetrics = [
    {
      label: 'Total Returns',
      value: totalReturns.toLocaleString(),
      helperText: 'Across all customers',
    },
    {
      label: 'Average Return Ratio',
      value: `${(averageReturnRatio * 100).toFixed(1)}%`,
      helperText: 'Across all customers',
    },
    {
      label: 'High-Risk Customers',
      value: highRiskCustomerCount,
      helperText: 'Risk score above 70',
    },
  ]

  return (
    <main>
      <header>
        <p>Operations Dashboard</p>
        <h1>Customer Return Risk Analyzer</h1>
        <p>Monitor customer return behaviour and identify potential risk.</p>
      </header>

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
          <select  // creates a dropdown menu*/
            value={selectedCategory}
            onChange={(event) => setSelectedCategory(event.target.value)}
          >
            <option value="All">All categories</option>
            <option value="Electronics">Electronics</option>
            <option value="Fashion">Fashion</option>
            <option value="Home & Kitchen">Home & Kitchen</option>
            <option value="Shoes">Shoes</option>
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
  customers={sortedCustomers}
  totalCustomers={customers.length}
  onViewProfile={setSelectedCustomer}
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
