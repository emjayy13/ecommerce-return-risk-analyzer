import {
  Cell,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
} from 'recharts'

const RISK_COLORS = {
  High: '#dc2626',
  Medium: '#d97706',
  Low: '#16a34a',
}

function RiskSegmentsChart({ data }) {
  return (
    <section aria-labelledby="risk-chart-heading">
      <h2 id="risk-chart-heading">Customer Segments by Risk Level</h2>

      <div className="chart-container">
        <ResponsiveContainer width="100%" height={300}>
          <PieChart>
            <Pie
              data={data}
              dataKey="value"
              nameKey="name"
              cx="50%"
              cy="50%"
              outerRadius={100}
              label
            >
              {data.map((entry) => (
                <Cell key={entry.name} fill={RISK_COLORS[entry.name]} />
              ))}
            </Pie>
            <Tooltip />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      </div>
    </section>
  )
}

export default RiskSegmentsChart