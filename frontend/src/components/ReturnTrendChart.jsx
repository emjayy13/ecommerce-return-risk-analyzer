import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

function formatShortDate(date) {
  const [, month, day] = date.split('-')
  return `${day}/${month}`
}

function ReturnTrendChart({ data = [] }) {
  const hasData = data.length > 0

  return (
    <section className="chart-card" aria-labelledby="trend-chart-heading">
      <div className="chart-card__header">
        <div>
          <p className="chart-card__eyebrow">Timeline analysis</p>
          <h2 id="trend-chart-heading">Recent Return Trend</h2>
        </div>

        <p className="chart-card__description">
          Daily number of recorded product returns over time.
        </p>
      </div>

      {hasData ? (
        <div className="chart-container">
          <ResponsiveContainer width="100%" height={320}>
            <LineChart
              data={data}
              margin={{ top: 12, right: 20, left: 0, bottom: 8 }}
            >
              <CartesianGrid
                stroke="#e2e8f0"
                strokeDasharray="4 4"
                vertical={false}
              />

              <XAxis
                dataKey="date"
                axisLine={false}
                tickLine={false}
                tick={{ fill: '#64748b', fontSize: 12 }}
                tickFormatter={formatShortDate}
              />

              <YAxis
                allowDecimals={false}
                axisLine={false}
                tickLine={false}
                tick={{ fill: '#64748b', fontSize: 12 }}
              />

              <Tooltip
                labelFormatter={(date) => `Date: ${date}`}
                formatter={(value) => [`${value} returns`, 'Total']}
                contentStyle={{
                  border: '1px solid #dbe4f0',
                  borderRadius: '12px',
                  boxShadow: '0 10px 25px rgba(23, 32, 51, 0.12)',
                }}
              />

              <Line
                type="monotone"
                dataKey="returns"
                name="Returns"
                stroke="#7c3aed"
                strokeWidth={3}
                dot={{
                  r: 4,
                  fill: '#ffffff',
                  stroke: '#7c3aed',
                  strokeWidth: 3,
                }}
                activeDot={{
                  r: 7,
                  fill: '#7c3aed',
                  stroke: '#ffffff',
                  strokeWidth: 3,
                }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <p className="chart-empty-state">
          No return trend data is currently available.
        </p>
      )}
    </section>
  )
}

export default ReturnTrendChart