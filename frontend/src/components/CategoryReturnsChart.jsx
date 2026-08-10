import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

function CategoryReturnsChart({ data = [] }) {
  const hasData = data.length > 0

  return (
    <section
      className="chart-card"
      aria-labelledby="category-chart-heading"
    >
      <div className="chart-card__header">
        <div>
          <p className="chart-card__eyebrow">Category analysis</p>
          <h2 id="category-chart-heading">Returns by Category</h2>
        </div>

        <p className="chart-card__description">
          Number of recorded returns in each product category.
        </p>
      </div>

      {hasData ? (
        <div className="chart-container">
          <ResponsiveContainer width="100%" height={320}>
            <BarChart
              data={data}
              margin={{ top: 12, right: 12, left: 0, bottom: 8 }}
            >
              <CartesianGrid
                stroke="#e2e8f0"
                strokeDasharray="4 4"
                vertical={false}
              />

              <XAxis
                dataKey="category"
                axisLine={false}
                tickLine={false}
                tick={{ fill: '#64748b', fontSize: 12 }}
              />

              <YAxis
                allowDecimals={false}
                axisLine={false}
                tickLine={false}
                tick={{ fill: '#64748b', fontSize: 12 }}
              />

              <Tooltip
                formatter={(value) => [`${value} returns`, 'Total']}
                contentStyle={{
                  border: '1px solid #dbe4f0',
                  borderRadius: '12px',
                  boxShadow: '0 10px 25px rgba(23, 32, 51, 0.12)',
                }}
                cursor={{ fill: 'rgba(37, 99, 235, 0.06)' }}
              />

              <Bar
                dataKey="returns"
                name="Returns"
                fill="#2563eb"
                maxBarSize={64}
                radius={[8, 8, 0, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <p className="chart-empty-state">
          No category return data is currently available.
        </p>
      )}
    </section>
  )
}

export default CategoryReturnsChart