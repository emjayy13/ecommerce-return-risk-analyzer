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

function RiskSegmentsChart({ data = [] }) {
  const hasData = data.some((segment) => segment.value > 0)

  return (
    <section className="chart-card" aria-labelledby="risk-chart-heading">
      <div className="chart-card__header">
        <div>
          <p className="chart-card__eyebrow">Risk distribution</p>
          <h2 id="risk-chart-heading">Customer Risk Segments</h2>
        </div>

        <p className="chart-card__description">
          Distribution of customers across the available risk levels.
        </p>
      </div>

      {hasData ? (
        <div className="chart-container">
          <ResponsiveContainer width="100%" height={320}>
            <PieChart>
              <Pie
                data={data}
                dataKey="value"
                nameKey="name"
                cx="50%"
                cy="46%"
                innerRadius={58}
                outerRadius={98}
                paddingAngle={3}
                label={({ name, percent }) =>
                  `${name} ${Math.round(percent * 100)}%`
                }
                labelLine={{ stroke: '#94a3b8' }}
              >
                {data.map((segment) => (
                  <Cell
                    key={segment.name}
                    fill={RISK_COLORS[segment.name] || '#64748b'}
                    stroke="#ffffff"
                    strokeWidth={2}
                  />
                ))}
              </Pie>

              <Tooltip
                formatter={(value, name) => [
                  `${value} customer${value === 1 ? '' : 's'}`,
                  name,
                ]}
                contentStyle={{
                  border: '1px solid #dbe4f0',
                  borderRadius: '12px',
                  boxShadow: '0 10px 25px rgba(23, 32, 51, 0.12)',
                }}
              />

              <Legend
                verticalAlign="bottom"
                iconType="circle"
                wrapperStyle={{ fontSize: '13px' }}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <p className="chart-empty-state">
          No customer risk data is currently available.
        </p>
      )}
    </section>
  )
}

export default RiskSegmentsChart