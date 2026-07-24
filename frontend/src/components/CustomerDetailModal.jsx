
// this is for popup/dailog box for customer profile

function CustomerDetailModal({ customer, onClose }) {
  if (!customer) {
    return null
  }

  return (
    <div
      className="modal-backdrop"
      role="dialog"
      aria-modal="true"
      aria-labelledby="customer-modal-title"
    >
      <div className="modal">
        <button
          type="button"
          className="modal__close"
          onClick={onClose}
          aria-label="Close customer profile"
        >
          ×
        </button>

        <p className="modal__eyebrow">Customer Profile</p>
        <h2 id="customer-modal-title">{customer.name}</h2>

        <dl className="customer-details">
          <div>
            <dt>Category</dt>
            <dd>{customer.category}</dd>
          </div>
          <div>
            <dt>Return Ratio</dt>
            <dd>{Math.round(customer.returnRatio * 100)}%</dd>
          </div>
          <div>
            <dt>Total Orders</dt>
            <dd>{customer.totalOrders}</dd>
          </div>
          <div>
            <dt>Total Returns</dt>
            <dd>{customer.totalReturns}</dd>
          </div>
          <div>
            <dt>Risk Score</dt>
            <dd>{customer.riskScore}</dd>
          </div>
          <div>
            <dt>Risk Level</dt>
            <dd>{customer.riskLevel}</dd>
          </div>
        </dl>

        <section className="modal__section" aria-labelledby="flags-heading">
          <h3 id="flags-heading">Risk Flags</h3>

          {customer.flags.length > 0 ? (
            <ul className="flag-list">
              {customer.flags.map((flag) => (
                <li key={flag}>{flag}</li>
              ))}
            </ul>
          ) : (
            <p>No risk flags recorded.</p>
          )}
        </section>

        <section className="modal__section" aria-labelledby="history-heading">
          <h3 id="history-heading">Recent Return History</h3>

          <ul className="return-history">
            {customer.returnHistory.map((returnItem) => (
              <li key={returnItem.date}>
                <span>{returnItem.reason}</span>
                <time dateTime={returnItem.date}>{returnItem.date}</time>
              </li>
            ))}
          </ul>
        </section>
      </div>
    </div>
  )
}

export default CustomerDetailModal