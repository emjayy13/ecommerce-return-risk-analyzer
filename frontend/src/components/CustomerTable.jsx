function CustomerTable({ customers,totalCustomers, onViewProfile }) {
  return (
    <section aria-labelledby="customers-heading">
      <h2 id="customers-heading">Customer Risk List</h2>
      <p className="table-summary">
        Showing {customers.length} of {totalCustomers} customers
      </p>
      <div className="table-wrapper">
        <table>  {/* The table element is used to display tabular data. It contains a header (thead) and a body (tbody). The header defines the column names, while the body contains the actual customer data. */}
          <thead> {/*represents the columns of the table. It contains a single row (tr) with multiple header cells (th). Each th element represents a column header, such as "Customer", "Category", "Return Ratio", "Risk Score", "Risk Level", and "Profile". */}
            <tr> {/*it is a single row in the table header. It contains multiple header cells (th) that define the column names for the customer data. */}
              <th>Customer</th>
              
              <th>Return Ratio</th>
              <th>Risk Score</th>
              <th>Risk Level</th>
              <th>Profile</th>
            </tr>
          </thead>

          <tbody> {/* represents body of table i.e data of customers. It contains multiple rows (tr), each representing a customer. Each row contains multiple data cells (td) that display the customer's information, such as name, category, return ratio, risk score, risk level, and a button to view the customer's profile. */}
            
            {customers.length === 0 ? (
              <tr>
                <td className="empty-table-message" colSpan="5">
                  No customers match the selected filters.
                </td>
              </tr>
            ) : (
              customers.map((customer) => (
                <tr key={customer.id}>
                  <td>{customer.name}</td>
                  
                  <td>{Math.round(customer.returnRatio * 100)}%</td>
                  <td>{customer.riskScore}</td>
                  <td>{customer.riskLevel}</td>
                  <td>
                    <button
                      type="button"
                      onClick={() => onViewProfile(customer)}
                    >
                      View Profile
                    </button>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </section>
  )
}

export default CustomerTable