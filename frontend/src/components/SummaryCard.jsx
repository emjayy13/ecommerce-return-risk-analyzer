function SummaryCard({ label, value, helperText }) {
  return (
    <article className="summary-card">
      <p className="summary-card__label">{label}</p>
      <p className="summary-card__value">{value}</p>
      {helperText && (
        <p className="summary-card__helper">{helperText}</p> 
      )}
      
    </article>
  )
}

export default SummaryCard

{/* The helperText prop is optional. If it is provided, it will be rendered as a paragraph with the class summary-card__helper. If it is not provided, nothing will be rendered in that place. */}