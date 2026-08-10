const navigationItems = [
  { label: 'Dashboard', sectionId: 'dashboard' },
  { label: 'Customers', sectionId: 'customers' },
  { label: 'Analytics', sectionId: 'analytics' },
]

function Navbar() {
  return (
    <nav className="navbar" aria-label="Main navigation">
      <a className="navbar__brand" href="#dashboard">
        ReturnGuard
      </a>

      <div className="navbar__links">
        {navigationItems.map((item) => (
          <a
            key={item.sectionId}
            className="navbar__link"
            href={`#${item.sectionId}`}
          >
            {item.label}
          </a>
        ))}
      </div>
    </nav>
  )
}

export default Navbar