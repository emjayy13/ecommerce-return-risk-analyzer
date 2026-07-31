export const summaryMetrics = [
  {
    label: 'Total Returns',
    value: '1,248',
    helperText: 'Across all customers',
  },
  {
    label: 'Average Return Ratio',
    value: '24.6%',
    helperText: 'Returns divided by orders',
  },
  {
    label: 'High-Risk Customers',
    value: '14',
    helperText: 'Risk score above 70',
  },
]

export const customers = [
  {
    id: 'CUST001',
    name: 'Aarav Sharma',
    category: 'Fashion',
    totalOrders: 12,
    totalReturns: 5,
    returnRatio: 0.42,
    riskScore: 82,
    riskLevel: 'High',
    flags: ['Item mismatch history', 'Frequent return requests'],
    returnHistory: [
      { date: '2026-07-18', reason: 'Size issue' },
      { date: '2026-06-29', reason: 'Defective item' },
    ],
  },
  {
    id: 'CUST002',
    name: 'Diya Patel',
    category: 'Electronics',
    totalOrders: 17,
    totalReturns: 3,
    returnRatio: 0.18,
    riskScore: 35,
    riskLevel: 'Low',
    flags: [],
    returnHistory: [
      { date: '2026-07-09', reason: 'Changed mind' },
      { date: '2026-05-22', reason: 'Product not as expected' },
    ],
  },
  {
    id: 'CUST003',
    name: 'Kabir Singh',
    category: 'Shoes',
    totalOrders: 19,
    totalReturns: 7,
    returnRatio: 0.37,
    riskScore: 68,
    riskLevel: 'Medium',
    flags: ['Repeated size-related returns'],
    returnHistory: [
      { date: '2026-07-13', reason: 'Size issue' },
      { date: '2026-06-17', reason: 'Size issue' },
    ],
  },
  {
    id: 'CUST004',
    name: 'Meera Iyer',
    category: 'Fashion',
    totalOrders: 35,
    totalReturns: 18,
    returnRatio: 0.51,
    riskScore: 91,
    riskLevel: 'High',
    flags: ['Mismatch flag', 'Refund-fraud review required'],
    returnHistory: [
      { date: '2026-07-20', reason: 'Wrong item received' },
      { date: '2026-07-02', reason: 'Defective item' },
    ],
  },
  {
    id: 'CUST005',
    name: 'Rohan Gupta',
    category: 'Home & Kitchen',
    totalOrders: 17,
    totalReturns: 5,
    returnRatio: 0.29,
    riskScore: 47,
    riskLevel: 'Medium',
    flags: ['Frequent return requests'],
    returnHistory: [
      { date: '2026-06-25', reason: 'Product not as expected' },
      { date: '2026-05-14', reason: 'Damaged packaging' },
    ],
  },
]

export const categoryReturnData = [
  { category: 'Fashion', returns: 23 },
  { category: 'Electronics', returns: 3 },
  { category: 'Shoes', returns: 7 },
  { category: 'Home & Kitchen', returns: 5 },
]