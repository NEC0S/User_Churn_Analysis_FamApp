export const PRESETS = {
  loyal: {
    label: 'Loyal customer',
    txn_count: 22, total_amount: 11000, days_since: 1, txn_trend: 5,
    failure_rate: 1, support_tickets: 0, tenure: 340, offers: true,
  },
  atrisk: {
    label: 'At-risk customer',
    txn_count: 3, total_amount: 900, days_since: 12, txn_trend: -6,
    failure_rate: 18, support_tickets: 3, tenure: 150, offers: false,
  },
  new: {
    label: 'New user',
    txn_count: 2, total_amount: 400, days_since: 5, txn_trend: 1,
    failure_rate: 0, support_tickets: 0, tenure: 10, offers: false,
  },
  gone: {
    label: 'Already silent',
    txn_count: 0, total_amount: 0, days_since: 20, txn_trend: -3,
    failure_rate: 0, support_tickets: 1, tenure: 200, offers: false,
  },
}

export const DEFAULT_PROFILE = PRESETS.loyal
