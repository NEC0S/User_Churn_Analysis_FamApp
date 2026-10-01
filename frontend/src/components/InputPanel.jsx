import { PRESETS } from '../data/presets'

function Field({ label, value, unit = '', min, max, step = 1, onChange }) {
  return (
    <div className="mb-3.5">
      <label className="flex justify-between text-[0.78rem] text-muted mb-1.5">
        <span>{label}</span>
        <span className="text-ink font-mono text-[0.76rem]">{value}{unit}</span>
      </label>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full accent-focus"
      />
    </div>
  )
}

export default function InputPanel({ profile, setProfile, apiUrl, setApiUrl, onRun, loading }) {
  const update = (key) => (val) => setProfile((p) => ({ ...p, [key]: val }))

  return (
    <div className="bg-panel/90 backdrop-blur-sm border border-line/70 rounded-xl p-[18px] shadow-[0_1px_0_rgba(255,255,255,0.03)_inset] h-fit">
      <h2 className="text-[0.78rem] font-semibold text-muted mb-3.5 tracking-wide">CUSTOMER PROFILE</h2>

      <div className="mb-4">
        <label className="block text-[0.78rem] text-muted mb-1.5">Backend API URL</label>
        <input
          type="text" value={apiUrl} onChange={(e) => setApiUrl(e.target.value)}
          className="w-full bg-panel2 border border-line text-ink rounded-md px-2 py-1.5 text-[0.76rem] font-mono"
        />
      </div>

      <div className="flex flex-wrap gap-1.5 mb-4">
        {Object.entries(PRESETS).map(([key, p]) => (
          <button
            key={key}
            onClick={() => setProfile({ ...p })}
            className="bg-panel2 border border-line text-ink rounded-md px-2.5 py-1.5 text-[0.78rem] hover:border-focus transition-colors"
          >
            {p.label}
          </button>
        ))}
      </div>

      <Field label="Transactions in last 20 days" value={profile.txn_count} min={0} max={30} onChange={update('txn_count')} />
      <Field label="Total spend (₹)" value={profile.total_amount} min={0} max={15000} step={100} onChange={update('total_amount')} />
      <Field label="Days since last transaction" value={profile.days_since} min={0} max={30} onChange={update('days_since')} />
      <Field label="Trend (late half − early half)" value={profile.txn_trend} min={-15} max={15} onChange={update('txn_trend')} />
      <Field label="Failed transaction rate" value={profile.failure_rate} unit="%" min={0} max={50} onChange={update('failure_rate')} />
      <Field label="Support tickets" value={profile.support_tickets} min={0} max={10} onChange={update('support_tickets')} />
      <Field label="Tenure (days since signup)" value={profile.tenure} min={0} max={400} onChange={update('tenure')} />

      <div className="flex items-center justify-between mb-4">
        <span className="text-[0.78rem] text-muted">Has used in-app offers</span>
        <button
          onClick={() => setProfile((p) => ({ ...p, offers: !p.offers }))}
          className={`relative w-[38px] h-[21px] rounded-full border transition-colors ${
            profile.offers ? 'bg-safeDim border-safe' : 'bg-panel2 border-line'
          }`}
        >
          <span
            className={`absolute top-[2px] left-[2px] w-[15px] h-[15px] rounded-full transition-transform ${
              profile.offers ? 'translate-x-[17px] bg-safe' : 'bg-faint'
            }`}
          />
        </button>
      </div>

      <button
        onClick={onRun}
        disabled={loading}
        className="w-full bg-gradient-to-r from-focus to-[#4A7FD9] text-white rounded-lg py-2.5 text-[0.88rem] font-semibold hover:opacity-90 disabled:opacity-60 transition-opacity shadow-lg shadow-focus/20"
      >
        {loading ? 'Scoring…' : 'Run prediction →'}
      </button>
    </div>
  )
}
