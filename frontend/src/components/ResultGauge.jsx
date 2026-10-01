export default function ResultGauge({ probability, threshold, setThreshold, apiThreshold }) {
  const circumference = 427
  const prob = probability ?? 0
  const flagged = probability !== null && probability >= threshold
  const offset = circumference - circumference * prob
  const stroke = probability === null ? '#5B9DF0' : flagged ? '#E2543F' : '#2FB88A'

  return (
    <div className="flex flex-col items-center justify-center">
      <svg width="160" height="160" viewBox="0 0 160 160">
        <circle cx="80" cy="80" r="68" fill="none" stroke="#E4E8F0" strokeWidth="14" />
        <circle
          cx="80" cy="80" r="68" fill="none" stroke={stroke} strokeWidth="14"
          strokeDasharray={circumference} strokeDashoffset={probability === null ? circumference : offset}
          strokeLinecap="round" transform="rotate(-90 80 80)"
          style={{ transition: 'stroke-dashoffset .7s ease, stroke .3s' }}
        />
      </svg>
      <div className="mt-2.5 text-center">
        <div className="text-[2rem] font-bold font-mono">
          {probability === null ? '—' : `${(prob * 100).toFixed(1)}%`}
        </div>
        {probability !== null && (
          <>
            <div className={`text-[0.8rem] font-semibold mt-0.5 px-2.5 py-0.5 rounded-md inline-block ${
              flagged ? 'bg-riskDim text-risk' : 'bg-safeDim text-safe'
            }`}>
              {flagged ? 'Flag for retention' : 'No action needed'}
            </div>
            <div className="text-muted text-[0.74rem] mt-1.5">
              {(prob * 100).toFixed(1)}% {flagged ? '≥' : '<'} {threshold.toFixed(2)}
              {apiThreshold !== undefined && ` (trained default: ${apiThreshold.toFixed(2)})`}
            </div>
          </>
        )}
      </div>
      <div className="flex items-center gap-2 mt-3 w-full">
        <label className="text-[0.72rem] text-muted whitespace-nowrap">Threshold</label>
        <input
          type="range" min="0.1" max="0.9" step="0.01" value={threshold}
          onChange={(e) => setThreshold(parseFloat(e.target.value))}
          className="w-full accent-focus"
        />
        <span className="font-mono text-[0.72rem]">{threshold.toFixed(2)}</span>
      </div>
    </div>
  )
}
