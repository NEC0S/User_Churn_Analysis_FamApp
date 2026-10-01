export default function ExplanationBars({ factors, error }) {
  if (error) {
    return (
      <div className="bg-panel2 rounded-lg p-3.5 text-[0.82rem] leading-relaxed">
        <b className="text-risk">Couldn't reach the backend.</b> {error} Check that your API is
        running and the URL above is correct (and that CORS allows this origin).
      </div>
    )
  }

  if (!factors || factors.length === 0) {
    return (
      <div className="bg-panel2 rounded-lg p-3.5 text-[0.82rem] text-muted">
        Run a prediction to see which factors drove the score.
      </div>
    )
  }

  const maxAbs = Math.max(...factors.map((f) => Math.abs(f.contribution)), 0.01)
  const top = factors[0]
  const second = factors[1]

  return (
    <>
      <div className="flex flex-col gap-2.5">
        {factors.map((f) => {
          const pctW = Math.min(100, (Math.abs(f.contribution) / maxAbs) * 50)
          const isRisk = f.contribution >= 0
          return (
            <div key={f.feature} className="grid grid-cols-[150px_1fr_56px] items-center gap-2.5">
              <div className="text-[0.76rem] text-muted truncate">{f.feature}</div>
              <div className="relative h-[18px] bg-panel2 rounded overflow-hidden">
                <div className="absolute left-1/2 top-0 bottom-0 w-px bg-line" />
                <div
                  className={`absolute top-0 bottom-0 transition-all duration-500 ${isRisk ? 'bg-risk right-1/2' : 'bg-safe left-1/2'}`}
                  style={{ width: `${pctW}%` }}
                />
              </div>
              <div className={`font-mono text-[0.72rem] text-right ${isRisk ? 'text-risk' : 'text-safe'}`}>
                {isRisk ? '+' : ''}{f.contribution.toFixed(3)}
              </div>
            </div>
          )
        })}
      </div>
      <div className="bg-panel2 rounded-lg p-3.5 text-[0.82rem] leading-relaxed mt-3.5">
        <b className="text-focus">Why this score (real SHAP values):</b> <b>{top.feature}</b> had
        the largest effect, pushing {top.contribution >= 0 ? 'toward churn' : 'toward retention'},
        followed by <b>{second?.feature}</b>. Red bars push toward churn; green bars pull toward
        retention — computed per-prediction, not a fixed global ranking.
      </div>
    </>
  )
}
