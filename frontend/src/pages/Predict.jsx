import { useState, useEffect, useRef } from 'react'
import { Link } from 'react-router-dom'
import { ArrowUpRight } from 'lucide-react'
import InputPanel from '../components/InputPanel'
import PipelineView from '../components/PipelineView'
import ResultGauge from '../components/ResultGauge'
import ExplanationBars from '../components/ExplanationBars'
import { DEFAULT_PROFILE } from '../data/presets'
import { predictChurn, getApiBase, setApiBase } from '../api'

const STAGE_DELAY_MS = 220

export default function Predict() {
  const [profile, setProfile] = useState({ ...DEFAULT_PROFILE })
  const [apiUrl, setApiUrlState] = useState(getApiBase())
  const [loading, setLoading] = useState(false)
  const [activeStage, setActiveStage] = useState(-1)
  const [detail, setDetail] = useState({})
  const [apiResult, setApiResult] = useState(null)
  const [threshold, setThreshold] = useState(0.61)
  const [error, setError] = useState(null)
  const timers = useRef([])

  useEffect(() => setApiBase(apiUrl), [apiUrl])
  useEffect(() => () => timers.current.forEach(clearTimeout), [])

  async function runPrediction() {
    timers.current.forEach(clearTimeout)
    timers.current = []
    setLoading(true)
    setError(null)
    setActiveStage(-1)

    const payload = {
      txn_count: profile.txn_count,
      total_amount: profile.total_amount,
      days_since_last_txn: profile.days_since,
      txn_trend: profile.txn_trend,
      failure_rate: profile.failure_rate / 100,
      support_tickets: profile.support_tickets,
      tenure_days_at_cutoff: profile.tenure,
      has_used_offers: profile.offers,
    }

    setDetail({
      input: `<b>${profile.txn_count}</b> txns · ₹<b>${profile.total_amount}</b> · <b>${profile.days_since}</b>d since last`,
      features: `recency=<b>${profile.days_since}d</b>, freq=<b>${profile.txn_count}</b>, trend=<b>${profile.txn_trend >= 0 ? '+' : ''}${profile.txn_trend}</b>, fail=<b>${profile.failure_rate}%</b>`,
      standardize: 'z-scored + one-hot encoded',
      model: 'scoring on LightGBM…',
    })

    for (let i = 0; i <= 2; i++) {
      timers.current.push(setTimeout(() => setActiveStage(i), i * STAGE_DELAY_MS))
    }

    try {
      const result = await predictChurn(payload)
      setActiveStage(3)
      setThreshold(result.threshold)

      timers.current.push(setTimeout(() => {
        setDetail((d) => ({
          ...d,
          proba: `<b>${(result.probability * 100).toFixed(1)}%</b> churn probability`,
        }))
        setActiveStage(4)
      }, STAGE_DELAY_MS))

      timers.current.push(setTimeout(() => {
        setDetail((d) => ({
          ...d,
          decision: `${(result.probability * 100).toFixed(1)}% ${result.flagged ? '≥' : '<'} ${result.threshold.toFixed(2)} → <b>${result.flagged ? 'flag' : 'clear'}</b>`,
        }))
        setActiveStage(5)
        setApiResult(result)
        setLoading(false)
      }, STAGE_DELAY_MS * 2))
    } catch (err) {
      setError(err.message)
      setLoading(false)
      setActiveStage(3)
    }
  }

  return (
    <div className="relative z-10 max-w-[1180px] mx-auto px-5 pt-8 pb-16">
      <header className="mb-7 flex items-end justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-[1.75rem] font-bold tracking-tight leading-none mb-2">
            Score a customer.
          </h1>
          <p className="text-muted text-[0.9rem] max-w-md">
            Live inference on a trained LightGBM model. Every number below is real — not a
            simulation.
          </p>
        </div>
        <Link
          to="/pipeline"
          className="flex items-center gap-1 text-[0.8rem] text-focus hover:underline underline-offset-4"
        >
          How the raw data becomes a score <ArrowUpRight size={14} />
        </Link>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-[300px_1fr] gap-5">
        <InputPanel
          profile={profile} setProfile={setProfile}
          apiUrl={apiUrl} setApiUrl={setApiUrlState}
          onRun={runPrediction} loading={loading}
        />

        <div>
          <PipelineView activeIndex={activeStage} detail={detail} />

          <div className="bg-panel/90 backdrop-blur-sm border border-line/70 rounded-xl p-[18px] shadow-[0_1px_0_rgba(255,255,255,0.03)_inset]">
            <h2 className="text-[0.78rem] font-semibold text-muted mb-3.5 tracking-wide">RESULT</h2>
            <div className="grid grid-cols-1 md:grid-cols-[220px_1fr] gap-5">
              <ResultGauge
                probability={apiResult?.probability ?? null}
                threshold={threshold} setThreshold={setThreshold}
                apiThreshold={apiResult?.threshold}
              />
              <div>
                <ExplanationBars factors={apiResult?.top_factors} error={error} />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
