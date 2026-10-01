import {
  Database, Scissors, Wand2, Ruler, Boxes, Cpu, Target, Lightbulb,
} from 'lucide-react'

function Formula({ children }) {
  return (
    <code className="block bg-panel2/80 border border-line/60 rounded-lg px-3.5 py-2.5 text-[0.8rem] font-mono text-focus overflow-x-auto">
      {children}
    </code>
  )
}

function Step({ n, icon: Icon, title, kicker, children }) {
  return (
    <section className="relative pl-14 pb-10 last:pb-0">
      <div className="absolute left-0 top-0 w-9 h-9 rounded-lg bg-panel2 border border-line/70 flex items-center justify-center">
        <Icon size={16} className="text-focus" />
      </div>
      {n !== 7 && <div className="absolute left-[17px] top-9 bottom-0 w-px bg-line/60" />}
      <div className="text-[0.68rem] font-mono text-faint mb-1 tracking-wider">STEP {n}</div>
      <h3 className="text-[1.05rem] font-semibold mb-1">{title}</h3>
      {kicker && <p className="text-muted text-[0.85rem] mb-3.5 max-w-2xl">{kicker}</p>}
      <div className="space-y-3">{children}</div>
    </section>
  )
}

function Row({ label, value }) {
  return (
    <div className="flex justify-between items-baseline gap-4 py-1.5 border-b border-line/40 last:border-0 text-[0.82rem]">
      <span className="text-muted">{label}</span>
      <span className="font-mono text-ink text-right">{value}</span>
    </div>
  )
}

export default function DataPipeline() {
  return (
    <div className="relative z-10 max-w-[820px] mx-auto px-5 pt-8 pb-20">
      <header className="mb-10">
        <h1 className="text-[1.75rem] font-bold tracking-tight mb-2">
          From raw transactions to a probability.
        </h1>
        <p className="text-muted text-[0.9rem] max-w-xl">
          Every transformation below runs on the real training data — two tables, one model.
          Nothing here is simplified for the demo; this is the actual pipeline.
        </p>
      </header>

      <Step n={1} icon={Database} title="Two raw tables" kicker="No feature exists yet — just transaction logs and account attributes.">
        <div className="grid grid-cols-2 gap-3">
          <div className="bg-panel/80 border border-line/60 rounded-lg p-3.5">
            <div className="text-[0.72rem] font-mono text-safe mb-2">transactions</div>
            <div className="text-[0.76rem] text-muted leading-relaxed font-mono">
              user_id, transaction_date, amount, payment_method, transaction_type, status
            </div>
          </div>
          <div className="bg-panel/80 border border-line/60 rounded-lg p-3.5">
            <div className="text-[0.72rem] font-mono text-safe mb-2">users</div>
            <div className="text-[0.76rem] text-muted leading-relaxed font-mono">
              user_id, registration_date, age_group, city, device_type, support_tickets, ...
            </div>
          </div>
        </div>
      </Step>

      <Step n={2} icon={Scissors} title="Split by time, not by row" kicker="Every feature is built from one window; the label comes from a strictly later window. This is what prevents the model from seeing its own answer.">
        <Formula>{`feature_window = [Apr 19, May 8]   →  builds X (inputs)
label_window   = (May 8, May 19]   →  builds y (churn = 1 if silent)`}</Formula>
        <p className="text-[0.82rem] text-muted leading-relaxed">
          A transaction dated May 10 is never used to compute a feature — only to decide the
          label. The two windows never overlap.
        </p>
      </Step>

      <Step n={3} icon={Wand2} title="Raw rows → engineered features" kicker="Transactions are grouped per user and reduced to signals a model can actually use.">
        <div className="bg-panel/80 border border-line/60 rounded-lg p-3.5">
          <Row label="txn_count" value="count(txns in window)" />
          <Row label="total_amount / avg_amount" value="sum / mean(amount)" />
          <Row label="days_since_last_txn" value="cutoff − max(date)" />
          <Row label="failure_rate" value="mean(status == 'Failed')" />
          <Row label="txn_trend" value="count(late half) − count(early half)" />
          <Row label="n_merchants / n_payment_methods" value="nunique(merchant / method)" />
          <Row label="top_txn_type" value="most frequent category" />
        </div>
        <p className="text-[0.82rem] text-muted leading-relaxed">
          25 features total. Each row of the training table is one user, one time window.
        </p>
      </Step>

      <Step n={4} icon={Ruler} title="Standardize the numbers" kicker="Raw feature values live on wildly different scales — ₹11,000 vs. a 3-day recency. The model needs them comparable.">
        <Formula>{`z = (x − mean) / std      # every numeric feature, fit on training data only`}</Formula>
        <p className="text-[0.82rem] text-muted leading-relaxed">
          The mean and std are learned once, from the training split — never recomputed on data
          the model will later be tested against.
        </p>
      </Step>

      <Step n={5} icon={Boxes} title="Encode the categories" kicker="City, device type, top transaction category — text the model can't read as numbers.">
        <Formula>{`city = "Mumbai"  →  [city_Mumbai=1, city_Delhi=0, city_Pune=0, ...]`}</Formula>
        <p className="text-[0.82rem] text-muted leading-relaxed">
          One-hot encoding: each category becomes its own 0/1 column. No ordering is implied —
          "Mumbai" isn't treated as greater or less than "Delhi."
        </p>
      </Step>

      <Step n={6} icon={Cpu} title="Score with gradient-boosted trees" kicker="LightGBM — hundreds of shallow decision trees, each correcting the last one's mistakes.">
        <Formula>{`P(churn) = sigmoid( Σ tree_i(x) )      # tuned via Optuna, 15+ trials`}</Formula>
        <p className="text-[0.82rem] text-muted leading-relaxed">
          Hyperparameters — tree depth, leaf count, learning rate — were searched, not guessed,
          against a held-out validation split.
        </p>
      </Step>

      <Step n={7} icon={Target} title="Threshold, then explain" kicker="A probability alone isn't a decision. And a decision without a reason isn't actionable.">
        <Formula>{`flag = P(churn) ≥ 0.77          # picked to maximize expected ₹ value, not accuracy
SHAP(feature) = this row's pull on the score, vs. the model's average prediction`}</Formula>
        <p className="text-[0.82rem] text-muted leading-relaxed">
          The threshold comes from comparing campaign cost against customer value — not a generic
          statistical cutoff. The explanation bars on the Predict page are real SHAP values,
          computed fresh for that one customer, every time.
        </p>
      </Step>

      <div className="mt-4 flex items-start gap-3 bg-focusDim/40 border border-[#2A4160] rounded-xl p-4">
        <Lightbulb size={16} className="text-focus mt-0.5 flex-shrink-0" />
        <p className="text-[0.82rem] text-muted leading-relaxed">
          <b className="text-ink">Why this matters:</b> a churn score that can't be traced back to
          real behavior isn't trustworthy enough to act on. Every step above is auditable — you
          can point to the exact line of code that produced any number this app shows.
        </p>
      </div>
    </div>
  )
}
