const STAGES = [
  { key: 'input', title: 'Raw input' },
  { key: 'features', title: 'Feature engineering' },
  { key: 'standardize', title: 'Standardize + encode' },
  { key: 'model', title: 'Model scoring' },
  { key: 'proba', title: 'Probability' },
  { key: 'decision', title: 'Decision' },
]

export default function PipelineView({ activeIndex, detail }) {
  return (
    <div className="flex gap-2.5 overflow-x-auto pb-1.5 mb-4">
      {STAGES.map((s, i) => {
        const state = i < activeIndex ? 'done' : i === activeIndex ? 'active' : 'idle'
        return (
          <div
            key={s.key}
            className={`flex-1 min-w-[150px] bg-panel/90 backdrop-blur-sm border rounded-xl p-3.5 transition-all duration-300 ${
              state === 'active'
                ? 'opacity-100 border-focus -translate-y-0.5 shadow-lg shadow-focus/10'
                : state === 'done'
                ? 'opacity-100 border-line/70'
                : 'opacity-40 border-line/70'
            }`}
          >
            <div className="font-mono text-[0.68rem] text-focus mb-1.5">{String(i + 1).padStart(2, '0')}</div>
            <h3 className="text-[0.82rem] font-semibold mb-1.5">{s.title}</h3>
            <div className="text-[0.72rem] text-muted leading-relaxed min-h-[34px]"
                 dangerouslySetInnerHTML={{ __html: detail[s.key] || '' }} />
          </div>
        )
      })}
    </div>
  )
}
