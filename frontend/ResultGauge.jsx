import { NavLink } from 'react-router-dom'
import { Radar, Workflow } from 'lucide-react'

const linkBase =
  'flex items-center gap-1.5 px-3.5 py-2 rounded-lg text-[0.82rem] font-medium transition-all'

export default function Nav() {
  return (
    <nav className="sticky top-0 z-20 border-b border-line/70 bg-bg/85 backdrop-blur-md">
      <div className="max-w-[1180px] mx-auto px-5 h-14 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-6 h-6 rounded-md bg-gradient-to-br from-focus to-safe flex items-center justify-center">
            <span className="text-[0.6rem] font-black text-bg">C</span>
          </div>
          <span className="font-semibold text-[0.92rem] tracking-tight">ChurnLens</span>
        </div>
        <div className="flex items-center gap-1 bg-panel2/60 border border-line/60 rounded-xl p-1">
          <NavLink
            to="/"
            end
            className={({ isActive }) =>
              `${linkBase} ${isActive ? 'bg-focus text-white' : 'text-muted hover:text-ink'}`
            }
          >
            <Radar size={14} /> Predict
          </NavLink>
          <NavLink
            to="/pipeline"
            className={({ isActive }) =>
              `${linkBase} ${isActive ? 'bg-focus text-white' : 'text-muted hover:text-ink'}`
            }
          >
            <Workflow size={14} /> Data Pipeline
          </NavLink>
        </div>
      </div>
    </nav>
  )
}
