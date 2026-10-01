import { NavLink } from 'react-router-dom'

const link = ({ isActive }) =>
  `px-1 py-4 text-[0.85rem] font-semibold border-b-2 transition-colors ${
    isActive ? 'border-focus text-ink' : 'border-transparent text-muted hover:text-ink'
  }`

export default function Nav() {
  return (
    <nav className="sticky top-0 z-20 bg-panel/90 backdrop-blur border-b border-line">
      <div className="max-w-[1180px] mx-auto px-5 flex items-center justify-between">
        <div className="flex items-center gap-2.5 py-3">
          <svg width="26" height="26" viewBox="0 0 26 26" aria-hidden="true">
            <rect width="26" height="26" rx="7" fill="#0E1B2E" />
            <path d="M5 17 L10 11 L14 15 L21 7" fill="none" stroke="#6F93FF" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          <span className="font-extrabold text-[1rem] tracking-tight">ChurnLens</span>
          <span className="hidden sm:inline text-[0.78rem] text-faint border-l border-line pl-2.5 ml-1">
            Customer retention scoring
          </span>
        </div>
        <div className="flex items-center gap-6">
          <NavLink to="/" end className={link}>Score a customer</NavLink>
          <NavLink to="/pipeline" className={link}>How it works</NavLink>
        </div>
      </div>
    </nav>
  )
}