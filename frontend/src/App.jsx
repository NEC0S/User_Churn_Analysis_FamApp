import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Nav from './components/Nav'
import Predict from './pages/Predict'
import DataPipeline from './pages/DataPipeline'

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-bg">
        <div
          className="pointer-events-none fixed inset-0 opacity-40"
          style={{
            background:
              'radial-gradient(ellipse 900px 500px at 15% -10%, rgba(91,157,240,0.12), transparent), radial-gradient(ellipse 700px 500px at 100% 0%, rgba(47,184,138,0.08), transparent)',
          }}
        />
        <Nav />
        <Routes>
          <Route path="/" element={<Predict />} />
          <Route path="/pipeline" element={<DataPipeline />} />
        </Routes>
      </div>
    </BrowserRouter>
  )
}
