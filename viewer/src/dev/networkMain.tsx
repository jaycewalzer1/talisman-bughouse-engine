import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './dev.css'
import './network.css'
import NetworkDiagram from './NetworkDiagram'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <NetworkDiagram />
  </StrictMode>,
)
