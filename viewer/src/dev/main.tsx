import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './dev.css'
import DevPanel from './DevPanel'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <DevPanel />
  </StrictMode>,
)
