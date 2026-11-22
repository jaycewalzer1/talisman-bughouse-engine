import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './dev.css'
import './network.css'
import './durative.css'
import DurativeTree from './DurativeTree'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <DurativeTree />
  </StrictMode>,
)
