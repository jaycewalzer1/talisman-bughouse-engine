import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './dev.css'
import './study.css'
import Study from './Study'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <Study />
  </StrictMode>,
)
