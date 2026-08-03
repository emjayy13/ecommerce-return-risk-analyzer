import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />   {/* means React checks everything inside <App /> for potential problems*/}
  </StrictMode>,
)

{/*empty box named root in index.html file is the place where the React app will be rendered*/}