import React from 'react';
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App.jsx';
import './style.css';
import './agents.css';

const plainFetch = window.fetch.bind(window);

// si el servidor se reinicio la cookie de la sesion ya no sirve y la pagina se recarga una vez para recibir la nueva
window.fetch = async (...args) => {
  const res = await plainFetch(...args);

  if(res.status !== 403){
    return res;
  }

  const body = await res.clone().json().catch(() => ({}));
  let last = 0;

  try {
    last = Number(sessionStorage.getItem('sneo-reload') || 0);
  } catch(err) {
    last = 0;
  }

  if(body.message === 'Open SNEO from its own window to use it.' && Date.now() - last > 30000){
    try {
      sessionStorage.setItem('sneo-reload', String(Date.now()));
    } catch(err) {
      console.log(err.message);
    }

    window.location.reload();
  }

  return res;
};

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = {error: null};
  }
  static getDerivedStateFromError(error) {
    return {error};
  }
  render() {
    if(this.state.error){
      return (
        <div style={{padding: 32, color: '#e5e7eb', fontFamily: 'sans-serif', background: '#121316', minHeight: '100vh'}}>
          <h2 style={{marginBottom: 8}}>UI Error</h2>
          <pre style={{whiteSpace: 'pre-wrap', color: '#f87171', marginBottom: 16}}>{String(this.state.error?.message || this.state.error)}</pre>
          <button type="button" onClick={() => window.location.reload()} style={{padding: '8px 16px', cursor: 'pointer'}}>Reload</button>
        </div>
      );
    }
    return this.props.children;
  }
}

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </StrictMode>
);
