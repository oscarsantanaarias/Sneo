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

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>
);
