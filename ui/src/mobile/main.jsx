import React from 'react';
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import MobileApp from './MobileApp.jsx';
import '../style.css';
import '../agents.css';
import './mobile.css';

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <MobileApp />
  </StrictMode>
);
