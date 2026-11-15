import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { fileURLToPath } from 'url';

export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      input: {
        main: fileURLToPath(new URL('./index.html', import.meta.url)),
        phone: fileURLToPath(new URL('./m.html', import.meta.url))
      }
    }
  },
  server: {
    host: '127.0.0.1',
    port: 9999,
    strictPort: true
  },
  preview: {
    port: 9999,
    strictPort: true
  }
});
