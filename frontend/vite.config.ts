import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    host: true,
    proxy: {
      '/api': {
        target: 'http://ai-galaxy-backend:8080',
        changeOrigin: true,
        secure: false,
      },
      '/ws': {
        target: 'ws://ai-galaxy-backend:8080',
        ws: true,
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
  define: {
    // Define environment variables
    __DEV__: JSON.stringify(process.env.NODE_ENV === 'development'),
  },
})