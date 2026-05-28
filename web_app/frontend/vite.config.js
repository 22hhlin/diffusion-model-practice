import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    allowedHosts: ['784392-proxy-5173.dsw-gateway-cn-shanghai.data.aliyun.com'],
    proxy: {
      '/api': 'http://localhost:7860',
      '/outputs': 'http://localhost:7860',
      '/ws': {
        target: 'ws://localhost:7860',
        ws: true,
      },
    },
  },
})
