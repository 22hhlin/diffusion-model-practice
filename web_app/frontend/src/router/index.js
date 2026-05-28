import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  { path: '/', name: 'Txt2Img', component: () => import('../views/Txt2Img.vue') },
  { path: '/img2img', name: 'Img2Img', component: () => import('../views/Img2Img.vue') },
  { path: '/batch', name: 'Batch', component: () => import('../views/Batch.vue') },
  { path: '/history', name: 'History', component: () => import('../views/History.vue') },
]

export default createRouter({ history: createWebHistory(), routes })
