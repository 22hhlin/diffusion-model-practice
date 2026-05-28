import axios from 'axios'

const api = axios.create({ baseURL: '/api' })

export const authApi = {
  register: (username, password) => api.post('/auth/register', { username, password }),
  login: (username, password) => api.post('/auth/login', { username, password }),
}

export const generateApi = {
  txt2img: (params) => api.post('/generate/txt2img', params),
  img2img: (params) => api.post('/generate/img2img', params),
  batch: (params) => api.post('/generate/batch', params),
}

export const historyApi = {
  list: (username, limit = 50, offset = 0) => api.get(`/history/${username}`, { params: { limit, offset } }),
  delete: (username, item_id) => api.delete('/history/', { data: { username, item_id } }),
}

export default api
