import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

export const useUserStore = defineStore('user', () => {
  const user = ref(JSON.parse(localStorage.getItem('sd_user') || 'null'))
  const isLoggedIn = computed(() => !!user.value)

  function setUser(u) {
    user.value = u
    localStorage.setItem('sd_user', JSON.stringify(u))
  }

  function logout() {
    user.value = null
    localStorage.removeItem('sd_user')
  }

  return { user, isLoggedIn, setUser, logout }
})
