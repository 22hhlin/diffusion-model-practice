<template>
  <div class="login-page">
    <div class="login-card">
      <h1 class="login-title">SD LoRA</h1>
      <p class="login-subtitle">Stable Diffusion LoRA 文生图系统</p>
      <el-tabs v-model="tab" class="login-tabs">
        <el-tab-pane label="登录" name="login">
          <el-form @submit.prevent="handleLogin">
            <el-form-item>
              <el-input v-model="form.username" placeholder="用户名" prefix-icon="User" size="large" />
            </el-form-item>
            <el-form-item>
              <el-input v-model="form.password" type="password" placeholder="密码" prefix-icon="Lock" size="large" show-password />
            </el-form-item>
            <el-button type="primary" size="large" :loading="loading" @click="handleLogin" style="width:100%">登录</el-button>
          </el-form>
        </el-tab-pane>
        <el-tab-pane label="注册" name="register">
          <el-form @submit.prevent="handleRegister">
            <el-form-item>
              <el-input v-model="form.username" placeholder="用户名（至少2位）" prefix-icon="User" size="large" />
            </el-form-item>
            <el-form-item>
              <el-input v-model="form.password" type="password" placeholder="密码（至少4位）" prefix-icon="Lock" size="large" show-password />
            </el-form-item>
            <el-button type="primary" size="large" :loading="loading" @click="handleRegister" style="width:100%">注册</el-button>
          </el-form>
        </el-tab-pane>
      </el-tabs>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive } from 'vue'
import { ElMessage } from 'element-plus'
import { authApi } from '../api'
import { useUserStore } from '../store/user'

const userStore = useUserStore()
const tab = ref('login')
const loading = ref(false)
const form = reactive({ username: '', password: '' })

async function handleLogin() {
  if (!form.username || !form.password) return ElMessage.warning('请填写用户名和密码')
  loading.value = true
  try {
    const { data } = await authApi.login(form.username, form.password)
    userStore.setUser(data.user)
    ElMessage.success('登录成功')
  } catch (e) {
    ElMessage.error(e.response?.data?.detail || '登录失败')
  } finally {
    loading.value = false
  }
}

async function handleRegister() {
  if (!form.username || !form.password) return ElMessage.warning('请填写用户名和密码')
  loading.value = true
  try {
    const { data } = await authApi.register(form.username, form.password)
    userStore.setUser(data.user)
    ElMessage.success('注册成功')
  } catch (e) {
    ElMessage.error(e.response?.data?.detail || '注册失败')
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.login-page { min-height: 100vh; display: flex; align-items: center; justify-content: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
.login-card { background: #fff; border-radius: 16px; padding: 48px 40px; width: 400px; box-shadow: 0 20px 60px rgba(0,0,0,0.2); }
.login-title { text-align: center; font-size: 32px; font-weight: 700; background: linear-gradient(135deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 8px; }
.login-subtitle { text-align: center; color: #909399; margin-bottom: 32px; font-size: 14px; }
</style>
