<template>
  <el-container v-if="userStore.isLoggedIn" class="app-container">
    <el-header class="app-header">
      <div class="header-left">
        <span class="logo">SD LoRA</span>
        <el-menu mode="horizontal" :default-active="route.path" router :ellipsis="false" class="nav-menu">
          <el-menu-item index="/">
            <el-icon><Picture /></el-icon>文生图
          </el-menu-item>
          <el-menu-item index="/img2img">
            <el-icon><Edit /></el-icon>图生图
          </el-menu-item>
          <el-menu-item index="/batch">
            <el-icon><Files /></el-icon>批量生成
          </el-menu-item>
          <el-menu-item index="/history">
            <el-icon><Clock /></el-icon>历史记录
          </el-menu-item>
        </el-menu>
      </div>
      <div class="header-right">
        <span class="username">{{ userStore.user?.username }}</span>
        <el-button text @click="handleLogout">退出</el-button>
      </div>
    </el-header>
    <el-main class="app-main">
      <router-view />
    </el-main>
  </el-container>
  <LoginView v-else />
</template>

<script setup>
import { useRoute } from 'vue-router'
import { useUserStore } from './store/user'
import LoginView from './views/Login.vue'

const route = useRoute()
const userStore = useUserStore()

function handleLogout() {
  userStore.logout()
}
</script>

<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Helvetica Neue', Helvetica, 'PingFang SC', 'Hiragino Sans GB', Arial, sans-serif; background: #f5f7fa; }
.app-container { height: 100vh; }
.app-header {
  display: flex; align-items: center; justify-content: space-between;
  background: #fff; box-shadow: 0 1px 4px rgba(0,0,0,0.08); padding: 0 24px; height: 60px;
}
.header-left { display: flex; align-items: center; gap: 24px; }
.logo { font-size: 20px; font-weight: 700; background: linear-gradient(135deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.nav-menu { border-bottom: none; }
.header-right { display: flex; align-items: center; gap: 12px; }
.username { color: #606266; font-size: 14px; }
.app-main { padding: 24px; overflow-y: auto; }
</style>
