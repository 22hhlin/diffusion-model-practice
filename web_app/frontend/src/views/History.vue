<template>
  <div class="history-page">
    <el-card>
      <template #header>
        <div class="history-header">
          <span class="card-title">生成历史</span>
          <div class="history-filters">
            <el-select v-model="filter" placeholder="类型筛选" clearable style="width:120px">
              <el-option label="文生图" value="txt2img" />
              <el-option label="图生图" value="img2img" />
              <el-option label="批量" value="batch" />
            </el-select>
            <el-input v-model="searchText" placeholder="搜索提示词" clearable style="width:200px" prefix-icon="Search" />
          </div>
        </div>
      </template>

      <div v-if="loading" class="loading-container">
        <el-icon class="is-loading" size="32"><Loading /></el-icon>
      </div>
      <div v-else-if="filteredItems.length" class="history-list">
        <div v-for="item in filteredItems" :key="item.id" class="history-item">
          <div class="history-left">
            <div class="history-meta">
              <el-tag :type="typeTag(item.type)" size="small">{{ typeLabel(item.type) }}</el-tag>
              <span class="history-time">{{ formatTime(item.timestamp) }}</span>
              <span v-if="item.elapsed" class="history-elapsed">{{ item.elapsed }}s</span>
            </div>
            <div class="history-prompt">{{ item.prompt }}</div>
            <div class="history-params">
              steps={{ item.params?.steps }}, cfg={{ item.params?.guidance_scale }}
              <template v-if="item.params?.seed">, seed={{ item.params.seed }}</template>
              <template v-if="item.params?.strength">, strength={{ item.params.strength }}</template>
            </div>
          </div>
          <div class="history-right">
            <div class="history-images">
              <el-image
                v-for="(img, i) in (item.images || []).slice(0, 4)"
                :key="i"
                :src="'data:image/png;base64,' + img"
                fit="cover"
                class="history-thumb"
                :preview-src-list="item.images.map(i => 'data:image/png;base64,' + i)"
                :initial-index="i"
              />
            </div>
            <div class="history-actions">
              <el-button text size="small" @click="reuseParams(item)">
                <el-icon><Refresh /></el-icon>重用参数
              </el-button>
              <el-button text size="small" type="danger" @click="handleDelete(item.id)">
                <el-icon><Delete /></el-icon>删除
              </el-button>
            </div>
          </div>
        </div>
      </div>
      <el-empty v-else description="暂无生成历史" />
    </el-card>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { historyApi } from '../api'
import { useUserStore } from '../store/user'

const userStore = useUserStore()
const router = useRouter()
const items = ref([])
const loading = ref(false)
const filter = ref('')
const searchText = ref('')

const filteredItems = computed(() => {
  let result = items.value
  if (filter.value) result = result.filter(i => i.type === filter.value)
  if (searchText.value) {
    const q = searchText.value.toLowerCase()
    result = result.filter(i => i.prompt?.toLowerCase().includes(q))
  }
  return result
})

function typeLabel(type) {
  return { txt2img: '文生图', img2img: '图生图', batch: '批量' }[type] || type
}
function typeTag(type) {
  return { txt2img: '', img2img: 'success', batch: 'warning' }[type] || ''
}
function formatTime(ts) {
  if (!ts) return ''
  return new Date(ts * 1000).toLocaleString('zh-CN')
}

async function loadHistory() {
  loading.value = true
  try {
    const { data } = await historyApi.list(userStore.user.username, 100)
    items.value = data.items
  } catch (e) {
    ElMessage.error('加载历史失败')
  } finally {
    loading.value = false
  }
}

async function handleDelete(id) {
  try {
    await ElMessageBox.confirm('确定删除这条记录？', '确认')
    await historyApi.delete(userStore.user.username, id)
    items.value = items.value.filter(i => i.id !== id)
    ElMessage.success('已删除')
  } catch {}
}

function reuseParams(item) {
  if (item.type === 'img2img') {
    router.push('/img2img')
  } else {
    router.push('/')
  }
  ElMessage.info('参数已复制，请手动填入')
}

onMounted(loadHistory)
</script>

<style scoped>
.history-header { display: flex; justify-content: space-between; align-items: center; }
.card-title { font-size: 16px; font-weight: 600; }
.history-filters { display: flex; gap: 12px; }
.loading-container { display: flex; justify-content: center; padding: 60px; }
.history-list { display: flex; flex-direction: column; gap: 16px; }
.history-item { display: flex; justify-content: space-between; padding: 16px; background: #fafafa; border-radius: 8px; gap: 16px; }
.history-left { flex: 1; min-width: 0; }
.history-meta { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }
.history-time { color: #909399; font-size: 13px; }
.history-elapsed { color: #67c23a; font-size: 13px; }
.history-prompt { font-weight: 500; color: #303133; margin-bottom: 4px; word-break: break-all; }
.history-params { color: #909399; font-size: 12px; }
.history-right { display: flex; flex-direction: column; align-items: flex-end; gap: 8px; }
.history-images { display: flex; gap: 8px; }
.history-thumb { width: 80px; height: 80px; border-radius: 4px; }
.history-actions { display: flex; gap: 4px; }
</style>
