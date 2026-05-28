<template>
  <div class="batch-page">
    <el-row :gutter="24">
      <el-col :span="8">
        <el-card class="param-card">
          <template #header><span class="card-title">批量生成</span></template>
          <el-form label-position="top">
            <el-form-item label="提示词列表（每行一个）">
              <el-input v-model="promptsText" type="textarea" :rows="8" placeholder="a beautiful sunset&#10;a cute cat&#10;cyberpunk city" />
            </el-form-item>
            <el-form-item>
              <el-tag v-for="(_, i) in promptList" :key="i" closable @close="removePrompt(i)" style="margin: 2px 4px;">
                {{ promptList[i].slice(0, 20) }}{{ promptList[i].length > 20 ? '...' : '' }}
              </el-tag>
            </el-form-item>
            <el-form-item label="负面提示词">
              <el-input v-model="form.negative_prompt" type="textarea" :rows="2" />
            </el-form-item>
            <el-form-item label="采样步数">
              <el-slider v-model="form.steps" :min="10" :max="50" show-input />
            </el-form-item>
            <el-form-item label="引导系数">
              <el-slider v-model="form.guidance_scale" :min="1" :max="20" :step="0.5" show-input />
            </el-form-item>
            <el-form-item label="随机种子">
              <el-input v-model.number="form.seed" placeholder="-1 表示随机" />
            </el-form-item>
            <el-button type="primary" size="large" :loading="generating" :disabled="!promptList.length" @click="handleBatch" style="width:100%">
              {{ generating ? `生成中 (${doneCount}/${promptList.length})` : `批量生成 (${promptList.length} 条)` }}
            </el-button>
            <el-progress v-if="generating" :percentage="Math.round(doneCount / promptList.length * 100)" :stroke-width="8" style="margin-top:12px" />
          </el-form>
        </el-card>
      </el-col>

      <el-col :span="16">
        <el-card class="result-card">
          <template #header>
            <div class="result-header">
              <span class="card-title">生成结果</span>
              <span v-if="lastElapsed" class="elapsed">总耗时 {{ lastElapsed }}s</span>
            </div>
          </template>
          <div v-if="results.length" class="batch-results">
            <div v-for="(r, i) in results" :key="i" class="batch-item">
              <div class="batch-prompt">{{ r.prompt }}</div>
              <div v-if="r.ok" class="batch-images">
                <el-image v-for="(img, j) in r.images" :key="j" :src="'data:image/png;base64,' + img" fit="contain" class="batch-img" :preview-src-list="getPreviewList(r.images)" />
              </div>
              <el-alert v-else :title="r.error" type="error" show-icon :closable="false" />
            </div>
          </div>
          <el-empty v-else description="输入多个提示词开始批量生成" />
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { ref, reactive, computed } from 'vue'
import { ElMessage } from 'element-plus'
import { generateApi } from '../api'
import { useUserStore } from '../store/user'

const userStore = useUserStore()
const generating = ref(false)
const promptsText = ref('')
const doneCount = ref(0)
const results = ref([])
const lastElapsed = ref(null)

const form = reactive({
  negative_prompt: 'low quality, blurry, distorted',
  steps: 30,
  guidance_scale: 7.5,
  seed: null,
})

const promptList = computed(() => promptsText.value.split('\n').map(s => s.trim()).filter(Boolean))

function removePrompt(index) {
  const lines = promptsText.value.split('\n')
  lines.splice(index, 1)
  promptsText.value = lines.join('\n')
}

function getPreviewList(images) {
  return images.map(img => 'data:image/png;base64,' + img)
}

async function handleBatch() {
  if (!promptList.value.length) return ElMessage.warning('请输入至少一个提示词')

  generating.value = true
  doneCount.value = 0
  results.value = []

  try {
    const { data } = await generateApi.batch({
      username: userStore.user.username,
      prompts: promptList.value,
      negative_prompt: form.negative_prompt,
      steps: form.steps,
      guidance_scale: form.guidance_scale,
      seed: form.seed !== null && form.seed >= 0 ? form.seed : null,
    })
    results.value = data.results
    doneCount.value = promptList.value.length
    lastElapsed.value = data.elapsed
    ElMessage.success(`批量生成完成，共 ${data.results.length} 条`)
  } catch (e) {
    ElMessage.error(e.response?.data?.detail || '批量生成失败')
  } finally {
    generating.value = false
  }
}
</script>

<style scoped>
.batch-page { max-width: 1400px; margin: 0 auto; }
.param-card, .result-card { height: calc(100vh - 120px); overflow-y: auto; }
.card-title { font-size: 16px; font-weight: 600; }
.result-header { display: flex; justify-content: space-between; align-items: center; }
.elapsed { color: #909399; font-size: 13px; }
.batch-results { display: flex; flex-direction: column; gap: 16px; }
.batch-item { padding: 16px; background: #fafafa; border-radius: 8px; }
.batch-prompt { font-weight: 600; color: #303133; margin-bottom: 12px; }
.batch-images { display: flex; gap: 12px; flex-wrap: wrap; }
.batch-img { width: 200px; height: 200px; border-radius: 4px; }
</style>
