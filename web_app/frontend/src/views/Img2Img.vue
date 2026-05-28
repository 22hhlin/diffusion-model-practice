<template>
  <div class="img2img-page">
    <el-row :gutter="24">
      <el-col :span="8">
        <el-card class="param-card">
          <template #header><span class="card-title">图生图参数</span></template>
          <el-form label-position="top">
            <el-form-item label="上传原图">
              <el-upload
                class="upload-area"
                :auto-upload="false"
                :show-file-list="false"
                :on-change="handleFileChange"
                accept="image/*"
                drag
              >
                <img v-if="previewUrl" :src="previewUrl" class="preview-img" />
                <div v-else class="upload-placeholder">
                  <el-icon size="40"><Upload /></el-icon>
                  <p>拖拽或点击上传图片</p>
                </div>
              </el-upload>
            </el-form-item>
            <el-form-item label="正面提示词">
              <el-input v-model="form.prompt" type="textarea" :rows="3" placeholder="描述你想要的效果..." />
            </el-form-item>
            <el-form-item label="负面提示词">
              <el-input v-model="form.negative_prompt" type="textarea" :rows="2" />
            </el-form-item>
            <el-form-item label="变化强度">
              <el-slider v-model="form.strength" :min="0.1" :max="1" :step="0.05" show-input />
              <div class="param-hint">值越大变化越大，值越小越接近原图</div>
            </el-form-item>
            <el-form-item label="采样步数">
              <el-slider v-model="form.steps" :min="10" :max="50" show-input />
            </el-form-item>
            <el-form-item label="引导系数">
              <el-slider v-model="form.guidance_scale" :min="1" :max="20" :step="0.5" show-input />
            </el-form-item>
            <el-form-item label="生成数量">
              <el-input-number v-model="form.num_images" :min="1" :max="4" />
            </el-form-item>
            <el-form-item label="随机种子">
              <el-input v-model.number="form.seed" placeholder="-1 表示随机" />
            </el-form-item>
            <el-button type="primary" size="large" :loading="generating" :disabled="!imageBase64" @click="handleGenerate" style="width:100%">
              {{ generating ? '生成中...' : '开始生成' }}
            </el-button>
          </el-form>
        </el-card>
      </el-col>

      <el-col :span="16">
        <el-card class="result-card">
          <template #header>
            <div class="result-header">
              <span class="card-title">生成结果</span>
              <span v-if="lastElapsed" class="elapsed">耗时 {{ lastElapsed }}s</span>
            </div>
          </template>
          <div v-if="images.length" class="compare-container">
            <div class="compare-item" v-for="(img, i) in images" :key="i">
              <div class="compare-side">
                <p class="compare-label">原图</p>
                <img :src="previewUrl" class="compare-img" />
              </div>
              <div class="compare-side">
                <p class="compare-label">生成图</p>
                <el-image :src="'data:image/png;base64,' + img" fit="contain" :preview-src-list="previewList" :initial-index="i" class="compare-img" />
                <el-button text size="small" @click="downloadImage(img, i)" style="margin-top:8px">
                  <el-icon><Download /></el-icon>下载
                </el-button>
              </div>
            </div>
          </div>
          <el-empty v-else description="上传图片并输入提示词开始生成" />
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
const images = ref([])
const lastElapsed = ref(null)
const imageBase64 = ref('')
const previewUrl = ref('')

const form = reactive({
  prompt: '',
  negative_prompt: 'low quality, blurry, distorted',
  strength: 0.75,
  steps: 30,
  guidance_scale: 7.5,
  num_images: 1,
  seed: null,
})

const previewList = computed(() => images.value.map(img => 'data:image/png;base64,' + img))

function handleFileChange(file) {
  const reader = new FileReader()
  reader.onload = (e) => {
    previewUrl.value = e.target.result
    imageBase64.value = e.target.result.split(',')[1]
  }
  reader.readAsDataURL(file.raw)
}

async function handleGenerate() {
  if (!imageBase64.value) return ElMessage.warning('请上传图片')
  if (!form.prompt.trim()) return ElMessage.warning('请输入提示词')

  generating.value = true
  images.value = []

  try {
    const { data } = await generateApi.img2img({
      username: userStore.user.username,
      prompt: form.prompt,
      image_base64: imageBase64.value,
      negative_prompt: form.negative_prompt,
      strength: form.strength,
      steps: form.steps,
      guidance_scale: form.guidance_scale,
      seed: form.seed >= 0 ? form.seed : null,
      num_images: form.num_images,
    })
    images.value = data.images
    lastElapsed.value = data.elapsed
    ElMessage.success(`生成完成，耗时 ${data.elapsed}s`)
  } catch (e) {
    ElMessage.error(e.response?.data?.detail || '生成失败')
  } finally {
    generating.value = false
  }
}

function downloadImage(b64, index) {
  const link = document.createElement('a')
  link.href = 'data:image/png;base64,' + b64
  link.download = `img2img_${index}.png`
  link.click()
}
</script>

<style scoped>
.img2img-page { max-width: 1400px; margin: 0 auto; }
.param-card, .result-card { height: calc(100vh - 120px); overflow-y: auto; }
.card-title { font-size: 16px; font-weight: 600; }
.upload-area :deep(.el-upload-dragger) { padding: 16px; }
.preview-img { max-width: 100%; max-height: 200px; border-radius: 8px; }
.upload-placeholder { color: #909399; }
.param-hint { color: #909399; font-size: 12px; margin-top: 4px; }
.result-header { display: flex; justify-content: space-between; align-items: center; }
.elapsed { color: #909399; font-size: 13px; }
.compare-container { display: flex; flex-direction: column; gap: 24px; }
.compare-item { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; padding: 16px; background: #fafafa; border-radius: 8px; }
.compare-label { font-weight: 600; color: #606266; margin-bottom: 8px; text-align: center; }
.compare-img { width: 100%; max-height: 300px; object-fit: contain; border-radius: 4px; }
</style>
