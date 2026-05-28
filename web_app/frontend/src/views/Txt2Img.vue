<template>
  <div class="txt2img-page">
    <el-row :gutter="24">
      <el-col :span="8">
        <el-card class="param-card">
          <template #header><span class="card-title">生成参数</span></template>
          <el-form label-position="top">
            <el-form-item label="正面提示词">
              <el-input v-model="form.prompt" type="textarea" :rows="3" placeholder="描述你想要生成的图片..." />
            </el-form-item>
            <el-form-item label="负面提示词">
              <el-input v-model="form.negative_prompt" type="textarea" :rows="2" placeholder="不想要的内容..." />
            </el-form-item>

            <el-divider content-position="left">提示词模板</el-divider>
            <div class="template-tags">
              <el-tag v-for="t in templates" :key="t.name" class="template-tag" @click="applyTemplate(t)">
                {{ t.name }}
              </el-tag>
            </div>

            <el-divider content-position="left">参数设置</el-divider>
            <el-form-item label="采样步数">
              <el-slider v-model="form.steps" :min="10" :max="50" show-input />
            </el-form-item>
            <el-form-item label="引导系数 (CFG)">
              <el-slider v-model="form.guidance_scale" :min="1" :max="20" :step="0.5" show-input />
            </el-form-item>
            <el-form-item label="生成数量">
              <el-input-number v-model="form.num_images" :min="1" :max="8" />
            </el-form-item>
            <el-form-item label="随机种子">
              <el-input v-model.number="form.seed" placeholder="-1 表示随机">
                <template #append>
                  <el-button @click="form.seed = Math.floor(Math.random() * 999999999)">随机</el-button>
                </template>
              </el-input>
            </el-form-item>
            <el-form-item label="分辨率">
              <el-select v-model="form.resolution">
                <el-option label="512x512" :value="512" />
                <el-option label="768x768" :value="768" />
                <el-option label="512x768 (竖版)" :value="'512x768'" />
                <el-option label="768x512 (横版)" :value="'768x512'" />
              </el-select>
            </el-form-item>

            <el-button type="primary" size="large" :loading="generating" @click="handleGenerate" style="width:100%">
              <el-icon><Picture /></el-icon>
              {{ generating ? `生成中 ${progress}%` : '开始生成' }}
            </el-button>
            <el-progress v-if="generating" :percentage="progress" :stroke-width="8" style="margin-top:12px" />
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
          <div v-if="images.length" class="image-grid">
            <div v-for="(img, i) in images" :key="i" class="image-item">
              <el-image :src="'data:image/png;base64,' + img" fit="contain" :preview-src-list="previewList" :initial-index="i" class="gen-image" />
              <div class="image-actions">
                <el-button text size="small" @click="downloadImage(img, i)">
                  <el-icon><Download /></el-icon>下载
                </el-button>
              </div>
            </div>
          </div>
          <el-empty v-else description="输入提示词开始生成" />
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
const progress = ref(0)
const images = ref([])
const lastElapsed = ref(null)

const form = reactive({
  prompt: '',
  negative_prompt: 'low quality, blurry, distorted',
  steps: 30,
  guidance_scale: 7.5,
  num_images: 4,
  seed: null,
  resolution: 512,
})

const templates = [
  { name: '写实照片', prompt: 'a realistic photograph, detailed, high quality, 8k', negative: 'cartoon, anime, painting' },
  { name: '动漫风格', prompt: 'anime style, detailed illustration, vibrant colors', negative: 'realistic, photo' },
  { name: '油画风格', prompt: 'oil painting style, classical art, rich colors, detailed brushstrokes', negative: 'photo, digital art' },
  { name: '赛博朋克', prompt: 'cyberpunk style, neon lights, futuristic city, dark atmosphere', negative: 'natural, daylight' },
  { name: '水彩画', prompt: 'watercolor painting, soft colors, artistic, delicate', negative: 'photo, digital' },
  { name: '像素艺术', prompt: 'pixel art style, retro game, 16-bit', negative: 'realistic, high resolution' },
]

const previewList = computed(() => images.value.map(img => 'data:image/png;base64,' + img))

function applyTemplate(t) {
  form.prompt = (form.prompt ? form.prompt + ', ' : '') + t.prompt
  form.negative_prompt = t.negative
}

async function handleGenerate() {
  if (!form.prompt.trim()) return ElMessage.warning('请输入提示词')

  generating.value = true
  progress.value = 0
  images.value = []

  let width = 512, height = 512
  if (typeof form.resolution === 'string') {
    const [w, h] = form.resolution.split('x').map(Number)
    width = w; height = h
  } else {
    width = height = form.resolution
  }

  try {
    const { data } = await generateApi.txt2img({
      username: userStore.user.username,
      prompt: form.prompt,
      negative_prompt: form.negative_prompt,
      steps: form.steps,
      guidance_scale: form.guidance_scale,
      seed: form.seed >= 0 ? form.seed : null,
      num_images: form.num_images,
      width,
      height,
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
  link.download = `txt2img_${index}.png`
  link.click()
}
</script>

<style scoped>
.txt2img-page { max-width: 1400px; margin: 0 auto; }
.param-card, .result-card { height: calc(100vh - 120px); overflow-y: auto; }
.card-title { font-size: 16px; font-weight: 600; }
.template-tags { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }
.template-tag { cursor: pointer; transition: all 0.2s; }
.template-tag:hover { transform: scale(1.05); }
.result-header { display: flex; justify-content: space-between; align-items: center; }
.elapsed { color: #909399; font-size: 13px; }
.image-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; }
.image-item { border-radius: 8px; overflow: hidden; background: #fafafa; }
.gen-image { width: 100%; height: 300px; }
.image-actions { padding: 8px; text-align: center; }
</style>
