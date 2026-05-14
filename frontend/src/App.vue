<script setup>
import { ref, computed } from 'vue'
import axios from 'axios'
import { marked } from 'marked'
import DOMPurify from 'dompurify'

const text = ref('')
const depth = ref('standard')
const loading = ref(false)
const streaming = ref(false)
const result = ref(null)
const error = ref('')
const activeTab = ref('report')
const streamReport = ref('')

const agents = [
  { name: '文本解析', icon: '📝', key: 'text_analyst' },
  { name: '心理分析', icon: '🧠', key: 'psychology_analyst' },
  { name: '逻辑推理', icon: '🔍', key: 'logic_analyst' },
  { name: '报告生成', icon: '📊', key: 'report_generator' },
]

const agentProgress = ref({})

// 保存完整文本用于历史恢复
let lastFullText = ''

const exampleTexts = [
  { label: '职场对话', text: '领导说："这个方案还需要再完善一下。"小王回答："好的，我会尽快修改。"但他心里想："每次都是这样，要求不明确就让人改。"' },
  { label: '情感表达', text: '虽然我们已经分手了，但是我还是会想起那些美好的时光。也许时间会治愈一切，但此刻的我真的很痛苦。' },
  { label: '辩论观点', text: '你支持环保？那你是想回到原始社会吗？经济发展和环境保护根本不可能兼顾，看看那些发达国家，哪个不是先污染后治理？' },
  { label: '销售话术', text: '这款产品是我们的爆款，今天是最后一天特价，过了今天就恢复原价了。您看，已经有很多人下单了，机会难得啊。' },
]

function useExample(example) {
  text.value = example.text
}

async function analyze() {
  if (!text.value.trim()) return
  loading.value = true
  streaming.value = true
  error.value = ''
  result.value = null
  streamReport.value = ''
  agentProgress.value = {}
  lastFullText = text.value

  try {
    // 使用 SSE 流式接口
    const response = await fetch('/api/analyze/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: text.value,
        depth: depth.value,
        output_format: 'markdown'
      })
    })

    if (!response.ok) {
      const errData = await response.json()
      throw new Error(errData.detail || '分析失败')
    }

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue
        let data
        try {
          data = JSON.parse(line.slice(6))
        } catch {
          console.warn('SSE JSON 解析失败:', line)
          continue
        }

        if (data.type === 'progress') {
          agentProgress.value[data.agent] = data.status
        } else if (data.type === 'chunk') {
          streamReport.value += data.content
        } else if (data.type === 'done') {
          result.value = {
            report: data.report || streamReport.value,
            analyses: data.analyses || {},
            confidence: data.confidence || 0.5,
            depth: data.depth || depth.value
          }
          agents.forEach(a => agentProgress.value[a.key] = 'done')
        } else if (data.type === 'error') {
          throw new Error(data.error)
        }
      }
    }

    // 如果没有收到 done 事件但有流式内容
    if (!result.value && streamReport.value) {
      result.value = {
        report: streamReport.value,
        analyses: {},
        confidence: 0.5,
        depth: depth.value
      }
    }
  } catch (e) {
    error.value = e.message || '分析失败，请重试'
  } finally {
    loading.value = false
    streaming.value = false
  }
}

function retry() {
  error.value = ''
  analyze()
}

function renderMarkdown(md) {
  const html = marked(md || '')
  return DOMPurify.sanitize(html)
}

function confidenceColor(c) {
  if (c >= 0.7) return 'text-emerald-600'
  if (c >= 0.4) return 'text-amber-600'
  return 'text-red-500'
}

function confidenceWidth(c) {
  return `${Math.round(c * 100)}%`
}

// 从 localStorage 加载历史
const history = ref(JSON.parse(localStorage.getItem('analysis_history') || '[]'))

function saveToHistory() {
  if (!result.value) return
  const fullText = lastFullText || text.value
  const entry = {
    id: Date.now(),
    text: fullText.slice(0, 50) + (fullText.length > 50 ? '...' : ''),
    fullText: fullText,
    depth: depth.value,
    confidence: result.value.confidence,
    timestamp: new Date().toLocaleString('zh-CN'),
  }
  history.value.unshift(entry)
  if (history.value.length > 20) history.value.pop()
  localStorage.setItem('analysis_history', JSON.stringify(history.value))
}

function loadFromHistory(entry) {
  text.value = entry.fullText || entry.text
  depth.value = entry.depth
}
</script>

<template>
  <div class="min-h-screen bg-stone-50 text-stone-800">
    <!-- Header -->
    <header class="border-b border-stone-200 bg-white">
      <div class="max-w-4xl mx-auto px-6 py-4 flex items-center justify-between">
        <div>
          <h1 class="text-xl font-semibold tracking-tight">语言心理学话语分析</h1>
          <p class="text-sm text-stone-400 mt-0.5">基于多智能体协作的文本深度解析</p>
        </div>
        <div class="flex items-center gap-2 text-xs text-stone-400">
          <span class="w-2 h-2 rounded-full bg-emerald-400"></span>
          v0.2.0
        </div>
      </div>
    </header>

    <main class="max-w-4xl mx-auto px-6 py-8 space-y-8">
      <!-- Input Section -->
      <section class="space-y-4">
        <div class="relative">
          <textarea
            v-model="text"
            rows="6"
            placeholder="在这里输入或粘贴要分析的文本..."
            class="w-full p-4 rounded-lg border border-stone-200 bg-white text-sm leading-relaxed placeholder:text-stone-300 focus:outline-none focus:ring-2 focus:ring-stone-300 focus:border-transparent resize-none transition"
          ></textarea>
          <div class="absolute bottom-2 right-2 text-xs text-stone-300">
            {{ text.length }} 字
          </div>
        </div>

        <!-- Example Texts -->
        <div class="flex flex-wrap gap-2">
          <span class="text-xs text-stone-400 self-center">示例：</span>
          <button
            v-for="ex in exampleTexts"
            :key="ex.label"
            @click="useExample(ex)"
            class="px-2.5 py-1 rounded-md text-xs bg-white border border-stone-200 text-stone-500 hover:border-stone-400 hover:text-stone-700 transition"
          >
            {{ ex.label }}
          </button>
        </div>

        <div class="flex items-center justify-between">
          <div class="flex gap-2">
            <button
              v-for="d in ['quick', 'standard', 'deep']"
              :key="d"
              @click="depth = d"
              :class="[
                'px-3 py-1.5 rounded-md text-sm transition',
                depth === d
                  ? 'bg-stone-800 text-white'
                  : 'bg-white border border-stone-200 text-stone-500 hover:border-stone-400'
              ]"
            >
              {{ { quick: '快速', standard: '标准', deep: '深度' }[d] }}
            </button>
          </div>

          <button
            @click="analyze"
            :disabled="loading || !text.trim()"
            :class="[
              'px-5 py-2 rounded-lg text-sm font-medium transition',
              loading || !text.trim()
                ? 'bg-stone-200 text-stone-400 cursor-not-allowed'
                : 'bg-stone-800 text-white hover:bg-stone-700 active:scale-95'
            ]"
          >
            {{ loading ? '分析中...' : '开始分析' }}
          </button>
        </div>
      </section>

      <!-- Agent Progress -->
      <section v-if="loading || result" class="space-y-3">
        <h2 class="text-sm font-medium text-stone-500">分析进度</h2>
        <div class="grid grid-cols-2 md:grid-cols-4 gap-3">
          <div
            v-for="agent in agents"
            :key="agent.key"
            :class="[
              'p-3 rounded-lg border transition-all duration-300',
              agentProgress[agent.key] === 'done'
                ? 'border-emerald-200 bg-emerald-50'
                : agentProgress[agent.key] === 'running'
                ? 'border-amber-200 bg-amber-50'
                : agentProgress[agent.key] === 'error'
                ? 'border-red-200 bg-red-50'
                : 'border-stone-100 bg-white'
            ]"
          >
            <div class="flex items-center gap-2 mb-2">
              <span class="text-lg">{{ agent.icon }}</span>
              <span class="text-sm font-medium">{{ agent.name }}</span>
            </div>
            <div class="h-1.5 rounded-full bg-stone-100 overflow-hidden">
              <div
                :class="[
                  'h-full rounded-full transition-all duration-500',
                  agentProgress[agent.key] === 'done'
                    ? 'bg-emerald-400 w-full'
                    : agentProgress[agent.key] === 'running'
                    ? 'bg-amber-400 w-3/4 animate-pulse'
                    : agentProgress[agent.key] === 'error'
                    ? 'bg-red-400 w-full'
                    : 'bg-stone-200 w-0'
                ]"
              ></div>
            </div>
          </div>
        </div>
      </section>

      <!-- Error -->
      <div v-if="error" class="p-4 rounded-lg bg-red-50 border border-red-200 text-red-600 text-sm flex items-center justify-between">
        <span>{{ error }}</span>
        <button @click="retry" class="px-3 py-1 rounded bg-red-100 hover:bg-red-200 text-red-700 text-xs font-medium transition">
          重试
        </button>
      </div>

      <!-- Streaming Preview -->
      <section v-if="streaming && streamReport" class="space-y-3">
        <h2 class="text-sm font-medium text-stone-500">
          报告生成中
          <span class="inline-block animate-pulse">...</span>
        </h2>
        <div class="bg-white rounded-lg border border-stone-200 p-6">
          <div class="prose prose-sm prose-stone max-w-none" v-html="renderMarkdown(streamReport)"></div>
        </div>
      </section>

      <!-- Results -->
      <section v-if="result && !streaming" class="space-y-6">
        <!-- Confidence -->
        <div class="flex items-center gap-4 p-4 bg-white rounded-lg border border-stone-200">
          <div class="flex-1">
            <div class="flex items-center justify-between mb-1">
              <span class="text-sm text-stone-500">综合置信度</span>
              <span :class="['text-sm font-semibold', confidenceColor(result.confidence)]">
                {{ (result.confidence * 100).toFixed(0) }}%
              </span>
            </div>
            <div class="h-2 rounded-full bg-stone-100 overflow-hidden">
              <div
                class="h-full rounded-full transition-all duration-700"
                :class="[
                  result.confidence >= 0.7 ? 'bg-emerald-400' :
                  result.confidence >= 0.4 ? 'bg-amber-400' : 'bg-red-400'
                ]"
                :style="{ width: confidenceWidth(result.confidence) }"
              ></div>
            </div>
          </div>
          <div class="text-right">
            <div class="text-xs text-stone-400">分析深度</div>
            <div class="text-sm font-medium">{{ { quick: '快速', standard: '标准', deep: '深度' }[result.depth] }}</div>
          </div>
        </div>

        <!-- Tabs -->
        <div class="flex gap-1 p-1 bg-stone-100 rounded-lg w-fit">
          <button
            v-for="tab in [
              { key: 'report', label: '分析报告' },
              { key: 'analyses', label: '详细数据' },
            ]"
            :key="tab.key"
            @click="activeTab = tab.key"
            :class="[
              'px-4 py-1.5 rounded-md text-sm transition',
              activeTab === tab.key
                ? 'bg-white text-stone-800 shadow-sm'
                : 'text-stone-500 hover:text-stone-700'
            ]"
          >
            {{ tab.label }}
          </button>
        </div>

        <!-- Report Content -->
        <div v-if="activeTab === 'report'" class="bg-white rounded-lg border border-stone-200 p-6">
          <div class="prose prose-sm prose-stone max-w-none" v-html="renderMarkdown(result.report)"></div>
        </div>

        <!-- Analyses Data -->
        <div v-if="activeTab === 'analyses'" class="space-y-4">
          <div
            v-for="(data, name) in result.analyses"
            :key="name"
            class="bg-white rounded-lg border border-stone-200 p-4"
          >
            <h3 class="text-sm font-medium text-stone-600 mb-3 flex items-center gap-2">
              <span>{{ agents.find(a => a.key === name)?.icon || '📋' }}</span>
              {{ agents.find(a => a.key === name)?.name || name }}
            </h3>
            <pre class="text-xs text-stone-500 bg-stone-50 p-3 rounded overflow-x-auto">{{ JSON.stringify(data, null, 2) }}</pre>
          </div>
        </div>

        <!-- Disclaimer -->
        <div class="text-xs text-stone-400 p-3 bg-stone-50 rounded-lg border border-stone-100">
          ⚠️ 本分析基于语言学特征的辅助分析，仅供参考，不构成专业心理咨询、诊断或治疗建议。
        </div>
      </section>

      <!-- History -->
      <section v-if="history.length > 0 && !loading && !result" class="space-y-3">
        <h2 class="text-sm font-medium text-stone-500">最近分析</h2>
        <div class="space-y-2">
          <button
            v-for="entry in history.slice(0, 5)"
            :key="entry.id"
            @click="loadFromHistory(entry)"
            class="w-full p-3 rounded-lg border border-stone-200 bg-white text-left hover:border-stone-400 transition"
          >
            <div class="flex items-center justify-between">
              <span class="text-sm text-stone-600 truncate flex-1">{{ entry.text }}</span>
              <div class="flex items-center gap-2 ml-3">
                <span :class="['text-xs font-medium', confidenceColor(entry.confidence)]">
                  {{ (entry.confidence * 100).toFixed(0) }}%
                </span>
                <span class="text-xs text-stone-400">{{ entry.timestamp }}</span>
              </div>
            </div>
          </button>
        </div>
      </section>
    </main>
  </div>
</template>
