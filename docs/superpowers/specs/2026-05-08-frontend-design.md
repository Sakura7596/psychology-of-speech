# 前端界面设计文档

> **日期**：2026-05-08
> **状态**：已批准

## 技术栈

- Vue 3 (Composition API) + Vite
- Tailwind CSS
- axios（API 调用）
- marked（Markdown 渲染）

## 设计语言

- 基底：Notion/Linear 式克制设计
- 趣味：Agent 头像动画、置信度温度计、修辞图标标记
- 中文优化：思源黑体、宽松行距
- 色彩：中性灰为主，蓝绿色点缀（非渐变）

## 页面结构

### 输入区
- 大文本框，placeholder 引导
- 深度选择：三个按钮（快速/标准/深度）
- 开始分析按钮

### 进度区
- 5 个 Agent 状态卡片（头像 + 名称 + 进度条）
- 实时更新（轮询或 SSE）

### 报告区
- Markdown 渲染的分析报告
- Tab 切换：文本特征 / 心理动机 / 逻辑结构
- 置信度仪表盘
- 免责声明

## API 对接

- `POST /analyze` → `{text, depth, output_format}`
- `GET /health` → 健康检查
