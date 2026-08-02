# trade-pulse

> 状态: ✅ 有效
> 角色: 导航总索引
> 最后更新: 2026-08-02

Rule-driven quantitative trading framework — from daily signals to live execution. Supports multi-asset, multi-strategy, extensible.

标的：588000 科创50ETF 日线择时 | 链路：数据 → 特征 → 信号 → 推送 → 回测 → 实盘

---

## 📚 文档地图

### 规划（根目录）

| 文档 | 内容 | 状态 |
|:---|:---|:---:|
| [项目目标.md](项目目标.md) | 终极目标 / 当前状态 / 未来路线图 / 决策点 / 实盘 SOP | ✅ |
| [实战推进计划.md](实战推进计划.md) | M1→M2→M3 任务执行清单 | ✅ |

### 设计 + 调研（docs/）

| 文档 | 内容 | 状态 |
|:---|:---|:---:|
| [docs/信号规则系统设计.md](docs/信号规则系统设计.md) | 信号系统架构（权重/阈值以 config.json 为准） | ✅ |
| [docs/工具库索引.md](docs/工具库索引.md) | 工具库活文档（cron 每月同步上游） | ✅ |
| [docs/工具库调研_2026-08-02.md](docs/工具库调研_2026-08-02.md) | 工具库市场调研结论 | ✅ |
| [docs/组合优化工具调研.md](docs/组合优化工具调研.md) | 配仓方案调研（自研波动率倒数加权） | ✅ |
| [docs/Kronos 结论.md](docs/Kronos 结论.md) | Kronos 一页纸结论（❌ 无预测力） | ✅ |

### 日志（memory/）

| 内容 | 状态 |
|:---|:---:|
| 每日项目日志 2026-07-27 ~ 至今 | ✅ |
| 技术提炼.md（逐笔管线，参考） | 📦 参考 |
| 学习计划.md（Level-2 微观结构，已过时） | ⚠️ 已过时 |

### 归档（archive/ + docs/archive/）

| 文档 | 说明 |
|:---|:---|
| archive/项目计划.md | v1 规划（2026-07-29），被项目目标 v2 吸收 |
| docs/archive/InstOrderTrace 生态参考报告.md | 旧名生态参考（v1） |
| docs/archive/生态参考报告_v2.md | 生态参考 v2（Kronos 结论已过时） |
| docs/archive/Kronos验证报告.md 等 3 份 | Kronos 详细验证报告（证据链存档） |

---

## 🚀 快速入口

```bash
# 每日信号（14:25 cron 自动跑，手动亦可）
cd tools/daily_pipeline && python3 daily_panel.py --push

# 回测 + 绩效报告
cd tools/daily_pipeline && python3 backtest.py

# 健康检查（六查）
cd tools/daily_pipeline && python3 health_check.py

# UI 构建（GitHub Pages）
cd tools/ui && python3 build_ui.py
```

线上仪表盘: <https://w0odst0ck.github.io/trade-pulse/>

---

## ⚙️ 技术栈速览

| 层 | 组件 |
|:---|:---|
| 数据 | AkShare / EastMoney / baostock（三级 fallback）+ 质量闸门 |
| 特征 | momentum / trend / volume_price / rsrs（4 因子均分 0.25） |
| 信号 | 周线过滤 → 连续打分 → 三级状态机 → 30-70% 连续仓位 |
| 回测 | 自研 backtest.py（因子归因/分段归因/参数搜索/风险指标） |
| 推送 | 飞书 WyrmGate bot（信号卡片） |
| 调度 | OpenClaw cron（14:25/14:30/15:10/15:30） |
| UI | GitHub Pages + 自研 Canvas（零依赖） |
