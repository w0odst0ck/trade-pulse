#!/usr/bin/env python3
"""
build_ui.py — UI 构建统一入口

步骤：
  1. gen_data.py → docs/assets/data.json（原子写+校验）
  2. gen_dashboard.py → docs/dashboard.html
  3. 自动扫描 docs/*.html → 生成 docs/index.html 导航

任一步失败 → 非零退出码（cron 据此跳过 git commit，防止提交坏产物）

用法：
  python build_ui.py
  python build_ui.py --skip-data     # 跳过数据生成（调试）
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"

# 导航排序：index 永远第一，其余按名称
NAV_ORDER = ["index", "dashboard", "multi", "trade-log", "backtest"]


def step(name: str, cmd: list) -> bool:
    print(f"\n── {name} ──")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.stdout:
        print(r.stdout.strip())
    if r.returncode != 0:
        print(f"  [FAIL] {name}")
        if r.stderr:
            print(r.stderr.strip()[-500:])
        return False
    print(f"  [OK] {name}")
    return True


def get_page_title(path: Path) -> str:
    """从 HTML 的 <title> 提取页面名"""
    try:
        text = path.read_text(encoding="utf-8")
        m = re.search(r"<title>(.*?)</title>", text, re.S)
        return m.group(1).strip() if m else path.stem
    except Exception:
        return path.stem


def gen_index() -> bool:
    """扫描 docs/*.html 生成导航 index.html"""
    pages = []
    for p in sorted(DOCS_DIR.glob("*.html")):
        if p.name == "index.html":
            continue
        title = get_page_title(p)
        pages.append((p.stem, title))

    # 按 NAV_ORDER 排序，未列出的排后面
    def sort_key(item):
        stem = item[0]
        if stem in NAV_ORDER:
            return NAV_ORDER.index(stem)
        return 100

    pages.sort(key=sort_key)

    cards = []
    for stem, title in pages:
        cards.append(
            f'<a href="{stem}.html" class="card"><h2>{title}</h2>'
            f'<span class="desc">{stem}.html</span></a>'
        )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>trade-pulse 工具面板</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,Segoe UI,system-ui,sans-serif;background:#0d1117;color:#c9d1d9;padding:40px 20px;max-width:800px;margin:0 auto}}
h1{{font-size:22px;margin-bottom:8px}}
.sub{{color:#8b949e;font-size:14px;margin-bottom:24px}}
.card{{display:block;background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px 20px;margin-bottom:12px;text-decoration:none;color:#c9d1d9;transition:border-color .2s}}
.card:hover{{border-color:#58a6ff}}
.card h2{{font-size:16px;color:#e6edf3;margin-bottom:4px}}
.card .desc{{font-size:12px;color:#484f58}}
.ft{{text-align:center;color:#484f58;font-size:12px;padding:24px 0}}
</style>
</head>
<body>
<h1>trade-pulse 工具面板</h1>
<div class="sub">信号面板 / 数据可视化 / 交易记录</div>
{''.join(cards)}
<div class="ft">trade-pulse &#183; 自动生成</div>
</body>
</html>
"""
    out = DOCS_DIR / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"  [OK] {out} ({len(pages)} 个页面)")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-data", action="store_true", help="跳过数据生成")
    args = parser.parse_args()

    print("🔧 trade-pulse UI 构建")
    print("=" * 35)

    if not args.skip_data:
        if not step("生成共享数据", [sys.executable, str(SCRIPT_DIR / "gen_data.py")]):
            sys.exit(1)

    if not step("生成信号面板", [sys.executable, str(SCRIPT_DIR / "gen_dashboard.py")]):
        sys.exit(1)

    if not step("生成交易记录页", [sys.executable, str(SCRIPT_DIR / "gen_tradelog.py")]):
        sys.exit(1)

    if not gen_index():
        sys.exit(1)

    print(f"\n✅ UI 构建完成 → docs/")
    print(f"   本地预览: xdg-open {DOCS_DIR / 'index.html'}")


if __name__ == "__main__":
    main()
