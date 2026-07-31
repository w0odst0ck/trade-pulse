#!/usr/bin/env python3
"""
sync_awesome.py — 上游 awesome-systematic-trading 同步脚本

功能：
  1. 拉取上游 README.md
  2. 解析表格行，提取 repo 链接 + 描述（README 自带，零 API 依赖）
  3. 与本地索引已收录链接比对
  4. 过滤排除类别（crypto/HFT/非Python）
  5. 新增条目 → 追加到候选池

用法：
  python sync_awesome.py              # 正常同步
  python sync_awesome.py --dry-run    # 只输出 diff 不修改文件
  python sync_awesome.py --show-local # 显示本地已收录链接
"""

import argparse
import re
import sys
from datetime import date
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).parent.parent
INDEX_PATH = PROJECT_ROOT / "docs" / "工具库索引.md"

UPSTREAM_URL = "https://raw.githubusercontent.com/paperswithbacktest/awesome-systematic-trading/master/README.md"

# 候选池锚点（追加位置）
POOL_ANCHOR = "## 2. 候选池（随上游自动更新，未评估）"

# 排除类别关键词：repo 名或描述命中即跳过（对应排除清单）
EXCLUDE_KEYWORDS = [
    # 加密货币专项
    "bitcoin", "crypto", "binance", "stellar", "hummingbot", "freqtrade",
    "octobot", "ccxt", "cryptofeed", "orderbook", "coin", "arbitrag",
    "blackbird", "jesse", "kelp", "bittrex",
    # HFT / tick 级
    "hft", "tick", "high-frequency", "microstructure", "flashfunk",
    # 非 Python 技术栈
    "rust", "ocaml", "made with go", "made with c++", "made-with-go",
    "made-with-c++", "tectonicdb", "openlimits", "ta-rs", "gobacktest",
    "incremental", "graphkit", "mdf", "tributary", "marketstore",
    # 通用数据科学库（不是金融工具）
    "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy",
    "scipy", "pymc", "cvxpy", "ray", "dask", "prophet", "statsmodels",
    "tsfresh", "pmdarima", "pandas-datareader", "quandl",
    # 明确排除
    "openbb",
]


def fetch_upstream() -> str:
    r = requests.get(UPSTREAM_URL, timeout=30)
    r.raise_for_status()
    return r.text


def parse_table_rows(text: str) -> dict:
    """解析 README 表格：{repo: description}"""
    result = {}
    # 表格行格式: | [name](https://github.com/owner/repo) | description | ...
    pattern = re.compile(
        r"\|\s*\[[^\]]*\]\(https?://github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)\)\s*"
        r"\|\s*([^|]+)\|"
    )
    for m in pattern.finditer(text):
        repo = m.group(1).rstrip("/")
        desc = m.group(2).strip()
        result[repo] = desc
    return result


def extract_local_links(index_text: str) -> set:
    """从索引文件提取已收录的 github 链接"""
    links = set()
    for m in re.finditer(r"github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)", index_text):
        repo = m.group(1).rstrip("/")
        links.add(repo)
    return links


def is_excluded(repo: str, desc: str = "") -> bool:
    """判断是否属于已知排除类别"""
    text = f"{repo} {desc}".lower()
    return any(kw in text for kw in EXCLUDE_KEYWORDS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="只输出 diff 不改文件")
    parser.add_argument("--show-local", action="store_true", help="显示本地已收录")
    args = parser.parse_args()

    index_text = INDEX_PATH.read_text(encoding="utf-8")
    local_links = extract_local_links(index_text)
    if args.show_local:
        print(f"本地已收录 {len(local_links)} 个链接:")
        for l in sorted(local_links):
            print(f"  {l}")
        return

    print("拉取上游 README...")
    upstream = fetch_upstream()
    upstream_rows = parse_table_rows(upstream)
    print(f"上游表格条目: {len(upstream_rows)} 个 | 本地已收录: {len(local_links)} 个")

    # 新增（未收录的）
    new_items = {k: v for k, v in upstream_rows.items() if k not in local_links}
    print(f"新增(原始): {len(new_items)} 个")

    # 过滤排除类别
    filtered = {}
    for repo, desc in sorted(new_items.items()):
        if is_excluded(repo, desc):
            print(f"  ⛔ 排除: {repo} ({desc[:40]})")
            continue
        filtered[repo] = desc

    print(f"新增(过滤后): {len(filtered)} 个")

    if not filtered:
        print("无值得评估的新增，索引已是最新")
        return

    for repo, desc in sorted(filtered.items()):
        print(f"  🆕 {repo} — {desc[:50]}")

    if args.dry_run:
        return

    # 构建候选池追加内容
    today = date.today().isoformat()
    rows = []
    for repo, desc in sorted(filtered.items()):
        name = repo.split("/")[-1]
        rows.append(f"| {name} | github.com/{repo} | {desc} | 🆕 {today} |")

    block = (
        f"\n### 上游新增 {today}（自动同步）\n\n"
        f"| 名称 | 链接 | 描述 | 状态 |\n"
        f"|:---|:---|:---|:---:|\n" + "\n".join(rows) + "\n"
    )

    # 插入到候选池锚点之后
    idx = index_text.find(POOL_ANCHOR)
    if idx == -1:
        print("[ERR] 找不到候选池锚点，未写入")
        sys.exit(1)

    section_end = index_text.find("\n## ", idx + len(POOL_ANCHOR))
    if section_end == -1:
        section_end = len(index_text)

    new_text = index_text[:section_end] + block + index_text[section_end:]
    INDEX_PATH.write_text(new_text, encoding="utf-8")
    print(f"\n[OK] {len(filtered)} 条已追加到候选池 → {INDEX_PATH}")


if __name__ == "__main__":
    main()
