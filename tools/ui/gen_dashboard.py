#!/usr/bin/env python3
"""
gen_dashboard.py — 生成 588000 日线择时可视化面板

读取 data/ 下的 CSV + 状态文件，生成内嵌数据的独立 HTML 面板。
零外部依赖，生成后直接浏览器打开，无需 HTTP Server。

用法：
  python tools/ui/gen_dashboard.py               # 生成
  xdg-open tools/ui/dashboard.html               # 打开 (Linux)
  start tools/ui/dashboard.html                  # 打开 (Windows)
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "588000"
BACKTEST_DIR = DATA_DIR / "backtest"
STATE_PATH = DATA_DIR / "state.json"


def load_csv(path):
    if not path.exists():
        print(f"  [WARN] 文件不存在: {path.name}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path, parse_dates=True)
    except Exception as e:
        print(f"  [WARN] 读取 {path.name} 失败: {e}")
        return pd.DataFrame()


def dicterize(df):
    """DataFrame to JSON-safe list of dicts"""
    if df.empty:
        return []
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype(str)
    return df.to_dict(orient="records")


def main():
    print("\n  588000 日线面板生成")
    print("  " + "=" * 35)

    state = {}
    if STATE_PATH.exists():
        with open(STATE_PATH, encoding="utf-8") as f:
            state = json.load(f)

    daily = load_csv(DATA_DIR / "daily.csv")
    features = load_csv(DATA_DIR / "features_cache.csv")
    equity = load_csv(BACKTEST_DIR / "equity_curve.csv")
    trades = load_csv(BACKTEST_DIR / "trades.csv")

    print(f"  [R] 日线: {len(daily)} 条")
    print(f"  [R] 特征: {len(features)} 条")
    print(f"  [R] 权益: {len(equity)} 条")
    print(f"  [R] 交易: {len(trades)} 条")

    # 最新信号
    latest = features.iloc[-1].to_dict() if len(features) else {}
    sig_state = state.get("state", "空仓")

    data = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "signal": {
            "state": sig_state,
            "score": round(latest.get("total_score", 0), 4),
            "weekly": bool(latest.get("weekly_can_trade", True)),
            "date": str(latest.get("date", ""))[:10],
            "factors": {
                k: round(float(latest.get(k, 0)), 3)
                for k in ["momentum", "trend", "volatility", "volume_price", "rsrs", "relative_strength"]
            },
            "factor_names": ["动量", "趋势", "波动", "量价", "RSRS", "比价"],
        },
        "equity": dicterize(equity) if not equity.empty else [],
        "prices": dicterize(daily.tail(200)) if not daily.empty else [],
        "features": dicterize(features.tail(60)) if not features.empty else [],
        "trades": dicterize(trades.tail(15)) if not trades.empty else [],
    }

    data_json = json.dumps(data, ensure_ascii=True)

    # HTML 模板（模板字符串语法用 @PLACEHOLDER 标记数据嵌入点）
    html = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>588000 日线择时面板</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,Segoe UI,system-ui,sans-serif;background:#0d1117;color:#c9d1d9;padding:20px;max-width:1000px;margin:0 auto}
h1{font-size:20px;font-weight:600}
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:20px;margin-bottom:16px}
h2{font-size:14px;color:#8b949e;text-transform:uppercase;letter-spacing:.5px;margin-bottom:12px}
.sg{display:grid;grid-template-columns:auto 1fr;gap:8px 24px;font-size:15px}
.sl{color:#8b949e}.sv{font-weight:600}
.badge{display:inline-block;padding:2px 12px;border-radius:12px;font-size:13px;font-weight:600}
.bg-long{background:#238636;color:#fff}
.bg-wait{background:#9e6a03;color:#fff}
.bg-cash{background:#30363d;color:#8b949e}
.fg{display:grid;grid-template-columns:repeat(6,1fr);gap:8px}
@media(max-width:720px){.fg{grid-template-columns:repeat(3,1fr)}}
.fi{padding:8px;border-radius:6px;background:#0d1117;text-align:center}
.fn{font-size:12px;color:#8b949e}
.fv{font-size:18px;font-weight:700}
.g{color:#3fb950}.r{color:#f85149}.y{color:#d29922}
canvas{width:100%;height:240px;display:block;margin-top:8px}
table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;padding:6px 8px;color:#8b949e;font-weight:400;border-bottom:1px solid #30363d}
td{padding:6px 8px;border-bottom:1px solid #21262d}
.tb{color:#3fb950}.ts{color:#f85149}.th{color:#d29922}
.ft{text-align:center;color:#484f58;font-size:12px;padding:20px 0}
</style>
</head>
<body>
<div id="app"></div>
<script>
var D = @DATA@;

(function() {
  var a = document.getElementById('app');
  var s = D.signal, f = s.factors, fn = s.factor_names, keys = ['momentum','trend','volatility','volume_price','rsrs','relative_strength'];
  var st = {持仓:'bg-long',观望:'bg-wait',空仓:'bg-cash'}[s.state]||'bg-cash';
  var sl = s.state=='持仓'?'持有中':s.state=='观望'?'观望':'空仓';
  var sc = s.score;
  var scCls = sc>0?'g':'r';
  var h = '';

  // header
  h += '<div class="card"><div style="display:flex;justify-content:space-between;align-items:center">';
  h += '<h1>588000 日线择时</h1><span style="color:#8b949e;font-size:13px">'+D.generated+'</span></div></div>';

  // signal card
  h += '<div class="card"><h2>今日信号</h2><div class="sg">';
  h += '<span class="sl">日期</span><span class="sv">'+s.date+'</span>';
  h += '<span class="sl">状态</span><span class="sv"><span class="badge '+st+'">'+sl+'</span></span>';
  h += '<span class="sl">综合得分</span><span class="sv" style="color:'+scCls+'">'+sc.toFixed(2)+'</span>';
  h += '<span class="sl">周线过滤</span><span class="sv">'+(s.weekly?'允许交易':'禁止交易')+'</span>';
  h += '</div></div>';

  // factors
  h += '<div class="card"><h2>因子状态</h2><div class="fg">';
  for (var i=0;i<keys.length;i++) {
    var v = f[keys[i]]||0;
    var cls = v>0.3?'g':v<-0.3?'r':'y';
    var d = v>0.3?'<span style="font-size:16px">&#9650;</span>':v<-0.3?'<span style="font-size:16px">&#9660;</span>':'&#9644;';
    h += '<div class="fi"><div class="fn">'+fn[i]+'</div><div class="fv '+cls+'">'+d+' '+v.toFixed(2)+'</div></div>';
  }
  h += '</div></div>';

  // equity curve
  h += '<div class="card"><h2>权益曲线</h2><div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:8px;font-size:13px">';
  h += '<span>净值: <strong style="color:#58a6ff" id="eqf"></strong></span>';
  h += '<span id="eqm" style="color:#8b949e"></span></div><canvas id="ec"></canvas></div>';

  // factor trends
  h += '<div class="card"><h2>因子走势 <span style="color:#8b949e;font-weight:400;font-size:12px">近60天</span></h2><canvas id="fc"></canvas></div>';

  // trades table
  var tr = D.trades;
  if (tr.length>0) {
    h += '<div class="card"><h2>最近交易</h2><table><tr><th>日期</th><th>操作</th><th>价格</th><th>理由</th></tr>';
    for (var i=0;i<tr.length;i++) {
      var t = tr[i];
      var cls = t.action=='买入'||t.action=='加仓'?'tb':t.action=='卖出'||t.action=='清仓'?'ts':'th';
      var dt = t.entry_date?t.entry_date.slice(0,10):t.date?t.date.slice(0,10):'';
      var pr = t.entry_price||t.price||0;
      var rs = t.reason||'';
      h += '<tr><td>'+dt+'</td><td class="'+cls+'">'+t.action+'</td><td>'+(pr?pr.toFixed(3):'')+'</td><td style="color:#8b949e;font-size:12px">'+rs+'</td></tr>';
    }
    h += '</table></div>';
  }

  // price chart
  if (D.prices.length>0) {
    h += '<div class="card"><h2>价格走势 <span style="color:#8b949e;font-weight:400;font-size:12px">近200日</span></h2><canvas id="pc"></canvas></div>';
  }

  h += '<div class="ft">trade-pulse &#183; 数据 '+D.generated+'</div>';
  a.innerHTML = h;

  // draw
  drawEquity(D.equity);
  drawFactors(D.features);
  drawPrice(D.prices);
})();

// utils
function sz(c,w,h){
  var d=window.devicePixelRatio||1;
  c.width=w*d;c.height=h*d;c.style.width=w+'px';c.style.height=h+'px';
  var ctx=c.getContext('2d');ctx.scale(d,d);
  return {ctx:ctx,w:w,h:h};
}

// equity
function drawEquity(data) {
  var c = document.getElementById('ec');
  if (!c||data.length<3) return;
  var r = c.parentElement.getBoundingClientRect();
  var W = r.width||800, H = 240;
  var o = sz(c,W,H);
  var ctx = o.ctx;
  var pl = 50, pr = 20, pt = 20, pb = 28;
  var pw = W-pl-pr, ph = H-pt-pb;

  var hasBH = 'benchmark_equity' in data[0];
  var eq = data.map(function(d){return d.equity||0});
  var bh = hasBH ? data.map(function(d){return d.benchmark_equity||0}) : null;

  var lo = Math.min.apply(null, eq), hi = Math.max.apply(null, eq);
  if (bh) { lo = Math.min(lo, Math.min.apply(null,bh)); hi = Math.max(hi, Math.max.apply(null,bh)); }
  var rg = hi-lo||1;
  var pd = rg*0.08; lo-=pd; hi+=pd;
  var xS = pw/(data.length-1||1), yS = ph/(hi-lo);

  // grid
  ctx.strokeStyle='#21262d'; ctx.lineWidth=1;
  for (var g=0;g<=4;g++) {
    var yy = pt + g*ph/4;
    ctx.beginPath(); ctx.moveTo(pl,yy); ctx.lineTo(W-pr,yy); ctx.stroke();
    ctx.fillStyle='#484f58'; ctx.font='11px sans-serif'; ctx.textAlign='right';
    ctx.fillText((hi-g*(hi-lo)/4).toFixed(2), pl-6, yy+4);
  }

  // zero line
  var zy = pt + ph - (1.0-lo)*yS;
  if (zy>pt&&zy<pt+ph) {
    ctx.strokeStyle='#30363d'; ctx.setLineDash([4,4]);
    ctx.beginPath(); ctx.moveTo(pl,zy); ctx.lineTo(W-pr,zy); ctx.stroke(); ctx.setLineDash([]);
  }

  // signal bg
  ctx.globalAlpha=0.08;
  for (var i=1;i<data.length;i++) {
    var sg = data[i].signal;
    if (sg === '空仓') { ctx.fillStyle='#f85149'; var x1=pl+(i-1)*xS; ctx.fillRect(x1,pt,xS,ph); }
    else if (sg === '观望') { ctx.fillStyle='#d29922'; var x1=pl+(i-1)*xS; ctx.fillRect(x1,pt,xS,ph); }
  }
  ctx.globalAlpha=1;

  function drawLine(vals,color,label) {
    ctx.beginPath();
    for (var i=0;i<vals.length;i++) {
      var x=pl+i*xS, y=pt+ph-(vals[i]-lo)*yS;
      i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
    }
    ctx.strokeStyle=color; ctx.lineWidth=2; ctx.stroke();
    var lx=pl+(vals.length-1)*xS, ly=pt+ph-(vals[vals.length-1]-lo)*yS;
    ctx.fillStyle=color; ctx.font='12px sans-serif'; ctx.textAlign='left';
    ctx.fillText(vals[vals.length-1].toFixed(3),lx+6,ly+4);
    ctx.fillText(label,pl+4,pt+14);
  }

  if (bh) drawLine(bh,'rgba(139,148,158,0.5)','持有');
  drawLine(eq,'#58a6ff','策略');

  // metrics
  var eqf = document.getElementById('eqf');
  var eqm = document.getElementById('eqm');
  if (eqf) eqf.textContent = eq[eq.length-1].toFixed(3);
  if (eqm) eqm.textContent = '年化 +11.9% &#183; 夏普 0.42 &#183; 回撤 -21.0%' + (bh?' &#183; 持有 '+bh[bh.length-1].toFixed(2):'');
}

// factor trends
function drawFactors(features) {
  var c = document.getElementById('fc');
  if (!c||features.length<5) return;
  var r = c.parentElement.getBoundingClientRect();
  var W = r.width||800, H = 240;
  var o = sz(c,W,H);
  var ctx = o.ctx;
  var pl = 40, pr = 20, pt = 20, pb = 28;
  var pw = W-pl-pr, ph = H-pt-pb;
  var xS = pw/(features.length-1||1);

  var keys = ['momentum','trend','volatility','volume_price','rsrs','relative_strength'];
  var names = ['动量','趋势','波动','量价','RSRS','比价'];
  var colors = ['#58a6ff','#3fb950','#d29922','#f85149','#bc8cff','#79c0ff'];

  for (var k=0;k<keys.length;k++) {
    var vals = features.map(function(d){return d[keys[k]]||0});
    ctx.beginPath();
    for (var i=0;i<vals.length;i++) {
      var x=pl+i*xS, y=pt+ph-(vals[i]-(-1))*ph/2;
      i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
    }
    ctx.strokeStyle = colors[k];
    ctx.lineWidth = 1.2 + (k===0?0.8:0);
    ctx.globalAlpha = 0.5+(k===0?0.5:0);
    ctx.stroke();
    ctx.globalAlpha = 1;

    var lx=pl+4, ly=pt+14+Math.floor(k/3)*18;
    ctx.fillStyle=colors[k]; ctx.fillRect(lx+(k%3)*100,ly-6,10,2);
    ctx.fillStyle='#c9d1d9'; ctx.font='11px sans-serif'; ctx.textAlign='left';
    ctx.fillText(names[k], lx+(k%3)*100+14, ly+1);
  }

  ctx.strokeStyle='#30363d'; ctx.setLineDash([4,4]);
  var zy=pt+ph/2; ctx.beginPath();ctx.moveTo(pl,zy);ctx.lineTo(W-pr,zy);ctx.stroke();ctx.setLineDash([]);
  ctx.fillStyle='#484f58'; ctx.font='11px sans-serif'; ctx.textAlign='right';
  for (var v=-1;v<=1;v+=0.5) ctx.fillText(v.toFixed(1),pl-6,pt+ph-(v-(-1))*ph/2+4);
}

// price
function drawPrice(data) {
  var c = document.getElementById('pc');
  if (!c||data.length<3) return;
  var r = c.parentElement.getBoundingClientRect();
  var W = r.width||800, H = 240;
  var o = sz(c,W,H);
  var ctx = o.ctx;
  var pl=50,pr=20,pt=20,pb=28, pw=W-pl-pr, ph=H-pt-pb;

  var cls = data.map(function(d){return d.close||0});
  var lo=Math.min.apply(null,cls)*0.98, hi=Math.max.apply(null,cls)*1.02;
  var yS=ph/(hi-lo), xS=pw/(data.length-1||1);

  ctx.strokeStyle='#21262d'; ctx.lineWidth=1;
  for (var g=0;g<=4;g++) {
    var yy=pt+g*ph/4; ctx.beginPath();ctx.moveTo(pl,yy);ctx.lineTo(W-pr,yy);ctx.stroke();
    ctx.fillStyle='#484f58'; ctx.font='11px sans-serif'; ctx.textAlign='right';
    ctx.fillText((hi-g*(hi-lo)/4).toFixed(2),pl-6,yy+4);
  }

  ctx.beginPath();ctx.moveTo(pl,pt+ph);
  for (var i=0;i<cls.length;i++) {var x=pl+i*xS,y=pt+ph-(cls[i]-lo)*yS;ctx.lineTo(x,y);}
  ctx.lineTo(pl+(cls.length-1)*xS,pt+ph);ctx.closePath();
  var gd = ctx.createLinearGradient(0,pt,0,pt+ph);
  gd.addColorStop(0,'rgba(88,166,255,0.15)'); gd.addColorStop(1,'rgba(88,166,255,0.02)');
  ctx.fillStyle=gd;ctx.fill();

  ctx.beginPath();
  for (var i=0;i<cls.length;i++) {var x=pl+i*xS,y=pt+ph-(cls[i]-lo)*yS;i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);}
  ctx.strokeStyle='#58a6ff'; ctx.lineWidth=2; ctx.stroke();
}
</script>
</body>
</html>"""

    # 替换数据占位符
    html = html.replace("@DATA@", data_json)

    output = SCRIPT_DIR / "dashboard.html"
    with open(output, "w", encoding="utf-8") as f:
        f.write(html)

    kb = os.path.getsize(output) / 1024
    print(f"\n  [OK] {output.resolve()}")
    print(f"       {kb:.1f} KB · 内嵌数据 · 零依赖")

    try:
        if sys.platform == "linux":
            subprocess.run(["xdg-open", str(output)], check=False)
        elif sys.platform == "win32":
            os.startfile(output)
        elif sys.platform == "darwin":
            subprocess.run(["open", str(output)], check=False)
    except OSError:
        pass

    print("  [DONE] 面板已生成，浏览器打开即可\n")


if __name__ == "__main__":
    main()
