#!/usr/bin/env python3
"""
gen_dashboard.py — 生成信号面板 docs/dashboard.html

v2：数据改为外部加载（fetch assets/data.json?t=now），
    symbols 动态渲染（多标的自动支持），
    底部显示版本戳。

用法：
  python gen_dashboard.py              # 生成 docs/dashboard.html
  python gen_dashboard.py --serve      # 生成并启动本地 http server 预览（fetch 需要 http 协议）
"""

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import http.server
import socketserver

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
OUT_DIR = PROJECT_ROOT / "docs"
OUT_PATH = OUT_DIR / "dashboard.html"

# 静态模板：运行时用 JS 从 data.json 加载数据，动态渲染
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>trade-pulse 信号面板</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,Segoe UI,system-ui,sans-serif;background:#0d1117;color:#c9d1d9;padding:20px;max-width:1000px;margin:0 auto}
h1{font-size:20px;font-weight:600}
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:20px;margin-bottom:16px}
h2{font-size:14px;color:#8b949e;text-transform:uppercase;letter-spacing:.5px;margin-bottom:12px}
.sg{display:grid;grid-template-columns:auto 1fr;gap:8px 24px;font-size:15px}
.mg{display:grid;grid-template-columns:repeat(5,1fr);gap:10px}
.mi{background:#0d1117;border:1px solid #30363d;border-radius:6px;padding:10px 12px}
.mil{display:block;font-size:12px;color:#8b949e;margin-bottom:4px}
.miv{display:block;font-size:17px;font-weight:600}
.miv.up{color:#3fb950}.miv.dn{color:#f85149}.miv.md{color:#d29922}
@media(max-width:720px){.mg{grid-template-columns:repeat(2,1fr)}}
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
.err{color:#f85149;text-align:center;padding:40px 0;font-size:14px}
</style>
</head>
<body>
<div id="app"><div class="err">加载中...</div></div>
<script>
// HTML 转义（防 XSS：所有外部字段插值前先 esc）
function esc(s) {
  return String(s==null?'':s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}

// 本地预览（file:// 协议下 fetch 会被浏览器拦截）→ 读不到数据时提示用 http server
var isFile = location.protocol === 'file:';
var src = isFile
  ? '（file:// 下浏览器禁止 fetch，请用 python -m http.server 预览）'
  : 'assets/data.json?t=' + Date.now();

// cache-busting: 强制拿最新数据
fetch('assets/data.json?t=' + Date.now())
  .then(function(r){ if(!r.ok) throw new Error('HTTP '+r.status); return r.json(); })
  .then(render)
  .catch(function(e){ document.getElementById('app').innerHTML =
    '<div class="err">数据加载失败: '+esc(e.message)+'<br>'+(isFile?'（file:// 下浏览器禁止 fetch，请用 python -m http.server 预览）':'请确认 data.json 已生成')+'</div>'; });

function render(D) {
  var a = document.getElementById('app');
  var syms = D.symbols||['588000'];
  var h = '';

  // header
  h += '<div class="card"><div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px">';
  h += '<h1>trade-pulse 信号面板</h1><span style="color:#8b949e;font-size:13px">生成 '+esc(D.generated_at)+'</span></div>';
  h += '<div style="margin-top:8px;font-size:13px;color:#8b949e">';
  for (var i=0;i<syms.length;i++) h += '<span class="badge bg-cash" style="margin-right:6px">'+esc(syms[i])+'</span>';
  h += '</div></div>';

  // 每个标的一张信号卡
  for (var i=0;i<syms.length;i++) {
    var sym = syms[i];
    var s = (D.signals||{})[sym]||{};
    h += signalCard(sym, s);
  }

  // 回测绩效指标卡
  var mt = D.metrics||{};
  if (Object.keys(mt).length>0) {
    h += '<div class="card"><h2>回测绩效 <span style="color:#8b949e;font-weight:400;font-size:12px">全量回测</span></h2><div class="mg">';
    var mItems = [
      ['年化收益', mt.annual_return!=null ? (Number(mt.annual_return)>=0?'+':'')+Number(mt.annual_return).toFixed(2)+'%' : '—', Number(mt.annual_return)>=0?'up':'dn'],
      ['夏普比率', mt.sharpe!=null ? Number(mt.sharpe).toFixed(2) : '—', Number(mt.sharpe)>=0.5?'up':(Number(mt.sharpe)>=0?'md':'dn')],
      ['Sortino', mt.sortino!=null ? Number(mt.sortino).toFixed(2) : '—', Number(mt.sortino)>=0.5?'up':(Number(mt.sortino)>=0?'md':'dn')],
      ['Omega', mt.omega!=null ? Number(mt.omega).toFixed(2) : '—', Number(mt.omega)>=1.2?'up':(Number(mt.omega)>=1?'md':'dn')],
      ['最大回撤', mt.max_drawdown!=null ? Number(mt.max_drawdown).toFixed(1)+'%' : '—', 'dn'],
      ['卡玛比率', mt.calmar!=null ? Number(mt.calmar).toFixed(2) : '—', Number(mt.calmar)>=0.5?'up':(Number(mt.calmar)>=0?'md':'dn')],
      ['回撤持续(d)', mt.max_dd_duration!=null ? mt.max_dd_duration : '—', Number(mt.max_dd_duration)<=120?'up':(Number(mt.max_dd_duration)<=250?'md':'dn')],
      ['胜率', mt.win_rate!=null ? Number(mt.win_rate).toFixed(1)+'%' : '—', Number(mt.win_rate)>=45?'up':'md'],
      ['盈亏比', mt.profit_loss_ratio!=null ? Number(mt.profit_loss_ratio).toFixed(2) : '—', Number(mt.profit_loss_ratio)>=2?'up':'md'],
      ['交易次数', mt.trade_count!=null ? mt.trade_count : '—', '']
    ];
    for (var j=0;j<mItems.length;j++) {
      h += '<div class="mi"><span class="mil">'+mItems[j][0]+'</span><span class="miv '+mItems[j][2]+'">'+mItems[j][1]+'</span></div>';
    }
    h += '</div></div>';
  }

  // equity curve
  h += '<div class="card"><h2>权益曲线</h2><div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:8px;font-size:13px">';
  h += '<span>净值: <strong style="color:#58a6ff" id="eqf"></strong></span>';
  h += '<span id="eqm" style="color:#8b949e"></span></div><canvas id="ec"></canvas></div>';

  // factor trends
  h += '<div class="card"><h2>因子走势 <span style="color:#8b949e;font-weight:400;font-size:12px">近60天</span></h2><canvas id="fc"></canvas></div>';

  // trades
  var tr = D.trades||[];
  if (tr.length>0) {
    h += '<div class="card"><h2>最近交易</h2><table><tr><th>日期</th><th>操作</th><th>价格</th><th>理由</th></tr>';
    for (var i=0;i<tr.length;i++) {
      var t = tr[i];
      var cls = t.action=='买入'||t.action=='加仓'?'tb':t.action=='卖出'||t.action=='清仓'?'ts':'th';
      var dt = (t.entry_date||t.date||'').slice(0,10);
      var pr = t.entry_price||t.price||0;
      var rs = t.reason||'';
      h += '<tr><td>'+esc(dt)+'</td><td class="'+cls+'">'+esc(t.action)+'</td><td>'+(pr?Number(pr).toFixed(3):'')+'</td><td style="color:#8b949e;font-size:12px">'+esc(rs)+'</td></tr>';
    }
    h += '</table></div>';
  }

  // price
  if (D.prices&&D.prices.length>0) {
    h += '<div class="card"><h2>价格走势 <span style="color:#8b949e;font-weight:400;font-size:12px">近200日</span></h2><canvas id="pc"></canvas></div>';
  }

  h += '<div class="ft">trade-pulse &#183; 数据生成 '+esc(D.generated_at)+'</div>';
  a.innerHTML = h;

  drawEquity(D.equity||[]);
  drawFactors(D.features||[]);
  drawPrice(D.prices||[]);
}

function signalCard(sym, s) {
  var st = {持仓:'bg-long',观望:'bg-wait',空仓:'bg-cash'}[s.state]||'bg-cash';
  var sl = s.state==='持仓'?'持有中':s.state==='观望'?'观望':'空仓';
  var sc = s.score||0;
  var scCls = sc>0?'g':'r';
  var f = s.factors||{};
  var fn = s.factor_names||['动量','趋势','量价','RSRS'];
  var keys = ['momentum','trend','volume_price','rsrs'];

  var h = '<div class="card"><h2>'+esc(sym)+' 今日信号</h2><div class="sg">';
  h += '<span class="sl">日期</span><span class="sv">'+esc(s.date||'-')+'</span>';
  h += '<span class="sl">状态</span><span class="sv"><span class="badge '+st+'">'+sl+'</span></span>';
  h += '<span class="sl">综合得分</span><span class="sv" style="color:'+scCls+'">'+Number(sc).toFixed(2)+'</span>';
  h += '</div><div class="fg" style="margin-top:12px">';
  for (var i=0;i<keys.length;i++) {
    var v = f[keys[i]]||0;
    var cls = v>0.3?'g':v<-0.3?'r':'y';
    var d = v>0.3?'&#9650;':v<-0.3?'&#9660;':'&#9644;';
    h += '<div class="fi"><div class="fn">'+esc(fn[i])+'</div><div class="fv '+cls+'">'+d+' '+Number(v).toFixed(2)+'</div></div>';
  }
  h += '</div></div>';
  return h;
}

// ── 绘图工具（复用原 Canvas 逻辑）──
function sz(c,w,h){
  var d=window.devicePixelRatio||1;
  c.width=w*d;c.height=h*d;c.style.width=w+'px';c.style.height=h+'px';
  var ctx=c.getContext('2d');ctx.scale(d,d);
  return {ctx:ctx,w:w,h:h};
}
function drawLine(ctx,vals,pl,pt,pw,ph,xS,lo,hi,color,label){
  ctx.beginPath();
  for (var i=0;i<vals.length;i++){
    var x=pl+i*xS, y=pt+ph-(vals[i]-lo)/(hi-lo||1)*ph;
    i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
  }
  ctx.strokeStyle=color; ctx.lineWidth=2; ctx.stroke();
  ctx.fillStyle=color; ctx.font='12px sans-serif'; ctx.textAlign='left';
  ctx.fillText(label||'',pl+4,pt+14);
}
function drawEquity(data) {
  var c=document.getElementById('ec');
  if(!c||data.length<3) return;
  var r=c.parentElement.getBoundingClientRect();
  var W=r.width||800,H=240,o=sz(c,W,H),ctx=o.ctx;
  var pl=40,pr=20,pt=20,pb=28,pw=W-pl-pr,ph=H-pt-pb;
  var vals=data.map(function(x){return x.equity;});
  var bh=data.map(function(x){return x.benchmark_equity;}).filter(function(v){return v!=null;});
  var lo=Math.min.apply(null,vals),hi=Math.max.apply(null,vals);
  if(bh.length){lo=Math.min(lo,Math.min.apply(null,bh));hi=Math.max(hi,Math.max.apply(null,bh));}
  var pad=(hi-lo)*0.05||0.1; lo-=pad; hi+=pad;
  var xS=pw/(vals.length-1||1);
  if(bh.length) drawLine(ctx,bh,pl,pt,pw,ph,xS,lo,hi,'rgba(139,148,158,0.5)','持有');
  drawLine(ctx,vals,pl,pt,pw,ph,xS,lo,hi,'#58a6ff','策略');
  var eqf=document.getElementById('eqf');
  var eqm=document.getElementById('eqm');
  if(eqf) eqf.textContent=vals[vals.length-1].toFixed(3);
  if(eqm&&vals.length>1){
    var n=vals.length,totalRet=vals[n-1]/vals[0]-1,years=n/252;
    var annRet=years>0?Math.pow(1+totalRet,1/years)-1:0;
    var peak=vals[0],maxDD=0;
    for(var i=1;i<n;i++){if(vals[i]>peak)peak=vals[i];var dd=vals[i]/peak-1;if(dd<maxDD)maxDD=dd;}
    var rets=[];for(var i=1;i<n;i++)rets.push(vals[i]/vals[i-1]-1);
    var mean=0;for(var i=0;i<rets.length;i++)mean+=rets[i];mean/=rets.length;
    var sd=0;for(var i=0;i<rets.length;i++)sd+=(rets[i]-mean)*(rets[i]-mean);
    sd=Math.sqrt(sd/(rets.length-1))*Math.sqrt(252);
    var sharpe=sd>0?(annRet-0.02)/sd:0;
    eqm.textContent='年化 '+(annRet*100>=0?'+':'')+(annRet*100).toFixed(1)+'% &#183; 夏普 '+sharpe.toFixed(2)+' &#183; 回撤 '+(maxDD*100).toFixed(1)+'%'+(bh.length?' &#183; 持有 '+bh[bh.length-1].toFixed(2):'');
  }
}
function drawFactors(features){
  var c=document.getElementById('fc');
  if(!c||features.length<5) return;
  var r=c.parentElement.getBoundingClientRect();
  var W=r.width||800,H=240,o=sz(c,W,H),ctx=o.ctx;
  var pl=40,pr=20,pt=20,pb=28,pw=W-pl-pr,ph=H-pt-pb;
  var xS=pw/(features.length-1||1);
  var keys=['momentum','trend','volume_price','rsrs'];
  var names=['动量','趋势','量价','RSRS'];
  var colors=['#58a6ff','#3fb950','#d29922','#f85149'];
  for(var k=0;k<keys.length;k++){
    var vals=features.map(function(x){return x[keys[k]]||0;});
    var lo=-1,hi=1;
    ctx.strokeStyle=colors[k];ctx.lineWidth=1.5;ctx.beginPath();
    for(var i=0;i<vals.length;i++){
      var x=pl+i*xS,y=pt+ph-(vals[i]-lo)/(hi-lo)*ph;
      i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
    }
    ctx.stroke();
    ctx.fillStyle=colors[k];ctx.font='12px sans-serif';ctx.textAlign='left';
    ctx.fillText(names[k],pl+4,pt+14+k*16);
  }
  // zero line
  ctx.strokeStyle='rgba(139,148,158,0.3)';ctx.lineWidth=1;ctx.beginPath();
  var y0=pt+ph-(0-(-1))/2*ph;ctx.moveTo(pl,y0);ctx.lineTo(pl+pw,y0);ctx.stroke();
}
function drawPrice(prices){
  var c=document.getElementById('pc');
  if(!c||prices.length<5) return;
  var r=c.parentElement.getBoundingClientRect();
  var W=r.width||800,H=240,o=sz(c,W,H),ctx=o.ctx;
  var pl=40,pr=20,pt=20,pb=28,pw=W-pl-pr,ph=H-pt-pb;
  var vals=prices.map(function(x){return x.close;});
  var lo=Math.min.apply(null,vals),hi=Math.max.apply(null,vals);
  var pad=(hi-lo)*0.05||0.1;lo-=pad;hi+=pad;
  var xS=pw/(vals.length-1||1);
  drawLine(ctx,vals,pl,pt,pw,ph,xS,lo,hi,'#58a6ff','收盘');
  ctx.fillStyle='#8b949e';ctx.font='12px sans-serif';ctx.textAlign='left';
  ctx.fillText(vals[vals.length-1].toFixed(3),pl+(vals.length-1)*xS+6,pt+ph-(vals[vals.length-1]-lo)/(hi-lo)*ph+4);
}
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description='生成信号面板 docs/dashboard.html')
    parser.add_argument('--serve', action='store_true', help='生成后启动本地 http server 预览（file:// 下 fetch 会被浏览器拦截）')
    args = parser.parse_args()

    print("  [UI] 生成面板 docs/dashboard.html")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(HTML_TEMPLATE)

    kb = os.path.getsize(OUT_PATH) / 1024
    print(f"  [OK] {OUT_PATH.resolve()} ({kb:.0f}KB, 动态加载 data.json)")

    if args.serve:
        # file:// 下 fetch 被浏览器拦截，必须走 http server
        import webbrowser
        os.chdir(OUT_DIR)
        port = 8899
        # 端口被占用时自动换端口
        for p in range(port, port + 20):
            try:
                httpd = socketserver.TCPServer(("127.0.0.1", p), http.server.SimpleHTTPRequestHandler)
                port = p
                break
            except OSError:
                continue
        else:
            print("  [ERR] 8899-8918 端口均被占用，无法启动预览 server")
            sys.exit(1)
        url = f"http://127.0.0.1:{port}/index.html"
        print(f"  [SERVE] 本地预览: {url}  (Ctrl+C 停止)")
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
