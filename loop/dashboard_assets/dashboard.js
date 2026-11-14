/* loop dashboard client: experiment tree + score-over-time chart + detail panel, drawn from the
   JSON blob in #loop-data. Every stored string is LLM/log-derived: it only ever reaches the DOM via
   textContent / setAttribute (never parsed as markup). No network. */
(function () {
  'use strict';
  var blob = document.getElementById('loop-data');
  var D = JSON.parse(blob ? blob.textContent : '{"nodes":[]}');
  var TREE = document.getElementById('tree'), CHART = document.getElementById('chart');
  var PANEL = document.getElementById('panel'), TIP = document.getElementById('tip');
  var NS = TREE ? TREE.namespaceURI : null;
  var ST = {};
  try { ST = JSON.parse(sessionStorage.getItem('loopdash2') || '{}') || {}; } catch (e) { ST = {}; }

  // ---------------------------------------------------------------- helpers
  function h(tag, attrs, text) {
    var e = document.createElement(tag);
    for (var k in (attrs || {})) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = String(text);
    return e;
  }
  function s(tag, attrs, parent) {
    var e = document.createElementNS(NS, tag);
    for (var k in (attrs || {})) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function stext(parent, x, y, txt, cls, anchor) {
    var t = s('text', {x: x, y: y}, parent);
    if (cls) t.setAttribute('class', cls);
    if (anchor) t.setAttribute('text-anchor', anchor);
    t.textContent = txt;
    return t;
  }
  function f(x, d, sign) {
    if (x == null || !isFinite(x)) return '-';
    var v = Number(x).toFixed(d == null ? 4 : d);
    return sign !== false && x >= 0 ? '+' + v : v;
  }
  function ts(t) {
    if (!t) return '-';
    var d = new Date(t * 1000), p = function (n) { return (n < 10 ? '0' : '') + n; };
    return p(d.getMonth() + 1) + '-' + p(d.getDate()) + ' ' + p(d.getHours()) + ':' + p(d.getMinutes());
  }
  function trunc(x, n) { x = String(x || ''); return x.length > n ? x.slice(0, n - 1) + '…' : x; }
  function color(e, st) { e.style.setProperty('--c', 'var(--s-' + st + ')'); return e; }
  function save() {
    try {
      ST.y = window.scrollY;
      ST.open = [].map.call(document.querySelectorAll('details[open][id]'), function (d) { return d.id; });
      sessionStorage.setItem('loopdash2', JSON.stringify(ST));
    } catch (e) { /* storage may be blocked */ }
  }

  // ---------------------------------------------------------------- model
  // tree parent (tp): design point -> its batch; batch -> the parent most of its points share; a direct
  // child of root (or a batch hanging off root) -> its family branch node, which hangs off root.
  var byId = {}, kids = {}, tp = {}, ROOT = D.root || 'root';
  var rootNode = {id: ROOT, st: 'baseline', tag: 'baseline', fam: '', arm: 'baseline', U: 0, sd: null, P: null,
                  isRoot: true, ts: 0};
  byId[ROOT] = rootNode;
  D.nodes.forEach(function (n, i) { n.idx = i + 1; byId[n.id] = n; });
  var batches = {}, groups = {};
  (D.groups || []).forEach(function (g, i) {
    groups[g.id] = byId[g.id] = {id: g.id, isGroup: true, st: 'batch', tag: g.name, fam: g.name, arm: '',
                                 n: g.n, U: g.U, P: g.P, best: g.best, hot: g.hot, open: g.open, ord: i, sub: []};
    tp[g.id] = ROOT;
  });
  D.nodes.forEach(function (n) {
    if (n.batch == null) return;
    var bid = 'batch:d' + n.batch, meta = (D.batches || {})[bid] || {};
    var b = batches[bid] || (batches[bid] = byId[bid] = {id: bid, isBatch: true, st: 'batch', tag: 'design d' + n.batch,
                                                          fam: 'design', arm: 'design', pts: [], ts: n.ts,
                                                          p: byId[meta.p] ? meta.p : ROOT, grp: meta.grp});
    b.pts.push(n);
    b.ts = Math.min(b.ts, n.ts);
    tp[n.id] = bid;
  });
  function hangs(id, p, grp) { return p === ROOT && grp && groups['grp:' + grp] ? 'grp:' + grp : p; }
  Object.keys(batches).forEach(function (bid) {
    var b = batches[bid], m = b.pts.filter(function (x) { return x.U != null; });
    b.U = m.length ? Math.max.apply(null, m.map(function (x) { return x.U; })) : null;
    tp[bid] = hangs(bid, b.p, b.grp);
  });
  D.nodes.forEach(function (n) { if (n.batch == null) tp[n.id] = hangs(n.id, byId[n.p] ? n.p : ROOT, n.grp); });
  Object.keys(tp).forEach(function (c) { (kids[tp[c]] = kids[tp[c]] || []).push(c); });
  Object.keys(kids).forEach(function (k) {
    kids[k].sort(function (a, b) {
      var A = byId[a], B = byId[b];
      return A.isGroup && B.isGroup ? A.ord - B.ord : (A.ts || 0) - (B.ts || 0);
    });
  });
  Object.keys(groups).forEach(function (gid) {         // every real node under a branch (for filters)
    (function walk(id) {
      if (byId[id] && !byId[id].isGroup && !byId[id].isBatch) groups[gid].sub.push(byId[id]);
      (kids[id] || []).forEach(walk);
    })(gid);
  });
  // collapse state: the viewer's explicit toggles win; otherwise batches start folded and a branch
  // starts open only when small or holding an active / promising node.
  var TOG = ST.tog || {};
  function isCollapsed(id) {
    if (Object.prototype.hasOwnProperty.call(TOG, id)) return !!TOG[id];
    var n = byId[id];
    return n ? (n.isBatch ? true : n.isGroup ? !n.open : false) : false;
  }
  function toggle(id) { TOG[id] = isCollapsed(id) ? 0 : 1; ST.tog = TOG; }
  function reveal(id) {                                 // unfold every folded ancestor of id
    var c = tp[id], guard = 0;
    while (c && c !== ROOT && guard++ < 1000) { if (isCollapsed(c)) { TOG[c] = 0; ST.tog = TOG; } c = tp[c]; }
  }
  var incPath = {};
  (function () {
    var c = D.incumbent, guard = 0;
    while (c && c !== ROOT && byId[c] && guard++ < 1000) { incPath[c] = 1; c = tp[c]; }
  })();

  // ---------------------------------------------------------------- filters
  var FK = ['status', 'family', 'brief', 'arm', 'q'];
  function fv(k) { var e = document.getElementById('f-' + k); return e ? e.value : ''; }
  function match(n) {
    if (n.isRoot) return true;
    if (n.isBatch) return n.pts.some(match);
    if (n.isGroup) return n.sub.some(match);
    var v = {status: fv('status'), family: fv('family'), brief: fv('brief'), arm: fv('arm')};
    if (v.status && n.st !== v.status) return false;
    if (v.family && n.fam !== v.family) return false;
    if (v.brief && n.brief !== v.brief) return false;
    if (v.arm && n.arm !== v.arm) return false;
    var q = fv('q').toLowerCase();
    if (q) {
      var hay = [n.id, n.tag, n.fam, n.claim, n.fals, n.rej, n.st, n.comp, n.lev].join(' ').toLowerCase();
      if (hay.indexOf(q) < 0) return false;
    }
    return true;
  }

  // ---------------------------------------------------------------- tree
  var W = 160, H = 36, DX = 236, DY = 42, T = ST.t || null, POS = {};
  function layout() {
    POS = {};
    var leaf = 0;
    (function walk(id, depth) {
      var ch = isCollapsed(id) ? [] : (kids[id] || []);
      ch.forEach(function (c) { walk(c, depth + 1); });
      var y = ch.length ? POS[ch[0]].y : (leaf++) * DY;   // parent level with its first child (evo-style)
      POS[id] = {x: depth * DX, y: y};
    })(ROOT, 0);
  }
  function edgePath(a, b) {
    var x1 = a.x + W, y1 = a.y + H / 2, x2 = b.x, y2 = b.y + H / 2, m = (x1 + x2) / 2;
    return 'M' + x1 + ',' + y1 + 'C' + m + ',' + y1 + ' ' + m + ',' + y2 + ' ' + x2 + ',' + y2;
  }
  function applyT(vp) { vp.setAttribute('transform', 'translate(' + T.x + ',' + T.y + ') scale(' + T.k + ')'); }
  function fit() {
    var xs = Object.keys(POS).map(function (k) { return POS[k]; });
    var bw = Math.max.apply(null, xs.map(function (p) { return p.x; })) + W + 40;
    var bh = Math.max.apply(null, xs.map(function (p) { return p.y; })) + H + 40;
    var r = TREE.getBoundingClientRect();
    var k = Math.min(1, r.width / bw, r.height / bh);
    if (k < 0.75) k = Math.max(0.75, Math.min(1, r.width / bw));   // too big to fit: stay readable, pan instead
    T = {k: k, x: 16, y: 16};
  }
  function visible(id) {                                 // id, or its nearest ancestor on screen
    var guard = 0;
    while (id && !POS[id] && guard++ < 1000) id = tp[id];
    return id && POS[id] ? id : null;
  }
  function drawTree() {
    if (!TREE || !TREE.getBoundingClientRect().width) return;
    while (TREE.firstChild) TREE.removeChild(TREE.firstChild);
    layout();
    if (!T) fit();
    var vp = s('g', {'class': 'vp'}, TREE);
    applyT(vp);
    var eg = s('g', {}, vp), mg = s('g', {}, vp), ng = s('g', {}, vp);
    Object.keys(POS).forEach(function (id) {
      var n = byId[id];
      if (n.isRoot) return;
      var par = tp[id], merge = n.mp && n.mp.length;
      if (POS[par]) {
        var e = s('path', {d: edgePath(POS[par], POS[id]), 'class': 'edge' + (incPath[id] ? ' inc' : '')
                           + (merge && byId[par].isGroup ? ' faint' : '')}, eg);
        if (!match(n)) e.classList.add('dim');
      }
      if (!merge) return;
      // merge: each parent's edge (from the parent itself, or the folded branch that holds it) converges
      // on one junction just left of the node, then a single arrow enters it
      var p = POS[id], jx = p.x - 42, jy = p.y + H / 2, seen = {};
      n.mp.forEach(function (m) {
        var v = visible(m);
        if (!v || v === id || seen[v]) return;
        seen[v] = 1;
        var a = POS[v], y1 = a.y + H / 2, d;
        if (a.x + W < jx) {                             // a parent to the left: out of its right side
          var x1 = a.x + W, mx = (x1 + jx) / 2;
          d = 'M' + x1 + ',' + y1 + 'C' + mx + ',' + y1 + ' ' + mx + ',' + jy + ' ' + jx + ',' + jy;
        } else {                                        // same column or right of it: loop out of its left side
          d = 'M' + a.x + ',' + y1 + 'C' + (Math.min(a.x, jx) - 36) + ',' + y1 + ' ' + (jx - 36) + ',' + jy + ' ' + jx + ',' + jy;
        }
        s('path', {d: d, 'class': 'edge mrg' + (v !== m ? ' via' : '')}, mg);
      });
      s('path', {d: 'M' + jx + ',' + jy + 'L' + (p.x - 1) + ',' + jy, 'class': 'edge mrg in'}, mg);
      s('circle', {cx: jx, cy: jy, r: 3.5, 'class': 'mj'}, mg);
      stext(mg, jx + 5, jy - 4, 'merge', 'ml');
    });
    Object.keys(POS).forEach(function (id) { drawNode(ng, byId[id], POS[id]); });
  }
  function drawNode(parent, n, p) {
    var g = s('g', {'class': 'nd', transform: 'translate(' + p.x + ',' + p.y + ')', 'data-id': n.id}, parent);
    color(g, n.st);
    if (n.st === 'queued') g.classList.add('q');
    if (n.id === D.incumbent) g.classList.add('inc');
    if (n.id === SEL) g.classList.add('sel');
    if (n.arm === 'moonshot') g.classList.add('moon');
    if (n.isBatch) g.classList.add('batch');
    if (n.isGroup) { g.classList.add('grp'); if (n.hot) g.classList.add('hot'); }
    if (n.mp && n.mp.length) g.classList.add('mnode');
    if (!match(n)) g.classList.add('dim');
    s('rect', {'class': 'bd', width: W, height: H, rx: 6}, g);
    s('rect', {'class': 'sb', x: 3, y: 4, width: 4, height: H - 8, rx: 2}, g);
    var fold = isCollapsed(n.id) ? '▸ ' : '▾ ';
    var badge = n.isBatch ? fold + n.pts.length + ' pts' : n.isGroup ? fold + n.n + (n.n === 1 ? ' idea' : ' ideas')
      : n.comp ? 'compound' : n.arm === 'crossover' ? 'crossover' : n.arm === 'moonshot' ? 'moonshot'
      : n.arm === 'design' ? 'stacked' : n.id === D.incumbent ? 'incumbent' : '';
    stext(g, 12, 14, n.isGroup ? trunc(n.tag.replace('_', ' '), 14) : n.isBatch ? n.tag : n.id, 'nid');
    var u = n.U;
    if (u != null || n.isRoot) {
      stext(g, W - 6, 14, n.isRoot ? '0' : (n.isBatch ? 'max ' : '') + f(u), 'u' + (u > 0 ? ' up' : u < 0 ? ' dn' : ''), 'end');
    } else {
      stext(g, W - 6, 14, n.isBatch || n.isGroup ? '' : n.st.replace('_', ' '), 't2', 'end');
    }
    var sub = n.isBatch ? (n.p === ROOT ? 'from root' : 'from ' + n.p)
      : n.isGroup ? (n.P != null ? 'best P ' + f(n.P, 2, false) : 'unmeasured') : n.tag;
    stext(g, 12, 27, trunc(sub, badge ? 14 : 24), 't2');
    if (badge) stext(g, W - 6, 27, badge, 't2 bdg', 'end');
    if (n.P != null) {
      s('rect', {'class': 'pt', x: 8, y: H - 5, width: W - 16, height: 2.5, rx: 1}, g);
      s('rect', {'class': 'pb', x: 8, y: H - 5, width: Math.max(0.5, (W - 16) * n.P), height: 2.5, rx: 1}, g);
    }
    g.addEventListener('mouseenter', function (ev) { tip(ev, n); });
    g.addEventListener('mousemove', moveTip);
    g.addEventListener('mouseleave', hideTip);
    g.addEventListener('click', function () {
      if (moved) return;
      if (n.isBatch || n.isGroup) toggle(n.id);
      select(n.id);
    });
  }
  var drag = null, moved = false;
  if (TREE) {
    TREE.addEventListener('wheel', function (ev) {
      ev.preventDefault();
      var r = TREE.getBoundingClientRect(), mx = ev.clientX - r.left, my = ev.clientY - r.top;
      var k2 = Math.max(0.15, Math.min(3, T.k * Math.exp(-ev.deltaY * 0.0015)));
      T.x = mx - (mx - T.x) * k2 / T.k; T.y = my - (my - T.y) * k2 / T.k; T.k = k2;
      applyT(TREE.querySelector('.vp')); ST.t = T;
    }, {passive: false});
    TREE.addEventListener('pointerdown', function (ev) {
      drag = {x: ev.clientX, y: ev.clientY, tx: T.x, ty: T.y}; moved = false;
    });
    window.addEventListener('pointermove', function (ev) {
      if (!drag) return;
      var dx = ev.clientX - drag.x, dy = ev.clientY - drag.y;
      if (!moved && Math.abs(dx) + Math.abs(dy) < 4) return;
      moved = true; TREE.classList.add('drag');
      T.x = drag.tx + dx; T.y = drag.ty + dy; applyT(TREE.querySelector('.vp')); ST.t = T;
    });
    window.addEventListener('pointerup', function () {
      drag = null; TREE.classList.remove('drag');
      setTimeout(function () { moved = false; }, 0);
    });
    TREE.addEventListener('dblclick', function () { T = null; ST.t = null; drawTree(); });
  }

  // ---------------------------------------------------------------- tooltip
  function tip(ev, n) {
    if (!TIP) return;
    var lines = n.isBatch
      ? [n.tag + ' (' + n.pts.length + ' points)', 'best U ' + f(n.U)]
      : n.isGroup ? [n.tag + ' branch: ' + n.n + ' ideas', 'best U ' + f(n.U) + (n.best ? ' (' + n.best + ')' : ''),
                     'P(U>0) ' + (n.P != null ? f(n.P, 2, false) : '-'), n.hot ? 'holds an active / promising node' : '']
      : [n.id + '  ' + n.st, (n.fam || '') + '/' + (n.tag || ''),
         'U ' + (n.isRoot ? '0 (baseline)' : f(n.U) + (n.sd != null ? ' ± ' + f(n.sd, 4, false) : '')),
         'P(U>0) ' + (n.P != null ? f(n.P, 2, false) : '-')];
    if (n.claim) lines.push(trunc(n.claim, 160));
    TIP.textContent = lines.join('\n');
    TIP.hidden = false;
    moveTip(ev);
  }
  function moveTip(ev) {
    if (!TIP) return;
    TIP.style.left = Math.min(ev.clientX + 14, window.innerWidth - 370) + 'px';
    TIP.style.top = (ev.clientY + 14) + 'px';
  }
  function hideTip() { if (TIP) TIP.hidden = true; }

  // ---------------------------------------------------------------- chart
  function drawChart() {
    if (!CHART || !CHART.getBoundingClientRect().width) return;
    while (CHART.firstChild) CHART.removeChild(CHART.firstChild);
    var r = CHART.getBoundingClientRect(), Wd = Math.max(300, r.width), Hd = Math.max(160, r.height);
    CHART.setAttribute('viewBox', '0 0 ' + Wd + ' ' + Hd);
    var L = 58, R = 14, TP = 10, B = 26, N = D.nodes.length;
    var sig = D.sigma || 0, lo = -sig, hi = sig;
    D.nodes.forEach(function (n) {
      if (n.U == null) return;
      var e = n.sd || 0;
      lo = Math.min(lo, n.U - e); hi = Math.max(hi, n.U + e);
    });
    if (hi - lo < 1e-9) { lo -= 0.01; hi += 0.01; }
    var pad = (hi - lo) * 0.08; lo -= pad; hi += pad;
    var X = function (i) { return L + (i / (N + 1)) * (Wd - L - R); };
    var Y = function (v) { return TP + (hi - v) / (hi - lo) * (Hd - TP - B); };
    var step = Math.pow(10, Math.floor(Math.log10((hi - lo) / 4)));
    if ((hi - lo) / step > 10) step *= 2;
    if ((hi - lo) / step > 10) step *= 2.5;
    for (var v = Math.ceil(lo / step) * step; v <= hi; v += step) {
      s('line', {x1: L, x2: Wd - R, y1: Y(v), y2: Y(v), 'class': 'grid'}, CHART);
      stext(CHART, L - 4, Y(v) + 3, f(Math.abs(v) < step / 1e6 ? 0 : v, step < 0.001 ? 4 : 3), 'lbl', 'end');
    }
    if (sig) s('rect', {x: L, width: Wd - L - R, y: Y(sig), height: Y(-sig) - Y(sig), 'class': 'band'}, CHART);
    s('line', {x1: L, x2: Wd - R, y1: Y(0), y2: Y(0), 'class': 'ax0'}, CHART);
    var ticks = Math.min(N, 10);
    for (var t = 1; t <= ticks; t++) {
      var i = Math.round(t * N / ticks);
      stext(CHART, X(i), Hd - 8, '#' + i, 'lbl', 'middle');
    }
    // best-path stair: cumulative best corrected U over confirmed nodes (root = 0)
    var best = 0, pts = [[X(0), Y(0)]];
    D.nodes.forEach(function (n) {
      if (n.st === 'confirmed' && n.U != null && n.U > best) {
        pts.push([X(n.idx), Y(best)]); best = n.U; pts.push([X(n.idx), Y(best)]);
      }
    });
    pts.push([X(N + 1), Y(best)]);
    s('polyline', {points: pts.map(function (p) { return p.join(','); }).join(' '), 'class': 'stair'}, CHART);
    D.nodes.forEach(function (n) {
      var x = X(n.idx), dim = !match(n), y = Y(n.U != null ? n.U : 0), g = s('g', {}, CHART);
      color(g, n.st);
      if (dim) g.setAttribute('class', 'dim');
      if (n.U != null && n.sd) s('line', {x1: x, x2: x, y1: Y(n.U - n.sd), y2: Y(n.U + n.sd), 'class': 'eb'}, g);
      var cls = 'dot' + (n.U == null ? ' q' : '') + (n.id === SEL ? ' sel' : '');
      var d = n.batch != null ? s('rect', {x: x - 3.5, y: y - 3.5, width: 7, height: 7, 'class': cls}, g)
        : s('circle', {cx: x, cy: y, r: n.U == null ? 3.5 : 4.5, 'class': cls}, g);
      d.setAttribute('data-id', n.id);
      d.addEventListener('mouseenter', function (ev) { tip(ev, n); });
      d.addEventListener('mousemove', moveTip);
      d.addEventListener('mouseleave', hideTip);
      d.addEventListener('click', function () { select(n.id, true); });
      if (n.id === D.incumbent && n.U != null) {
        var r2 = 8, sp = [];
        for (var k = 0; k < 10; k++) {
          var a = Math.PI / 5 * k - Math.PI / 2, rr = k % 2 ? r2 / 2.4 : r2;
          sp.push((x + rr * Math.cos(a)).toFixed(1) + ',' + (y - 13 + rr * Math.sin(a)).toFixed(1));
        }
        s('polygon', {points: sp.join(' '), 'class': 'star'}, g);
      }
    });
  }

  // ---------------------------------------------------------------- detail panel
  var SEL = ST.sel || null;
  function chip(text, cls, st) {
    var c = h('span', {'class': 'chip' + (cls ? ' ' + cls : '')}, text);
    if (st) color(c, st);
    return c;
  }
  function link(id) {
    var a = h('a', {href: '#node-' + id, 'class': 'nid'}, id);
    return a;
  }
  function row(parent, label, text, cls) {
    if (!text) return;
    var p = h('p', {'class': 'small' + (cls ? ' ' + cls : '')});
    p.appendChild(h('b', {}, label + ' '));
    p.appendChild(document.createTextNode(text));
    parent.appendChild(p);
  }
  function axes(n) {
    var box = h('div', {'class': 'axes'});
    (n.axes || []).forEach(function (a) {
      var t = a[0], m = a[1], sig = a[2], pred = a[3];
      var r = h('div', {'class': 'ax' + (n.veto.indexOf(t) >= 0 ? ' veto' : '')});
      r.appendChild(h('span', {'class': 'an', title: t}, t));
      var tr = h('span', {'class': 'track'});
      tr.appendChild(h('span', {'class': 'mid'}));
      if (m != null && sig > 0) {
        var w = Math.min(1, Math.abs(m) / (2 * sig)) * 50;
        var b = h('span', {'class': 'bar ' + (m >= 0 ? 'bp' : 'bn')});
        b.style.left = (m >= 0 ? 50 : 50 - w) + '%'; b.style.width = w + '%';
        tr.appendChild(b);
      }
      r.appendChild(tr);
      r.appendChild(h('span', {'class': 'av'}, m != null ? f(m) : '-'));
      r.appendChild(h('span', {'class': 'ap muted'}, pred));
      box.appendChild(r);
    });
    return box;
  }
  function diff(text) {
    var d = h('details', {id: 'diff-' + SEL});
    d.appendChild(h('summary', {}, 'diff (' + (text.match(/\n/g) || []).length + ' lines)'));
    var pre = h('pre', {'class': 'diff'});
    text.split('\n').forEach(function (line) {
      var c = /^(diff |index |\+\+\+|---)/.test(line) ? 'dm' : /^@@/.test(line) ? 'dh'
        : line[0] === '+' ? 'da' : line[0] === '-' ? 'dd' : '';
      if (c) pre.appendChild(h('span', {'class': c}, line));
      else pre.appendChild(document.createTextNode(line + '\n'));
    });
    d.appendChild(pre);
    return d;
  }
  function childList(id) {
    var ch = kids[id] || [];
    if (!ch.length) return null;
    var box = h('div', {'class': 'small'});
    box.appendChild(h('h4', {}, 'children (' + ch.length + ')'));
    ch.forEach(function (c) {
      var n = byId[c], line = h('div');
      line.appendChild(chip(n.isBatch || n.isGroup ? n.tag : n.st, 'st', n.st));
      line.appendChild(n.isBatch ? h('a', {href: '#node-' + c}, n.pts.length + ' points')
        : n.isGroup ? h('a', {href: '#node-' + c}, n.n + ' ideas') : link(c));
      line.appendChild(h('span', {'class': 'muted'}, '  ' + (n.U != null ? 'U ' + f(n.U) : '') + '  ' + trunc(n.tag, 40)));
      box.appendChild(line);
    });
    return box;
  }
  function card(n) {
    var a = h('article', {'class': 'card'});
    var head = h('div', {'class': 'ih'});
    head.appendChild(h('span', {'class': 'nid'}, n.id));
    head.appendChild(chip(n.st, 'st', n.st));
    if (n.id === D.incumbent) head.appendChild(chip('incumbent', 'hot'));
    a.appendChild(head);
    var meta = h('div', {'class': 'small muted'});
    meta.appendChild(document.createTextNode(ts(n.ts) + ' · ' + n.arm + '/' + n.op + ' · brief ' + n.brief
      + ' · ' + n.be + ' · parent '));
    meta.appendChild(link(n.p || ROOT));
    if (n.bank) meta.appendChild(document.createTextNode(' · bank ' + n.bank));
    if (n.mp && n.mp.length) {
      meta.appendChild(document.createTextNode(' · merge of '));
      n.mp.forEach(function (m, i) { if (i) meta.appendChild(document.createTextNode(' + ')); meta.appendChild(link(m)); });
      if (n.sfi) meta.appendChild(document.createTextNode(' (knob source inferred)'));
    } else if (n.p2) { meta.appendChild(document.createTextNode(' + ')); meta.appendChild(link(n.p2)); }
    a.appendChild(meta);
    a.appendChild(h('div', {'class': 'util'}, n.U != null
      ? 'U ' + f(n.U) + ' ± ' + f(n.sd, 4, false) + ' · P(U>0) ' + f(n.P, 2, false)
        + (n.Pg != null ? ' · P(gain>0) ' + f(n.Pg, 2, false) + ' · P(U>inc) ' + f(n.Pi, 2, false) : '') : 'unmeasured'));
    var tags = h('div', {'class': 'tags'});
    tags.appendChild(chip(n.fam, 'fam'));
    tags.appendChild(chip(n.tag, 'tag'));
    if (n.comp) tags.appendChild(chip('compound ' + n.comp, 'hot'));
    if (['moonshot', 'crossover', 'design'].indexOf(n.arm) >= 0) tags.appendChild(chip(n.arm, 'hot'));
    if (n.lev) tags.appendChild(chip('levers ' + n.lev, 'lev'));
    a.appendChild(tags);
    if (n.claim) a.appendChild(h('p', {'class': 'claim'}, n.claim));
    row(a, 'falsified if', n.fals);
    a.appendChild(axes(n));
    row(a, 'veto', n.veto.join(', '), 'neg');
    row(a, 'rejected', n.rej, 'neg');
    if (n.ho.length) {
      var hd = h('details', {id: 'ho-' + n.id});
      hd.appendChild(h('summary', {}, 'handoff note'));
      var dl = h('dl');
      n.ho.forEach(function (kv) { dl.appendChild(h('dt', {}, kv[0])); dl.appendChild(h('dd', {}, kv[1])); });
      hd.appendChild(dl);
      a.appendChild(hd);
    }
    n.pm.forEach(function (pm) {
      var d = h('div', {'class': 'pm'});
      d.appendChild(h('b', {}, 'postmortem [' + pm[0] + '] '));
      d.appendChild(document.createTextNode(pm[1]));
      [['mechanism', pm[2]], ['next', pm[3]]].forEach(function (kv) {
        if (!kv[1]) return;
        d.appendChild(h('br'));
        d.appendChild(h('span', {'class': 'muted'}, kv[0] + ': '));
        d.appendChild(document.createTextNode(kv[1]));
      });
      a.appendChild(d);
    });
    var tl = h('div', {'class': 'tl'});
    n.tr.forEach(function (t, i) {
      if (i) tl.appendChild(document.createTextNode(' → '));
      tl.appendChild(chip(t[0], 'st-' + t[0]));
      tl.appendChild(h('span', {'class': 'muted'}, ts(t[1]) + (t[2] ? ' · ' + t[2] : '')));
    });
    a.appendChild(tl);
    if (n.runs.length) {
      var rr = h('div', {'class': 'small muted'}, 'runs: ');
      n.runs.forEach(function (r) { rr.appendChild(chip(r[0], 'r-' + r[1])); });
      a.appendChild(rr);
    }
    var cl = childList(n.id);
    if (cl) a.appendChild(cl);
    if (n.diff) a.appendChild(diff(n.diff));
    return a;
  }
  function batchCard(b) {
    var a = h('article', {'class': 'card'});
    var head = h('div', {'class': 'ih'});
    head.appendChild(h('span', {'class': 'nid'}, b.tag));
    head.appendChild(chip(b.pts.length + ' design points', 'st', 'batch'));
    a.appendChild(head);
    var cnt = {};
    b.pts.forEach(function (p) { cnt[p.st] = (cnt[p.st] || 0) + 1; });
    var c = h('div', {'class': 'tags'});
    Object.keys(cnt).forEach(function (k) { c.appendChild(chip(k + ' ' + cnt[k], 'st', k)); });
    a.appendChild(c);
    a.appendChild(h('p', {'class': 'small muted'}, 'one shared seed per point; effects are in World model › Designed batches. Click the batch node to expand / collapse.'));
    var box = h('div', {'class': 'small'});
    b.pts.slice().sort(function (x, y) { return (y.U == null ? -1e9 : y.U) - (x.U == null ? -1e9 : x.U); }).forEach(function (p) {
      var line = h('div');
      line.appendChild(chip(p.st, 'st', p.st));
      line.appendChild(link(p.id));
      line.appendChild(h('span', {'class': 'muted'}, '  ' + (p.U != null ? 'U ' + f(p.U) : 'unmeasured') + '  ' + p.tag));
      box.appendChild(line);
    });
    a.appendChild(box);
    return a;
  }
  function groupCard(g) {
    var a = h('article', {'class': 'card'});
    var head = h('div', {'class': 'ih'});
    head.appendChild(h('span', {'class': 'nid'}, g.tag));
    head.appendChild(chip(g.n + ' ideas', 'st', 'batch'));
    if (g.hot) head.appendChild(chip('active / promising', 'hot'));
    a.appendChild(head);
    a.appendChild(h('div', {'class': 'util'}, g.U != null ? 'best U ' + f(g.U) + ' · P(U>0) ' + f(g.P, 2, false)
      + (g.best ? ' · ' + g.best : '') : 'nothing measured yet'));
    var cnt = {};
    g.sub.forEach(function (p) { cnt[p.st] = (cnt[p.st] || 0) + 1; });
    var c = h('div', {'class': 'tags'});
    Object.keys(cnt).forEach(function (k) { c.appendChild(chip(k + ' ' + cnt[k], 'st', k)); });
    a.appendChild(c);
    a.appendChild(h('p', {'class': 'small muted'}, 'direct children of root in this family (lineage stays under its parent). Click the branch node to expand / collapse.'));
    var box = h('div', {'class': 'small'});
    g.sub.slice().sort(function (x, y) { return (y.U == null ? -1e9 : y.U) - (x.U == null ? -1e9 : x.U); }).forEach(function (p) {
      var line = h('div');
      line.appendChild(chip(p.st, 'st', p.st));
      line.appendChild(link(p.id));
      line.appendChild(h('span', {'class': 'muted'}, '  ' + (p.U != null ? 'U ' + f(p.U) : 'unmeasured') + '  ' + trunc(p.tag, 50)));
      box.appendChild(line);
    });
    a.appendChild(box);
    return a;
  }
  function renderPanel() {
    if (!PANEL) return;
    while (PANEL.firstChild) PANEL.removeChild(PANEL.firstChild);
    var n = SEL && byId[SEL];
    if (!n) { PANEL.appendChild(h('p', {'class': 'muted'}, 'click a node in the tree or chart')); return; }
    if (n.isBatch) { PANEL.appendChild(batchCard(n)); return; }
    if (n.isGroup) { PANEL.appendChild(groupCard(n)); return; }
    if (n.isRoot) {
      var a = h('article', {'class': 'card'});
      a.appendChild(h('div', {'class': 'nid'}, n.id + ' (baseline)'));
      a.appendChild(h('p', {'class': 'small muted'}, 'the paired control for every seed; U is measured relative to it'));
      var cl = childList(n.id);
      if (cl) a.appendChild(cl);
      PANEL.appendChild(a);
      return;
    }
    PANEL.appendChild(card(n));
  }
  function select(id, pan) {
    SEL = id; ST.sel = id;
    var n = byId[id];
    if (n) reveal(id);
    [].forEach.call(document.querySelectorAll('.nd.sel,.dot.sel'), function (e) { e.classList.remove('sel'); });
    drawTree();
    if (pan && POS[id] && T && TREE) {                  // from the chart / a link: bring it into view
      var r = TREE.getBoundingClientRect(), x = T.x + POS[id].x * T.k, y = T.y + POS[id].y * T.k;
      if (x < 0 || y < 0 || x + W * T.k > r.width || y + H * T.k > r.height) {
        T.x = Math.min(16, r.width / 2 - (POS[id].x + W / 2) * T.k); T.y = r.height / 2 - (POS[id].y + H / 2) * T.k;
        ST.t = T; applyT(TREE.querySelector('.vp'));
      }
    }
    drawChart(); renderPanel();
  }

  // ---------------------------------------------------------------- legend, tabs, wiring
  function legend() {
    var box = document.getElementById('legend');
    if (!box) return;
    var seen = {};
    D.nodes.forEach(function (n) { seen[n.st] = 1; });
    (D.statuses || []).concat(['batch']).forEach(function (st) {
      if (st !== 'batch' && !seen[st]) return;
      if (st === 'batch' && !Object.keys(batches).length) return;
      var sp = h('span');
      var i = h('i', st === 'queued' ? {'class': 'q'} : {});
      color(i, st);
      sp.appendChild(i);
      sp.appendChild(document.createTextNode(st.replace('_', ' ')));
      box.appendChild(sp);
    });
    [['gold', 'incumbent / its path'], ['hot', 'merge: crossover parents / compound knob source (dashed)']].forEach(function (x) {
      var sp = h('span'), i = h('i');
      i.style.background = 'var(--' + x[0] + ')';
      sp.appendChild(i); sp.appendChild(document.createTextNode(x[1])); box.appendChild(sp);
    });
    box.appendChild(h('span', {}, 'bottom bar = P(U>0) · dashed border = moonshot · thick border = family branch'));
  }
  function tab(name) {
    ST.tab = name;
    [].forEach.call(document.querySelectorAll('section.tab'), function (sec) { sec.hidden = sec.id !== name; });
    [].forEach.call(document.querySelectorAll('nav a[data-tab]'), function (a) { a.classList.toggle('on', a.dataset.tab === name); });
    if (name === 'ideas') { drawTree(); drawChart(); }
  }
  function onHash() {
    var m = /^#node-(.+)$/.exec(location.hash);
    if (m) {
      var id = decodeURIComponent(m[1]);
      if (byId[id]) { tab('ideas'); select(id, true); }
      history.replaceState(null, '', location.pathname + location.search);
    }
  }
  function refilter() {
    ST.f = {};
    FK.forEach(function (k) { ST.f[k] = fv(k); });
    var n = D.nodes.filter(match).length;
    var fc = document.getElementById('f-count');
    if (fc) fc.textContent = n + ' of ' + D.nodes.length + ' match';
    drawTree(); drawChart();
  }
  [].forEach.call(document.querySelectorAll('nav a[data-tab]'), function (a) {
    a.addEventListener('click', function (ev) { ev.preventDefault(); tab(a.dataset.tab); });
  });
  FK.forEach(function (k) {
    var e = document.getElementById('f-' + k);
    if (!e) return;
    if (ST.f && ST.f[k]) e.value = ST.f[k];
    e.addEventListener('input', refilter);
  });
  window.addEventListener('hashchange', onHash);
  window.addEventListener('pagehide', save);
  var rt = null;
  window.addEventListener('resize', function () { clearTimeout(rt); rt = setTimeout(drawChart, 120); });
  legend();
  tab(ST.tab || 'ideas');
  refilter();
  renderPanel();
  onHash();
  (ST.open || []).forEach(function (i) { var d = document.getElementById(i); if (d) d.open = true; });
  if (ST.y) window.scrollTo(0, ST.y);
  setTimeout(function () { save(); location.reload(); }, 60000);
})();
