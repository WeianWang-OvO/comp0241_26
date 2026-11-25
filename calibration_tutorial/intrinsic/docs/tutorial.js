/* COMP0241 Week 3 calibration tutorial: page behaviour and the four interactive figures.
   No build step. Data for the residual and coverage plots comes from sample-data.js. */
(function () {
  'use strict';

  var RED = '#e30613', INK = '#000', MUTE = '#6b6b6b', FAINT = '#c9c9c9';
  var SVGNS = 'http://www.w3.org/2000/svg';
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };
  function fmt(n, d) { var s = Math.abs(n).toFixed(d); return (n < 0 && +s !== 0 ? '−' : '') + s; }
  function fmtS(n, d) { var s = Math.abs(n).toFixed(d); return (+s === 0 ? ' ' : n < 0 ? '−' : '+') + s; }
  function el(tag, attrs, parent) {
    var e = document.createElementNS(SVGNS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function txt(parent, x, y, s, cls, extra) {
    var t = el('text', Object.assign({ x: x, y: y }, extra || {}), parent);
    if (cls) t.setAttribute('class', cls);
    t.textContent = s;
    return t;
  }
  function store(key, val) {
    try {
      if (val === undefined) return JSON.parse(localStorage.getItem(key) || 'null');
      localStorage.setItem(key, JSON.stringify(val));
    } catch (e) { return null; }
  }
  function statList(dl, rows) {
    dl.innerHTML = '';
    var out = {};
    rows.forEach(function (r) {
      var d = document.createElement('div'); d.className = 'stat' + (r.hot ? ' hot' : '');
      var dt = document.createElement('dt'); dt.textContent = r.label;
      var dd = document.createElement('dd');
      d.appendChild(dt); d.appendChild(dd); dl.appendChild(d);
      out[r.id] = function (v, unit) { dd.innerHTML = ''; dd.appendChild(document.createTextNode(v)); if (unit) { var u = document.createElement('span'); u.className = 'u'; u.textContent = unit; dd.appendChild(u); } };
    });
    return out;
  }

  /* ---------------------------------------------------------------- code blocks */
  function setupCode() {
    $$('pre code').forEach(function (code) {
      var raw = code.textContent;
      code.parentNode.parentNode.dataset.raw = raw;
      if (window.hljs && !code.classList.contains('language-plaintext')) {
        try { window.hljs.highlightElement(code); } catch (e) { /* keep plain text */ }
      }
      var start = parseInt(code.dataset.start || '0', 10);
      if (!start) return;
      var hl = {};
      (code.dataset.hl || '').split(',').forEach(function (part) {
        if (!part) return;
        var ab = part.split('-').map(Number);
        for (var i = ab[0]; i <= (ab[1] || ab[0]); i++) hl[i] = true;
      });
      code.innerHTML = code.innerHTML.split('\n').map(function (line, i) {
        var n = start + i;
        var inner = '<span class="ln">' + n + '</span>' + line;
        return hl[n] ? '<span class="hl-line">' + inner + '</span>' : inner;
      }).join('\n');
    });
    $$('.code .copy').forEach(function (btn) {
      var file = btn.parentNode.querySelector('.file');
      btn.setAttribute('aria-label', 'Copy code: ' + (file ? file.textContent : 'snippet'));
      btn.addEventListener('click', function () {
        var text = btn.closest('.code').dataset.raw || '';
        var done = function () { btn.textContent = 'Copied'; setTimeout(function () { btn.textContent = 'Copy'; }, 1400); };
        if (navigator.clipboard && window.isSecureContext) {
          navigator.clipboard.writeText(text).then(done, function () { fallbackCopy(text); done(); });
        } else { fallbackCopy(text); done(); }
      });
    });
  }
  function fallbackCopy(text) {
    var ta = document.createElement('textarea');
    ta.value = text; ta.setAttribute('readonly', ''); ta.style.position = 'fixed'; ta.style.opacity = '0';
    document.body.appendChild(ta); ta.select();
    try { document.execCommand('copy'); } catch (e) { /* ignore */ }
    document.body.removeChild(ta);
  }

  /* ---------------------------------------------------------------- math */
  function setupMath() {
    if (!window.renderMathInElement) return;
    window.renderMathInElement(document.body, {
      delimiters: [{ left: '$$', right: '$$', display: true }, { left: '\\(', right: '\\)', display: false }],
      throwOnError: false
    });
  }

  /* ---------------------------------------------------------------- toc + progress */
  function setupNav() {
    // Narrow screens hide the sticky sidebar; give them the same contents as a collapsible list after the hero.
    var toc = $('.toc'), hero = $('.hero');
    if (toc && hero) {
      var d = document.createElement('details'); d.className = 'toc-inline';
      var parts = $$('section.part-s').length;
      var sm = document.createElement('summary');
      sm.textContent = 'Contents · ' + (parts ? parts + ' parts' : $$('section.s').length + ' sections'); d.appendChild(sm);
      var wrap = document.createElement('div'); wrap.className = 'toc-inline-body';
      // only the contents' own top-level children: the nested lists come along inside their parts
      Array.prototype.slice.call(toc.children).forEach(function (n) {
        if (n.matches('.part, ol')) wrap.appendChild(n.cloneNode(true));
      });
      d.appendChild(wrap);
      hero.parentNode.insertBefore(d, hero.nextSibling);
      $$('a', d).forEach(function (a) { a.addEventListener('click', function () { d.open = false; }); });
    }
    var links = $$('.toc a');
    var byId = {};
    links.forEach(function (a) { byId[a.getAttribute('href').slice(1)] = a; });
    var sections = $$('section.part-s, section.s');  // parts and their subsections, in page order
    var bar = $('#progress');
    function onScroll() {
      var h = document.documentElement;
      var f = h.scrollTop / Math.max(1, h.scrollHeight - h.clientHeight);
      if (bar) bar.style.width = (100 * Math.min(1, Math.max(0, f))) + '%';
      var cur = null;
      for (var i = 0; i < sections.length; i++) {
        if (sections[i].getBoundingClientRect().top < 140) cur = sections[i].id;
      }
      links.forEach(function (a) { a.classList.remove('on'); });
      if (cur && byId[cur]) byId[cur].classList.add('on');
      var part = cur && document.getElementById(cur).closest('section.part-s');  // and the part it belongs to
      if (part && byId[part.id]) byId[part.id].classList.add('on');
      // the contents can be taller than the window: keep the current entry in view
      var on = cur && byId[cur];
      if (on && toc && toc.scrollHeight > toc.clientHeight) {
        var tr = toc.getBoundingClientRect(), r = on.getBoundingClientRect();
        if (r.bottom > tr.bottom - 8) toc.scrollTop += r.bottom - tr.bottom + tr.height / 3;
        else if (r.top < tr.top + 8) toc.scrollTop -= tr.top - r.top + tr.height / 3;
      }
    }
    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();
  }

  /* ---------------------------------------------------------------- checklist */
  function setupChecklist() {
    var KEY = 'comp0241-w3-calib-checklist';
    var boxes = $$('#checklist input[type="checkbox"]');
    if (!boxes.length) return;
    var saved = store(KEY) || {};
    var count = $('#check-count');
    function update() {
      var n = boxes.filter(function (b) { return b.checked; }).length;
      count.textContent = n + ' / ' + boxes.length + ' done';
    }
    boxes.forEach(function (b) {
      b.checked = !!saved[b.dataset.k];
      b.addEventListener('change', function () { saved[b.dataset.k] = b.checked; store(KEY, saved); update(); });
    });
    $('#check-reset').addEventListener('click', function () { boxes.forEach(function (b) { b.checked = false; }); saved = {}; store(KEY, saved); update(); });
    update();
  }

  /* ---------------------------------------------------------------- 07: inner corners diagram */
  function drawBoard() {
    var svg = $('#board-svg'); if (!svg) return;
    var x0 = 70, y0 = 62, s = 40, C = 8, R = 5;
    for (var r = 0; r < R; r++) for (var c = 0; c < C; c++) {
      el('rect', { x: x0 + c * s, y: y0 + r * s, width: s, height: s, fill: (r + c) % 2 === 0 ? '#111' : '#fff' }, svg);
    }
    el('rect', { x: x0, y: y0, width: C * s, height: R * s, fill: 'none', stroke: INK, 'stroke-width': 1.5 }, svg);
    for (var j = 1; j < R; j++) for (var i = 1; i < C; i++) {
      el('circle', { cx: x0 + i * s, cy: y0 + j * s, r: 5.5, fill: RED, stroke: '#fff', 'stroke-width': 1.5 }, svg);
    }
    // counts across the top
    el('line', { x1: x0, y1: 44, x2: x0 + C * s, y2: 44, stroke: INK, 'stroke-width': 1 }, svg);
    for (var k = 0; k <= C; k++) el('line', { x1: x0 + k * s, y1: 40, x2: x0 + k * s, y2: 48, stroke: INK, 'stroke-width': 1 }, svg);
    txt(svg, x0, 30, '8 squares across', 'mono');
    txt(svg, x0 + C * s, 30, '7 inner corners', 'mono', { 'text-anchor': 'end', fill: RED, style: 'fill:' + RED });
    // counts down the side
    var xs = x0 + C * s + 18;
    el('line', { x1: xs, y1: y0, x2: xs, y2: y0 + R * s, stroke: INK, 'stroke-width': 1 }, svg);
    for (var m = 0; m <= R; m++) el('line', { x1: xs - 4, y1: y0 + m * s, x2: xs + 4, y2: y0 + m * s, stroke: INK, 'stroke-width': 1 }, svg);
    txt(svg, xs + 14, y0 + 92, '5 squares down', 'mono');
    txt(svg, xs + 14, y0 + 112, '4 inner corners', 'mono', { fill: RED, style: 'fill:' + RED });
    // square size
    var yb = y0 + R * s + 22;
    el('line', { x1: x0, y1: yb, x2: x0 + s, y2: yb, stroke: INK, 'stroke-width': 1 }, svg);
    el('line', { x1: x0, y1: yb - 5, x2: x0, y2: yb + 5, stroke: INK }, svg);
    el('line', { x1: x0 + s, y1: yb - 5, x2: x0 + s, y2: yb + 5, stroke: INK }, svg);
    txt(svg, x0 + s + 10, yb + 4, 'square_size = 34 mm', 'mono');
    txt(svg, x0 + C * s, yb + 30, '7 × 4 = 28 corners per photo', 'mono', { 'text-anchor': 'end', fill: RED, style: 'fill:' + RED });
  }

  /* ---------------------------------------------------------------- 08: coverage map */
  function drawCoverage() {
    var svg = $('#coverage-svg'), D = window.CALIB_SAMPLE; if (!svg || !D) return;
    var W = D.size[0], H = D.size[1], zw = W * 0.2, zh = H * 0.2;
    el('rect', { x: 0, y: 0, width: W, height: H, fill: '#fff' }, svg);
    for (var gx = 64; gx < W; gx += 64) el('line', { x1: gx, y1: 0, x2: gx, y2: H, stroke: '#eee' }, svg);
    for (var gy = 48; gy < H; gy += 48) el('line', { x1: 0, y1: gy, x2: W, y2: gy, stroke: '#eee' }, svg);
    var pts = [];
    D.images.forEach(function (im) { im.obs.forEach(function (p) { pts.push(p); }); });
    [[0, 0], [W - zw, 0], [0, H - zh], [W - zw, H - zh]].forEach(function (z) {
      var n = pts.filter(function (p) { return p[0] >= z[0] && p[0] < z[0] + zw && p[1] >= z[1] && p[1] < z[1] + zh; }).length;
      el('rect', { x: z[0] + 1, y: z[1] + 1, width: zw - 2, height: zh - 2, fill: 'rgba(227,6,19,0.06)', stroke: RED, 'stroke-width': 1.5, 'stroke-dasharray': '6 4' }, svg);
      txt(svg, z[0] + zw / 2, z[1] + zh / 2 + 5, n + (n === 1 ? ' corner' : ' corners'), 'mono', { 'text-anchor': 'middle', fill: RED, style: 'fill:' + RED + ';font-size:14px' });
    });
    pts.forEach(function (p) { el('circle', { cx: p[0], cy: p[1], r: 3.2, fill: INK }, svg); });
    txt(svg, W / 2, 22, '640 × 480 frame · ' + pts.length + ' corners from ' + D.images.length + ' photos', 'mono', { 'text-anchor': 'middle', fill: MUTE, style: 'fill:' + MUTE + ';font-size:13px' });
  }

  /* ---------------------------------------------------------------- 03: distortion explorer */
  function setupDistortion() {
    var cv = $('#dist-canvas'); if (!cv) return;
    var W = 640, H = 480, F = 550, CX = 320, CY = 240;
    var ctx = cv.getContext('2d');
    var dpr = Math.min(2, window.devicePixelRatio || 1);
    cv.width = W * dpr; cv.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    cv.tabIndex = 0;
    var PRESETS = {
      none:   { k1: 0, k2: 0, k3: 0, p1: 0, p2: 0 },
      barrel: { k1: -0.3, k2: 0.08, k3: 0, p1: 0, p2: 0 },
      pin:    { k1: 0.25, k2: 0.05, k3: 0, p1: 0, p2: 0 },
      tan:    { k1: 0, k2: 0, k3: 0, p1: 0.02, p2: -0.015 },
      // the lab script's coefficients for the sample images, from sample-data.js (no second copy to go stale)
      sample: window.CALIB_SAMPLE ? { k1: window.CALIB_SAMPLE.D[0], k2: window.CALIB_SAMPLE.D[1], k3: window.CALIB_SAMPLE.D[4],
                                      p1: window.CALIB_SAMPLE.D[2], p2: window.CALIB_SAMPLE.D[3] } : null
    };
    if (!PRESETS.sample) { var sb = $('#dist-presets button[data-p="sample"]'); if (sb) sb.hidden = true; }
    var KEYS = ['k1', 'k2', 'k3', 'p1', 'p2'];
    var DEC = { k1: 3, k2: 3, k3: 2, p1: 3, p2: 3 };
    var st = Object.assign({}, PRESETS.barrel);
    var probe = { u: 96, v: 72 };
    var sliders = {};
    KEYS.forEach(function (k) {
      sliders[k] = $('#' + k);
      sliders[k].addEventListener('input', function () {
        st[k] = parseFloat(sliders[k].value);
        markPreset(null);
        draw();
      });
    });
    var stats = statList($('#dist-stats'), [
      { id: 'xy', label: 'Probe x, y (normalised)' },
      { id: 'r2', label: 'r² = x² + y²' },
      { id: 'rad', label: 'Radial factor 1 + k₁r² + k₂r⁴ + k₃r⁶' },
      { id: 'tan', label: 'Tangential shift (x, y)' },
      { id: 'xyd', label: 'x_d, y_d' },
      { id: 'shift', label: 'Probe moves by', hot: true },
      { id: 'corner', label: 'Largest image-corner shift' }
    ]);
    function distort(u, v) {
      var x = (u - CX) / F, y = (v - CY) / F;
      var r2 = x * x + y * y, rad = 1 + st.k1 * r2 + st.k2 * r2 * r2 + st.k3 * r2 * r2 * r2;
      var tx = 2 * st.p1 * x * y + st.p2 * (r2 + 2 * x * x);
      var ty = st.p1 * (r2 + 2 * y * y) + 2 * st.p2 * x * y;
      var xd = x * rad + tx, yd = y * rad + ty;
      return { x: x, y: y, r2: r2, rad: rad, tx: tx, ty: ty, xd: xd, yd: yd, u: F * xd + CX, v: F * yd + CY };
    }
    function markPreset(name) {
      $$('#dist-presets button').forEach(function (b) { b.setAttribute('aria-pressed', b.dataset.p === name ? 'true' : 'false'); });
    }
    $$('#dist-presets button').forEach(function (b) {
      b.addEventListener('click', function () {
        st = Object.assign({}, PRESETS[b.dataset.p]);
        markPreset(b.dataset.p);
        draw();
      });
    });
    function polyline(pts, color, width) {
      ctx.beginPath();
      pts.forEach(function (p, i) { if (i) ctx.lineTo(p[0], p[1]); else ctx.moveTo(p[0], p[1]); });
      ctx.strokeStyle = color; ctx.lineWidth = width; ctx.stroke();
    }
    function draw() {
      KEYS.forEach(function (k) {
        sliders[k].value = st[k];
        $('#' + k + 'v').textContent = fmtS(st[k], DEC[k]);
      });
      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, W, H);
      var step = 40, u, v, pts;
      // ideal grid
      ctx.lineCap = 'round';
      for (u = 0; u <= W; u += step) polyline([[u, 0], [u, H]], FAINT, 1);
      for (v = 0; v <= H; v += step) polyline([[0, v], [W, v]], FAINT, 1);
      // distorted grid
      for (u = 0; u <= W; u += step) {
        pts = [];
        for (v = 0; v <= H; v += 4) { var d = distort(u, v); pts.push([d.u, d.v]); }
        polyline(pts, INK, (u === 0 || u === W) ? 2.4 : 1.1);
      }
      for (v = 0; v <= H; v += step) {
        pts = [];
        for (u = 0; u <= W; u += 4) { var e = distort(u, v); pts.push([e.u, e.v]); }
        polyline(pts, INK, (v === 0 || v === H) ? 2.4 : 1.1);
      }
      // principal point
      ctx.strokeStyle = MUTE; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(CX - 7, CY); ctx.lineTo(CX + 7, CY); ctx.moveTo(CX, CY - 7); ctx.lineTo(CX, CY + 7); ctx.stroke();
      // probe
      var p = distort(probe.u, probe.v);
      ctx.beginPath(); ctx.arc(probe.u, probe.v, 7, 0, 2 * Math.PI);
      ctx.fillStyle = 'rgba(255,255,255,.9)'; ctx.fill(); ctx.strokeStyle = MUTE; ctx.lineWidth = 1.5; ctx.stroke();
      ctx.beginPath(); ctx.moveTo(probe.u, probe.v); ctx.lineTo(p.u, p.v); ctx.strokeStyle = RED; ctx.lineWidth = 2; ctx.stroke();
      ctx.beginPath(); ctx.arc(p.u, p.v, 5.5, 0, 2 * Math.PI); ctx.fillStyle = RED; ctx.fill();
      var shift = Math.hypot(p.u - probe.u, p.v - probe.v);
      var cs = 0;
      [[0, 0], [W, 0], [0, H], [W, H]].forEach(function (q) { var c = distort(q[0], q[1]); cs = Math.max(cs, Math.hypot(c.u - q[0], c.v - q[1])); });
      stats.xy(fmt(p.x, 3) + ', ' + fmt(p.y, 3));
      stats.r2(fmt(p.r2, 3));
      stats.rad(fmt(p.rad, 4));
      stats.tan(fmt(p.tx, 4) + ', ' + fmt(p.ty, 4));
      stats.xyd(fmt(p.xd, 3) + ', ' + fmt(p.yd, 3));
      stats.shift(shift.toFixed(1), 'px');
      stats.corner(cs.toFixed(1), 'px');
    }
    function toImage(e) {
      var r = cv.getBoundingClientRect();
      return { u: Math.max(0, Math.min(W, (e.clientX - r.left) / r.width * W)), v: Math.max(0, Math.min(H, (e.clientY - r.top) / r.height * H)) };
    }
    var dragging = false;
    cv.addEventListener('pointerdown', function (e) { dragging = true; cv.setPointerCapture(e.pointerId); probe = toImage(e); draw(); });
    cv.addEventListener('pointermove', function (e) { if (dragging) { probe = toImage(e); draw(); } });
    cv.addEventListener('pointerup', function () { dragging = false; });
    cv.addEventListener('pointercancel', function () { dragging = false; });
    cv.addEventListener('keydown', function (e) {
      var d = e.shiftKey ? 40 : 8, k = e.key;
      if (k === 'ArrowLeft') probe.u -= d; else if (k === 'ArrowRight') probe.u += d;
      else if (k === 'ArrowUp') probe.v -= d; else if (k === 'ArrowDown') probe.v += d; else return;
      probe.u = Math.max(0, Math.min(W, probe.u)); probe.v = Math.max(0, Math.min(H, probe.v));
      e.preventDefault(); draw();
    });
    markPreset('barrel');
    draw();
  }

  /* ---------------------------------------------------------------- 12: reprojection residuals */
  function setupResiduals() {
    var svg = $('#res-svg'), D = window.CALIB_SAMPLE; if (!svg || !D) return;
    var W = D.size[0], H = D.size[1];
    var segWrap = $('#res-imgs'), mag = $('#res-mag'), magv = $('#res-magv'), bars = $('#res-bars');
    var sel = 'all';
    var short = function (name) { var m = name.match(/(\d+)\.jpg$/); return m ? m[1] : name; };
    var opts = ['all'].concat(D.images.map(function (im) { return short(im.name); }));
    opts.forEach(function (o) {
      var b = document.createElement('button');
      b.textContent = o === 'all' ? 'All' : o; b.dataset.v = o;
      b.addEventListener('click', function () { sel = o; draw(); });
      segWrap.appendChild(b);
    });
    var stats = statList($('#res-stats'), [
      { id: 'sel', label: 'RMS error, this selection', hot: true },
      { id: 'max', label: 'Largest single residual' },
      { id: 'rms', label: 'Overall RMS (ret, slide 10)' },
      { id: 'code', label: 'Script’s “total error”' }
    ]);
    mag.addEventListener('input', draw);
    function draw() {
      var M = parseFloat(mag.value);
      magv.textContent = '×' + M;
      $$('button', segWrap).forEach(function (b) { b.setAttribute('aria-pressed', b.dataset.v === sel ? 'true' : 'false'); });
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      var ims = sel === 'all' ? D.images : D.images.filter(function (im) { return short(im.name) === sel; });
      el('rect', { x: 0, y: 0, width: W, height: H, fill: '#fff' }, svg);
      if (sel !== 'all') {
        el('image', { href: 'img/sample_' + sel + '.jpg', x: 0, y: 0, width: W, height: H, opacity: 0.5, preserveAspectRatio: 'none' }, svg);
      } else {
        for (var gx = 64; gx < W; gx += 64) el('line', { x1: gx, y1: 0, x2: gx, y2: H, stroke: '#eee' }, svg);
        for (var gy = 48; gy < H; gy += 48) el('line', { x1: 0, y1: gy, x2: W, y2: gy, stroke: '#eee' }, svg);
      }
      var sq = 0, n = 0, mx = 0;
      ims.forEach(function (im) {
        im.obs.forEach(function (p, i) {
          var r = im.res[i];
          var len = Math.hypot(r[0], r[1]);
          sq += r[0] * r[0] + r[1] * r[1]; n++; if (len > mx) mx = len;
          // residual = observed - projected; draw from the detected corner towards the projected one
          el('line', { x1: p[0], y1: p[1], x2: p[0] - r[0] * M, y2: p[1] - r[1] * M, stroke: RED, 'stroke-width': 2, 'stroke-linecap': 'round' }, svg);
          el('circle', { cx: p[0], cy: p[1], r: 3, fill: INK, stroke: '#fff', 'stroke-width': 1 }, svg);
        });
      });
      // scale bar: 0.5 px magnified
      var L = 0.5 * M, bx = W - 16 - L, by = H - 18;
      el('rect', { x: bx - 10, y: by - 22, width: L + 20, height: 34, fill: 'rgba(255,255,255,.88)' }, svg);
      el('line', { x1: bx, y1: by, x2: bx + L, y2: by, stroke: RED, 'stroke-width': 3 }, svg);
      txt(svg, bx + L, by - 8, '0.5 px × ' + M, 'mono', { 'text-anchor': 'end', style: 'font-size:12px' });
      stats.sel(Math.sqrt(sq / n).toFixed(3), 'px');
      stats.max(mx.toFixed(2), 'px');
      stats.rms(D.rms.toFixed(3), 'px');
      stats.code(D.code_total_error.toFixed(3));
      drawBars();
    }
    function drawBars() {
      while (bars.firstChild) bars.removeChild(bars.firstChild);
      var x0 = 30, y0 = 12, w = 262, h = 108, maxv = 1.0;
      var bw = w / D.images.length;
      [0, 0.5, 1].forEach(function (t) {
        var y = y0 + h - t / maxv * h;
        el('line', { x1: x0, y1: y, x2: x0 + w, y2: y, stroke: t ? '#e2e2e2' : INK, 'stroke-width': 1 }, bars);
        txt(bars, x0 - 5, y + 4, t.toFixed(1), 'mono', { 'text-anchor': 'end', style: 'font:500 10px var(--mono);fill:' + MUTE });
      });
      D.images.forEach(function (im, i) {
        var s = short(im.name), on = sel === s, v = im.rms;
        var bh = v / maxv * h, x = x0 + i * bw + bw * 0.2;
        var r = el('rect', { x: x, y: y0 + h - bh, width: bw * 0.6, height: bh, fill: on ? RED : INK, style: 'cursor:pointer' }, bars);
        r.addEventListener('click', function () { sel = s; draw(); });
        var t = el('title', {}, r); t.textContent = im.name + ': ' + v.toFixed(3) + ' px RMS';
        txt(bars, x + bw * 0.3, y0 + h + 14, s, 'mono', { 'text-anchor': 'middle', style: 'font:500 10px var(--mono);fill:' + (on ? RED : INK) });
        txt(bars, x + bw * 0.3, y0 + h - bh - 4, v.toFixed(2), 'mono', { 'text-anchor': 'middle', style: 'font:500 9.5px var(--mono);fill:' + (on ? RED : MUTE) });
      });
      var yr = y0 + h - D.rms / maxv * h;
      el('line', { x1: x0, y1: yr, x2: x0 + w, y2: yr, stroke: INK, 'stroke-dasharray': '4 3', 'stroke-width': 1 }, bars);
      txt(bars, x0, y0 + h + 34, 'RMS per photo (px)', 'mono', { style: 'font:500 10px var(--mono);letter-spacing:.06em;text-transform:uppercase;fill:' + MUTE });
      el('line', { x1: x0, y1: y0 + h + 48, x2: x0 + 18, y2: y0 + h + 48, stroke: INK, 'stroke-dasharray': '4 3', 'stroke-width': 1 }, bars);
      txt(bars, x0 + 24, y0 + h + 52, 'overall RMS = ' + D.rms.toFixed(3) + ' px', 'mono', { style: 'font:500 10px var(--mono);fill:' + INK });
    }
    draw();
  }

  /* ---------------------------------------------------------------- 13: undistortion before/after */
  function setupUndistort() {
    var img = $('#und-img'); if (!img) return;
    var V = {
      a0:    { src: 'img/undistort_alpha0.jpg', cap: 'alpha = 0', size: '640 × 480', w: 100, K: [561.19, 560.93, 362.05, 229.46], roi: '(0, 0, 639, 479)' },
      a1:    { src: 'img/undistort_alpha1.jpg', cap: 'alpha = 1', size: '640 × 480', w: 100, K: [558.20, 556.38, 361.89, 230.12], roi: '(2, 3, 636, 475)' },
      a1roi: { src: 'img/undistort_alpha1_roi.jpg', cap: 'alpha = 1, ROI in red', size: '640 × 480', w: 100, K: [558.20, 556.38, 361.89, 230.12], roi: '(2, 3, 636, 475)' },
      crop:  { src: 'img/undistort_alpha1_cropped.jpg', cap: 'Cropped to ROI (what the script shows)', size: '636 × 475', w: 636 / 640 * 100, K: [558.20, 556.38, 361.89, 230.12], roi: '(2, 3, 636, 475)' }
    };
    var stats = statList($('#und-stats'), [
      { id: 'f', label: 'New fx, fy (px)' },
      { id: 'c', label: 'New cx, cy (px)' },
      { id: 'roi', label: 'roi = (x, y, w, h)', hot: true },
      { id: 'size', label: 'Image shown' }
    ]);
    $('#und-stats').style.display = 'grid';
    $('#und-stats').style.gridTemplateColumns = 'repeat(auto-fit, minmax(170px, 1fr))';
    $('#und-stats').style.columnGap = '16px';
    function show(k) {
      var v = V[k];
      img.src = v.src; img.style.width = v.w + '%';
      img.alt = 'calibration_image_02 undistorted: ' + v.cap;
      $('#und-cap').textContent = v.cap; $('#und-size').textContent = v.size;
      stats.f(v.K[0].toFixed(1) + ', ' + v.K[1].toFixed(1));
      stats.c(v.K[2].toFixed(1) + ', ' + v.K[3].toFixed(1));
      stats.roi(v.roi);
      stats.size(v.size);
      $$('#und-seg button').forEach(function (b) { b.setAttribute('aria-pressed', b.dataset.v === k ? 'true' : 'false'); });
    }
    $$('#und-seg button').forEach(function (b) { b.addEventListener('click', function () { show(b.dataset.v); }); });
    // preload
    Object.keys(V).forEach(function (k) { var i = new Image(); i.src = V[k].src; });
    show('crop');
  }

  /* ---------------------------------------------------------------- folded code */
  // Code walk-throughs start closed. Open them all for printing, and open the one a link points into.
  function setupFolds() {
    var folds = $$('details.fold');
    if (!folds.length) return;
    window.addEventListener('beforeprint', function () {
      folds.forEach(function (d) { d.dataset.wasOpen = d.open ? '1' : ''; d.open = true; });
    });
    window.addEventListener('afterprint', function () {
      folds.forEach(function (d) { if (d.dataset.wasOpen !== undefined) d.open = d.dataset.wasOpen === '1'; });
    });
    function openTarget() {
      var id = location.hash.slice(1);
      try { id = decodeURIComponent(id); } catch (e) { /* a cut-off link: use the hash as it is */ }
      var t = id && document.getElementById(id);
      var d = t && t.closest('details.fold');
      if (d && !d.open) { d.open = true; t.scrollIntoView(); }
    }
    window.addEventListener('hashchange', openTarget);
    openTarget();
  }

  function init() {
    setupCode();
    setupFolds();
    setupMath();
    setupNav();
    setupChecklist();
    drawBoard();
    drawCoverage();
    setupDistortion();
    setupResiduals();
    setupUndistort();
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
