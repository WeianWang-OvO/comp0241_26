/* Stereo calibration tutorial: code blocks, maths, contents, and the depth-from-disparity widget. */
(function () {
  'use strict';
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };
  var SVGNS = 'http://www.w3.org/2000/svg';
  function el(tag, attrs, parent) {
    var e = document.createElementNS(SVGNS, tag);
    Object.keys(attrs || {}).forEach(function (k) { e.setAttribute(k, attrs[k]); });
    if (parent) parent.appendChild(e);
    return e;
  }
  function text(parent, x, y, s, attrs) {
    var t = el('text', Object.assign({ x: x, y: y }, attrs || {}), parent);
    t.textContent = s;
    return t;
  }

  /* ---------------------------------------------------------------- code blocks (as in Week 3) */
  function setupCode() {
    $$('pre code').forEach(function (code) {
      code.parentNode.parentNode.dataset.raw = code.textContent;
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
        var raw = btn.closest('.code').dataset.raw || '';
        var done = function () { btn.textContent = 'Copied'; setTimeout(function () { btn.textContent = 'Copy'; }, 1400); };
        if (navigator.clipboard && window.isSecureContext) {
          navigator.clipboard.writeText(raw).then(done, function () { fallbackCopy(raw); done(); });
        } else { fallbackCopy(raw); done(); }
      });
    });
  }
  function fallbackCopy(s) {
    var ta = document.createElement('textarea');
    ta.value = s; ta.setAttribute('readonly', ''); ta.style.position = 'fixed'; ta.style.opacity = '0';
    document.body.appendChild(ta); ta.select();
    try { document.execCommand('copy'); } catch (e) { /* ignore */ }
    document.body.removeChild(ta);
  }

  function setupMath() {
    if (!window.renderMathInElement) return;
    window.renderMathInElement(document.body, {
      delimiters: [{ left: '$$', right: '$$', display: true }, { left: '\\(', right: '\\)', display: false }],
      throwOnError: false
    });
  }

  /* ---------------------------------------------------------------- contents + progress (as in Week 3) */
  function setupNav() {
    var toc = $('.toc'), hero = $('.hero');
    var sections = $$('section.s');
    if (toc && hero) {
      var d = document.createElement('details'); d.className = 'toc-inline';
      var parts = $$('section.part-s').length;
      var sm = document.createElement('summary');
      sm.textContent = 'Contents · ' + (parts ? parts + ' parts' : sections.length + ' sections'); d.appendChild(sm);
      var wrap = document.createElement('div'); wrap.className = 'toc-inline-body';
      // only the contents' own top-level children: the nested lists come along inside their parts
      Array.prototype.slice.call(toc.children).forEach(function (n) {
        if (n.matches('.part, ol')) wrap.appendChild(n.cloneNode(true));
      });
      d.appendChild(wrap);
      hero.parentNode.insertBefore(d, hero.nextSibling);
      $$('a', d).forEach(function (a) { a.addEventListener('click', function () { d.open = false; }); });
    }
    var links = $$('.toc a'), byId = {};
    links.forEach(function (a) { byId[a.getAttribute('href').slice(1)] = a; });
    sections = $$('section.part-s, section.s');  // parts and their subsections, in page order
    var bar = $('#progress');
    function onScroll() {
      var h = document.documentElement;
      var f = h.scrollTop / Math.max(1, h.scrollHeight - h.clientHeight);
      if (bar) bar.style.width = (100 * Math.min(1, Math.max(0, f))) + '%';
      var cur = null;
      for (var i = 0; i < sections.length; i++) if (sections[i].getBoundingClientRect().top < 140) cur = sections[i].id;
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

  /* ---------------------------------------------------------------- depth from disparity */
  // Rectified pair seen from above: two cameras B mm apart, a point straight ahead at distance Z.
  // d = f B / Z pixels; a matching error of e pixels moves the depth by about Z^2 e / (f B).
  function setupDepth() {
    var svg = $('#dz-svg'), plot = $('#dz-plot');
    if (!svg || !plot) return;
    var inputs = { B: $('#dz-B'), f: $('#dz-f'), Z: $('#dz-Z'), e: $('#dz-e') };
    var outs = { B: $('#dz-Bv'), f: $('#dz-fv'), Z: $('#dz-Zv'), e: $('#dz-ev') };
    var stats = $('#dz-stats');
    var rows = {};
    [['d', 'Disparity d = f·B/Z'], ['dz', 'Depth error ±'], ['rel', 'Relative depth error'], ['nd', 'numDisparities to reach Z']].forEach(function (r) {
      var div = document.createElement('div'); div.className = 'stat' + (r[0] === 'dz' ? ' hot' : '');
      var dt = document.createElement('dt'); dt.textContent = r[1];
      var dd = document.createElement('dd');
      div.appendChild(dt); div.appendChild(dd); stats.appendChild(div);
      rows[r[0]] = dd;
    });
    function set(dd, v, unit) {
      dd.textContent = v;
      if (unit) { var u = document.createElement('span'); u.className = 'u'; u.textContent = unit; dd.appendChild(u); }
    }
    var sample = $('#dz-sample');
    if (sample) sample.addEventListener('click', function () { inputs.B.value = 80; inputs.f.value = 637; inputs.Z.value = 450; inputs.e.value = 0.25; draw(); });

    function draw() {
      var B = +inputs.B.value, f = +inputs.f.value, Z = +inputs.Z.value, e = +inputs.e.value;
      outs.B.textContent = B + ' mm'; outs.f.textContent = f + ' px'; outs.Z.textContent = Z + ' mm'; outs.e.textContent = e.toFixed(2) + ' px';
      var d = f * B / Z, dz = Z * Z * e / (f * B);
      set(rows.d, d.toFixed(1), 'px');
      set(rows.dz, dz < 10 ? dz.toFixed(1) : dz.toFixed(0), 'mm');
      set(rows.rel, (100 * dz / Z).toFixed(2), '%');
      set(rows.nd, String(16 * Math.ceil(d / 16)), 'px');

      // ---- scene (schematic, not to scale: the baseline is drawn larger)
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      var W = 520, H = 300, cx = W / 2, camY = 262, zMax = 3000;
      var half = 40 + 140 * (B - 40) / 160;            // drawn half-baseline
      var py = camY - 26 - (camY - 60) * Math.min(Z, zMax) / zMax; // drawn point height
      var L = cx - half, R = cx + half, fl = 26;        // camera centres, drawn focal length
      el('line', { x1: 20, y1: camY, x2: W - 20, y2: camY, stroke: '#ddd' }, svg);
      [L, R].forEach(function (x, i) {
        el('line', { x1: x - 34, y1: camY - fl, x2: x + 34, y2: camY - fl, stroke: '#000', 'stroke-width': 2 }, svg);
        el('path', { d: 'M' + (x - 9) + ' ' + (camY + 12) + 'L' + x + ' ' + camY + 'L' + (x + 9) + ' ' + (camY + 12) + 'Z', fill: '#000' }, svg);
        text(svg, x, camY + 30, i ? 'right' : 'left', { 'text-anchor': 'middle', class: 'dz-lab' });
        el('line', { x1: x, y1: camY, x2: x, y2: 40, stroke: '#bbb', 'stroke-dasharray': '3 4' }, svg);
      });
      var P = { x: cx, y: py };
      [L, R].forEach(function (x) {
        el('line', { x1: P.x, y1: P.y, x2: x, y2: camY, stroke: '#e30613', 'stroke-width': 1.5 }, svg);
      });
      // where the rays cross each image line, relative to that camera's centre
      var hitL = L + (P.x - L) * fl / (camY - P.y), hitR = R + (P.x - R) * fl / (camY - P.y);
      el('circle', { cx: hitL, cy: camY - fl, r: 4, fill: '#e30613' }, svg);
      el('circle', { cx: hitR, cy: camY - fl, r: 4, fill: '#e30613' }, svg);
      el('circle', { cx: P.x, cy: P.y, r: 6, fill: '#e30613' }, svg);
      text(svg, P.x + 12, P.y + 4, 'Z = ' + Z + ' mm', { class: 'dz-lab red' });
      el('line', { x1: L, y1: camY + 44, x2: R, y2: camY + 44, stroke: '#000' }, svg);
      text(svg, cx, camY + 40, 'B = ' + B + ' mm', { 'text-anchor': 'middle', class: 'dz-lab' });
      text(svg, 20, 24, 'Seen from above · not to scale', { class: 'dz-lab muted' });
      text(svg, W - 20, 24, 'x_left − x_right = ' + d.toFixed(1) + ' px', { 'text-anchor': 'end', class: 'dz-lab' });

      // ---- depth error against distance
      while (plot.firstChild) plot.removeChild(plot.firstChild);
      var PW = 520, PH = 180, x0 = 56, x1 = PW - 16, y0 = PH - 30, y1 = 14;
      var zs = [], z;
      for (z = 200; z <= 3000; z += 25) zs.push(z);
      var errMax = 3000 * 3000 * e / (f * B);
      var mag = Math.pow(10, Math.floor(Math.log10(errMax)));  // axis top: the next 1, 2 or 5 times a power of ten
      var top = [1, 2, 5, 10].map(function (m) { return m * mag; }).filter(function (v) { return v >= errMax; })[0];
      var X = function (zz) { return x0 + (x1 - x0) * (zz - 200) / 2800; };
      var Y = function (er) { return y0 - (y0 - y1) * Math.min(er, top) / top; };
      el('line', { x1: x0, y1: y0, x2: x1, y2: y0, stroke: '#000' }, plot);
      el('line', { x1: x0, y1: y0, x2: x0, y2: y1, stroke: '#000' }, plot);
      [500, 1000, 1500, 2000, 2500, 3000].forEach(function (zz) {
        el('line', { x1: X(zz), y1: y0, x2: X(zz), y2: y0 + 4, stroke: '#000' }, plot);
        text(plot, X(zz), y0 + 17, (zz / 1000) + ' m', { 'text-anchor': 'middle', class: 'dz-tick' });
      });
      [0, 0.5, 1].forEach(function (q) {
        el('line', { x1: x0 - 4, y1: Y(q * top), x2: x0, y2: Y(q * top), stroke: '#000' }, plot);
        text(plot, x0 - 7, Y(q * top) + 4, String(+(q * top).toFixed(1)), { 'text-anchor': 'end', class: 'dz-tick' });
      });
      text(plot, x0 + 6, y1 + 4, 'depth error ± (mm)', { class: 'dz-tick' });
      el('polyline', { points: zs.map(function (zz) { return X(zz) + ',' + Y(zz * zz * e / (f * B)); }).join(' '), fill: 'none', stroke: '#000', 'stroke-width': 2 }, plot);
      var zc = Math.min(Math.max(Z, 200), 3000);
      el('circle', { cx: X(zc), cy: Y(dz), r: 5, fill: '#e30613' }, plot);
    }
    Object.keys(inputs).forEach(function (k) { inputs[k].addEventListener('input', draw); });
    draw();
  }

  /* ---------------------------------------------------------------- folded code (as in Week 3) */
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

  function init() { setupCode(); setupFolds(); setupMath(); setupNav(); setupDepth(); }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
