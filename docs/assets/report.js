(function () {
  "use strict";

  var root = document.documentElement;
  var storageKey = "ffn-report-theme";

  function preferredTheme() {
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches
      ? "dark"
      : "light";
  }

  function currentTheme() {
    return root.getAttribute("data-theme") || preferredTheme();
  }

  function updateThemeButton() {
    var button = document.querySelector("[data-theme-toggle]");
    if (!button) return;
    var theme = currentTheme();
    button.textContent = theme === "dark" ? "☀" : "☾";
    button.setAttribute("aria-label", theme === "dark" ? "切换到浅色主题" : "切换到深色主题");
    button.title = theme === "dark" ? "浅色主题" : "深色主题";
  }

  function initTheme() {
    try {
      var saved = localStorage.getItem(storageKey);
      if (saved === "light" || saved === "dark") root.setAttribute("data-theme", saved);
    } catch (_) {
      /* Local files may deny storage; the system preference remains a safe fallback. */
    }
    updateThemeButton();
    var button = document.querySelector("[data-theme-toggle]");
    if (!button) return;
    button.addEventListener("click", function () {
      var next = currentTheme() === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      try { localStorage.setItem(storageKey, next); } catch (_) { /* no-op */ }
      updateThemeButton();
    });
  }

  function initProgress() {
    var bar = document.querySelector(".reading-progress");
    if (!bar) return;
    var scheduled = false;
    function update() {
      var doc = document.documentElement;
      var max = Math.max(1, doc.scrollHeight - doc.clientHeight);
      bar.style.width = Math.min(100, Math.max(0, window.scrollY / max * 100)) + "%";
      scheduled = false;
    }
    window.addEventListener("scroll", function () {
      if (!scheduled) {
        scheduled = true;
        window.requestAnimationFrame(update);
      }
    }, { passive: true });
    update();
  }

  function initToc() {
    var links = Array.prototype.slice.call(document.querySelectorAll(".toc a[href^='#']"));
    if (!links.length || !("IntersectionObserver" in window)) return;
    var byId = {};
    links.forEach(function (link) { byId[link.getAttribute("href").slice(1)] = link; });
    var sections = Object.keys(byId).map(function (id) { return document.getElementById(id); }).filter(Boolean);
    var visible = {};
    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) { visible[entry.target.id] = entry.isIntersecting; });
      var active = sections.filter(function (section) { return visible[section.id]; })[0];
      if (!active) {
        active = sections.slice().reverse().find(function (section) {
          return section.getBoundingClientRect().top < 150;
        });
      }
      links.forEach(function (link) { link.classList.remove("active"); });
      if (active && byId[active.id]) byId[active.id].classList.add("active");
    }, { rootMargin: "-90px 0px -68% 0px", threshold: [0, 0.01] });
    sections.forEach(function (section) { observer.observe(section); });
  }

  function el(name, attrs, parent, text) {
    var node = document.createElementNS("http://www.w3.org/2000/svg", name);
    Object.keys(attrs || {}).forEach(function (key) { node.setAttribute(key, attrs[key]); });
    if (typeof text !== "undefined") node.textContent = text;
    if (parent) parent.appendChild(node);
    return node;
  }

  function signed(value, digits) {
    if (value === 0) return Number(value).toFixed(digits);
    return (value > 0 ? "+" : "−") + Math.abs(value).toFixed(digits);
  }

  function renderRangeChart(host, config) {
    var rows = config.rows || [];
    var W = config.width || 720;
    var labelW = config.labelWidth || 112;
    var valueW = config.valueWidth || 58;
    var right = config.right || 18;
    var top = 12;
    var rowH = config.rowHeight || 40;
    var bottom = 34;
    var plotW = W - labelW - valueW - right;
    var H = top + rows.length * rowH + bottom;
    var min = config.domain[0];
    var max = config.domain[1];
    var x = function (v) { return labelW + (v - min) / (max - min) * plotW; };
    var svg = el("svg", {
      viewBox: "0 0 " + W + " " + H,
      role: "img",
      "aria-label": config.aria || "数值范围与均值图"
    });

    rows.forEach(function (row, index) {
      if (!row.shade) return;
      el("rect", {
        x: 0, y: top + index * rowH,
        width: W, height: rowH,
        fill: "var(--down-soft)", opacity: 0.48
      }, svg);
    });

    if (config.band) {
      el("rect", {
        x: x(config.band[0]), y: top,
        width: x(config.band[1]) - x(config.band[0]),
        height: rows.length * rowH,
        fill: "var(--slate-soft)"
      }, svg);
    }

    var ticks = config.ticks || [];
    ticks.forEach(function (tick) {
      var zero = tick === 0;
      el("line", {
        x1: x(tick), x2: x(tick), y1: top, y2: top + rows.length * rowH,
        stroke: zero ? "var(--ink-3)" : "var(--line)",
        "stroke-width": zero ? 1.3 : 1,
        "stroke-dasharray": zero ? "" : "2 4"
      }, svg);
      el("text", {
        x: x(tick), y: H - 10, "text-anchor": "middle",
        fill: "var(--ink-3)", "font-size": 10.5,
        "font-family": "var(--mono)"
      }, svg, signed(tick, config.tickDigits || 0));
    });

    rows.forEach(function (row, index) {
      var cy = top + index * rowH + rowH / 2;
      var color = row.mean < 0 ? "var(--down)" : "var(--up)";
      if (row.tone === "slate") color = "var(--slate)";
      if (row.tone === "amber") color = "var(--amber)";

      el("text", {
        x: labelW - 12, y: cy + 0.5, "text-anchor": "end", "dominant-baseline": "middle",
        fill: "var(--ink)", "font-size": row.emphasis ? 12.5 : 12,
        "font-weight": row.emphasis ? 700 : 450,
        "font-family": "var(--sans)"
      }, svg, row.label);

      var lo = Math.min(row.min, row.max);
      var hi = Math.max(row.min, row.max);
      el("rect", {
        x: x(lo), y: cy - 5, width: Math.max(2, x(hi) - x(lo)), height: 10,
        rx: 5, fill: color, opacity: 0.24
      }, svg);
      el("line", {
        x1: x(row.mean), x2: x(row.mean), y1: cy - 9, y2: cy + 9,
        stroke: "var(--ink)", "stroke-width": 2.2
      }, svg);
      el("circle", {
        cx: x(row.mean), cy: cy, r: 4.4,
        fill: color, stroke: "var(--card)", "stroke-width": 1.5
      }, svg);
      el("text", {
        x: W - right, y: cy + 0.5, "text-anchor": "end", "dominant-baseline": "middle",
        fill: color, "font-size": 11, "font-weight": 700,
        "font-family": "var(--mono)"
      }, svg, signed(row.mean, config.valueDigits || 2));
    });

    el("text", {
      x: labelW + plotW / 2, y: H - 1, "text-anchor": "middle",
      fill: "var(--ink-3)", "font-size": 9.5, "font-family": "var(--mono)"
    }, svg, config.unit || "");
    host.replaceChildren(svg);
  }

  function renderSeedChart(host, config) {
    var groups = config.groups || [];
    var W = config.width || 760;
    var H = config.height || 330;
    var left = config.left || 66;
    var right = config.right || 24;
    var top = config.top || 22;
    var bottom = config.bottom || 54;
    var plotW = W - left - right;
    var plotH = H - top - bottom;
    var min = config.domain[0];
    var max = config.domain[1];
    var y = function (v) { return top + (max - v) / (max - min) * plotH; };
    var center = function (index) { return left + plotW * (index + 0.5) / groups.length; };
    var svg = el("svg", {
      viewBox: "0 0 " + W + " " + H,
      role: "img",
      "aria-label": config.aria || "逐随机种子的总体分布"
    });

    if (config.band) {
      el("rect", {
        x: left, y: y(config.band[1]), width: plotW,
        height: y(config.band[0]) - y(config.band[1]),
        fill: "var(--slate-soft)"
      }, svg);
    }

    (config.ticks || []).forEach(function (tick) {
      var zero = tick === 0;
      el("line", {
        x1: left, x2: W - right, y1: y(tick), y2: y(tick),
        stroke: zero ? "var(--ink-3)" : "var(--line)",
        "stroke-width": zero ? 1.3 : 1,
        "stroke-dasharray": zero ? "" : "2 4"
      }, svg);
      el("text", {
        x: left - 11, y: y(tick) + 0.5, "text-anchor": "end",
        "dominant-baseline": "middle", fill: "var(--ink-3)",
        "font-size": 10.5, "font-family": "var(--mono)"
      }, svg, signed(tick, config.tickDigits || 1));
    });

    groups.forEach(function (group, groupIndex) {
      var values = group.values || [];
      var color = group.tone === "down" ? "var(--down)" : "var(--up)";
      var cx = center(groupIndex);
      var spread = Math.min(82, plotW / groups.length * 0.42);
      values.forEach(function (value, valueIndex) {
        var pointLabel = "seed " + (group.seedStart + valueIndex) + "：" + signed(value, 3) + "pp";
        var offset = values.length > 1
          ? (valueIndex / (values.length - 1) - 0.5) * spread
          : 0;
        var point = el("circle", {
          cx: cx + offset, cy: y(value), r: 4.6,
          fill: color, opacity: 0.78,
          stroke: "var(--card)", "stroke-width": 1.2,
          tabindex: 0, role: "img", "aria-label": pointLabel
        }, svg);
        el("title", {}, point, pointLabel);
      });

      el("line", {
        x1: cx - spread * 0.72, x2: cx + spread * 0.72,
        y1: y(group.mean), y2: y(group.mean),
        stroke: "var(--ink)", "stroke-width": 3
      }, svg);
      el("text", {
        x: cx + spread * 0.82, y: y(group.mean) - 7,
        "text-anchor": "start", fill: color,
        "font-size": 11, "font-weight": 700,
        "font-family": "var(--mono)"
      }, svg, signed(group.mean, 2) + "pp");
      el("text", {
        x: cx, y: H - 20, "text-anchor": "middle",
        fill: "var(--ink)", "font-size": 12.5,
        "font-weight": 700, "font-family": "var(--sans)"
      }, svg, group.label);
    });

    el("text", {
      x: 14, y: top + plotH / 2, "text-anchor": "middle",
      transform: "rotate(-90 14 " + (top + plotH / 2) + ")",
      fill: "var(--ink-3)", "font-size": 9.5,
      "font-family": "var(--mono)"
    }, svg, config.unit || "");
    host.replaceChildren(svg);
  }

  function renderAllCharts() {
    var charts = window.REPORT_CHARTS || {};
    document.querySelectorAll("[data-range-chart]").forEach(function (host) {
      var key = host.getAttribute("data-range-chart");
      if (charts[key]) renderRangeChart(host, charts[key]);
    });
    var seedCharts = window.REPORT_SEED_CHARTS || {};
    document.querySelectorAll("[data-seed-chart]").forEach(function (host) {
      var key = host.getAttribute("data-seed-chart");
      if (seedCharts[key]) renderSeedChart(host, seedCharts[key]);
    });
  }

  function initPermutationDemo() {
    document.querySelectorAll("[data-permutation-demo]").forEach(function (demo) {
      var button = demo.querySelector("[data-permutation-toggle]");
      var target = demo.querySelector("[data-permutation-target]");
      if (!button || !target) return;
      var original = (target.getAttribute("data-original-order") || "a,b,c,d").split(",");
      var permuted = (target.getAttribute("data-permuted-order") || "c,a,d,b").split(",");
      var state = target.getAttribute("data-state") || "permuted";

      function reorder(order) {
        order.forEach(function (key) {
          var item = target.querySelector("[data-neuron='" + key + "']");
          if (item) target.appendChild(item);
        });
      }

      function sync() {
        button.textContent = state === "permuted" ? "恢复原编号" : "执行 π = (3, 1, 4, 2)";
        button.setAttribute("aria-pressed", state === "permuted" ? "true" : "false");
      }

      reorder(state === "permuted" ? permuted : original);
      sync();
      button.addEventListener("click", function () {
        state = state === "permuted" ? "original" : "permuted";
        target.setAttribute("data-state", state);
        reorder(state === "permuted" ? permuted : original);
        sync();
      });
    });
  }

  function init() {
    initTheme();
    initProgress();
    initToc();
    initPermutationDemo();
    renderAllCharts();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
