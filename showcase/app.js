/* ═══════════════════════════════════════
   BloodIQ Showcase — app.js
   Interactive charts, scroll animations,
   lightbox, and active nav tracking
   ═══════════════════════════════════════ */

'use strict';

// ─── Chart.js Global Defaults ───
Chart.defaults.color = '#94a3b8';
Chart.defaults.borderColor = 'rgba(99,102,241,0.1)';
Chart.defaults.font.family = "'Inter', system-ui, sans-serif";

const ACCENT   = '#6366f1';
const ACCENT2  = '#8b5cf6';
const ACCENT3  = '#06b6d4';
const GREEN    = '#10b981';
const RED      = '#ef4444';
const YELLOW   = '#f59e0b';
const PINK     = '#ec4899';

// ─── Gradient helper ───
function makeGradient(ctx, color1, color2) {
  const g = ctx.createLinearGradient(0, 0, 400, 0);
  g.addColorStop(0, color1);
  g.addColorStop(1, color2);
  return g;
}

// ════════════════════════════════════════
// 1. BASELINE COMPARISON CHART
// ════════════════════════════════════════
function initBaselineChart() {
  const canvas = document.getElementById('baselineChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  const labels = ['Hamming Loss ↓', 'Subset Accuracy ↑', 'Micro F1 ↑', 'Macro F1 ↑'];
  const models = {
    'RF Base':    [0.2454, 0.3047, 0.4845, 0.4694],
    'RF Balanced':[0.2772, 0.2326, 0.4817, 0.4715],
    'KNN':        [0.4313, 0.0527, 0.3946, 0.3756],
    'SVM':        [0.3509, 0.1047, 0.4066, 0.3872],
  };
  const colors = [ACCENT, ACCENT2, ACCENT3, GREEN];

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: Object.entries(models).map(([name, data], i) => ({
        label: name,
        data,
        backgroundColor: colors[i] + '33',
        borderColor: colors[i],
        borderWidth: 1.5,
        borderRadius: 6,
      })),
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      plugins: {
        legend: { position: 'top', labels: { padding: 16, boxWidth: 12 } },
        tooltip: {
          callbacks: {
            label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y.toFixed(4)}`,
          },
        },
      },
      scales: {
        x: { grid: { color: 'rgba(99,102,241,0.08)' } },
        y: {
          grid: { color: 'rgba(99,102,241,0.08)' },
          min: 0, max: 0.55,
          ticks: { callback: v => v.toFixed(2) },
        },
      },
      animation: { duration: 1200, easing: 'easeOutQuart' },
    },
  });
}

// ════════════════════════════════════════
// 2. ENSEMBLE COMPARISON CHART
// ════════════════════════════════════════
function initEnsembleChart() {
  const canvas = document.getElementById('ensembleChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  // Show Macro F1 for Baseline vs Ensemble (real models only, excluding inflated SMOTE/ADA)
  const data = {
    labels: ['RF Base', 'RF Balanced', 'KNN', 'SVM', 'XGB Weighted', 'RF Tuned', 'LGBM Weighted'],
    macro_f1: [0.4694, 0.4715, 0.3756, 0.3872, 0.4438, 0.4463, 0.4496],
    micro_f1: [0.4845, 0.4817, 0.3946, 0.4066, 0.4544, 0.4607, 0.4607],
  };

  const bgColors = [
    ACCENT+'55', ACCENT+'44', ACCENT3+'44', PINK+'44',
    YELLOW+'55', GREEN+'55', ACCENT2+'55',
  ];
  const borderColors = [ACCENT, ACCENT, ACCENT3, PINK, YELLOW, GREEN, ACCENT2];

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: data.labels,
      datasets: [
        {
          label: 'Macro F1',
          data: data.macro_f1,
          backgroundColor: bgColors,
          borderColor: borderColors,
          borderWidth: 1.5,
          borderRadius: 6,
        },
        {
          label: 'Micro F1',
          data: data.micro_f1,
          backgroundColor: bgColors.map(c => c.replace('55','22').replace('44','22')),
          borderColor: borderColors,
          borderWidth: 1.5,
          borderRadius: 6,
          borderDash: [4,4],
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      plugins: {
        legend: { position: 'top', labels: { padding: 16, boxWidth: 12 } },
        tooltip: { callbacks: { label: c => ` ${c.dataset.label}: ${c.parsed.y.toFixed(4)}` } },
        annotation: {
          annotations: {
            separator: {
              type: 'line',
              xMin: 3.5, xMax: 3.5,
              borderColor: 'rgba(99,102,241,0.4)',
              borderWidth: 1.5,
              borderDash: [6,3],
              label: {
                display: true, content: 'Ensemble →',
                color: ACCENT, backgroundColor: 'transparent',
                position: 'start', font: { size: 10 },
              },
            },
          },
        },
      },
      scales: {
        x: { grid: { display: false } },
        y: {
          grid: { color: 'rgba(99,102,241,0.08)' },
          min: 0.3, max: 0.55,
          ticks: { callback: v => v.toFixed(2) },
        },
      },
      animation: { duration: 1400, easing: 'easeOutQuart' },
    },
  });
}

// ════════════════════════════════════════
// 3. LEADERBOARD RADAR / HORIZONTAL BAR
// ════════════════════════════════════════
function initLeaderboardChart() {
  const canvas = document.getElementById('leaderboardChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  // Full model leaderboard by Macro F1 (sorted descending, showing all)
  const models = [
    'XGB_SMOTE', 'AdaBoost', 'LGBM_weighted', 'RF_tuned', 'XGB_weighted',
    'RF_balanced', 'RF_base', 'SVM', 'KNN'
  ];
  const macroF1 = [0.9943, 0.8437, 0.4496, 0.4463, 0.4438, 0.4715, 0.4694, 0.3872, 0.3756];
  const colors = [
    '#c4b5fd', '#fca5a5', '#6ee7b7', '#86efac', '#fcd34d',
    '#a5b4fc', '#a5b4fc', '#f9a8d4', '#67e8f9',
  ];
  const borderColors = [
    '#8b5cf6', '#ef4444', '#10b981', '#10b981', '#f59e0b',
    '#6366f1', '#6366f1', '#ec4899', '#06b6d4',
  ];

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: models,
      datasets: [{
        label: 'Macro F1',
        data: macroF1,
        backgroundColor: colors.map(c => c + '55'),
        borderColor: borderColors,
        borderWidth: 2,
        borderRadius: 6,
      }],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: true,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: c => {
              const note = c.parsed.x > 0.9 ? ' ⚠ synthetic data inflation' : '';
              return ` Macro F1: ${c.parsed.x.toFixed(4)}${note}`;
            },
          },
        },
      },
      scales: {
        x: {
          grid: { color: 'rgba(99,102,241,0.08)' },
          min: 0, max: 1.0,
          ticks: { callback: v => v.toFixed(2) },
        },
        y: { grid: { display: false } },
      },
      animation: { duration: 1200, easing: 'easeOutQuart' },
    },
  });
}

// ════════════════════════════════════════
// 4. SCROLL ANIMATIONS (IntersectionObserver)
// ════════════════════════════════════════
function initScrollAnimations() {
  // Fade-in sections
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.08, rootMargin: '0px 0px -50px 0px' });

  document.querySelectorAll(
    '.plot-card, .problem-card, .metric-big-card, .pipeline-step, ' +
    '.dataset-card, .novelty-strip, .label-distribution, ' +
    '.chart-container, .metrics-table-wrap, .smote-explainer, ' +
    '.threshold-tuning-card, .final-model-hero, .best-per-label, ' +
    '.novelty-section, .cluster-size-visual, .threshold-comparison, ' +
    '.label-logic-card, .syllabus-coverage, .tech-stack-row'
  ).forEach(el => {
    el.classList.add('fade-in');
    observer.observe(el);
  });
}

// ════════════════════════════════════════
// 5. ACTIVE NAV SECTION TRACKING
// ════════════════════════════════════════
function initNavTracking() {
  const sections = document.querySelectorAll('section[id]');
  const links = document.querySelectorAll('.nav-links a');

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        links.forEach(link => link.classList.remove('active'));
        const active = document.querySelector(`.nav-links a[href="#${entry.target.id}"]`);
        if (active) active.classList.add('active');
      }
    });
  }, { threshold: 0.35 });

  sections.forEach(s => observer.observe(s));
}

// ════════════════════════════════════════
// 6. LIGHTBOX
// ════════════════════════════════════════
function initLightbox() {
  const overlay = document.createElement('div');
  overlay.className = 'lightbox-overlay';
  const img = document.createElement('img');
  const closeBtn = document.createElement('div');
  closeBtn.className = 'lightbox-close';
  closeBtn.innerHTML = '✕';
  overlay.appendChild(img);
  document.body.appendChild(overlay);
  document.body.appendChild(closeBtn);

  const close = () => {
    overlay.classList.remove('active');
    closeBtn.style.display = 'none';
    document.body.style.overflow = '';
  };
  const open = (src, alt) => {
    img.src = src;
    img.alt = alt || '';
    overlay.classList.add('active');
    closeBtn.style.display = 'flex';
    document.body.style.overflow = 'hidden';
  };

  document.querySelectorAll('.plot-card img').forEach(el => {
    el.style.cursor = 'zoom-in';
    el.addEventListener('click', () => open(el.src, el.alt));
  });

  overlay.addEventListener('click', close);
  closeBtn.addEventListener('click', close);
  document.addEventListener('keydown', e => { if (e.key === 'Escape') close(); });
}

// ════════════════════════════════════════
// 7. NAVBAR SCROLL EFFECT
// ════════════════════════════════════════
function initNavbarScroll() {
  const nav = document.getElementById('navbar');
  let ticking = false;
  window.addEventListener('scroll', () => {
    if (!ticking) {
      requestAnimationFrame(() => {
        if (window.scrollY > 60) {
          nav.style.background = 'rgba(5, 8, 16, 0.97)';
          nav.style.boxShadow = '0 4px 32px rgba(0,0,0,0.4)';
        } else {
          nav.style.background = 'rgba(5, 8, 16, 0.85)';
          nav.style.boxShadow = 'none';
        }
        ticking = false;
      });
      ticking = true;
    }
  });
}

// ════════════════════════════════════════
// 8. ACTIVE NAV LINK STYLING
// ════════════════════════════════════════
function injectActiveNavStyle() {
  const style = document.createElement('style');
  style.textContent = `
    .nav-links a.active {
      color: var(--text-primary) !important;
      background: rgba(99,102,241,0.15) !important;
    }
  `;
  document.head.appendChild(style);
}

// ════════════════════════════════════════
// 9. STAGGERED ENTRY ANIMATIONS
// ════════════════════════════════════════
function initStaggeredAnimations() {
  // Pipeline steps stagger
  const steps = document.querySelectorAll('.pipeline-step');
  steps.forEach((step, i) => {
    step.style.transitionDelay = `${i * 80}ms`;
  });

  // Metric cards stagger
  const cards = document.querySelectorAll('.metric-big-card');
  cards.forEach((card, i) => {
    card.style.transitionDelay = `${i * 60}ms`;
  });

  // BPL items
  const bplItems = document.querySelectorAll('.bpl-item');
  bplItems.forEach((item, i) => {
    item.style.transitionDelay = `${i * 50}ms`;
  });
}

// ════════════════════════════════════════
// 10. CHART LAZY INIT (observe canvas elements)
// ════════════════════════════════════════
function initChartLazy() {
  const chartInits = {
    baselineChart: initBaselineChart,
    ensembleChart: initEnsembleChart,
    leaderboardChart: initLeaderboardChart,
  };
  const initialized = {};

  const obs = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const id = entry.target.id;
        if (chartInits[id] && !initialized[id]) {
          initialized[id] = true;
          chartInits[id]();
          obs.unobserve(entry.target);
        }
      }
    });
  }, { threshold: 0.1 });

  Object.keys(chartInits).forEach(id => {
    const el = document.getElementById(id);
    if (el) obs.observe(el);
  });
}

// ════════════════════════════════════════
// INIT ALL
// ════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
  injectActiveNavStyle();
  initNavbarScroll();
  initNavTracking();
  initScrollAnimations();
  initStaggeredAnimations();
  initLightbox();
  initChartLazy();

  // Smooth scroll for nav links
  document.querySelectorAll('a[href^="#"]').forEach(link => {
    link.addEventListener('click', e => {
      e.preventDefault();
      const target = document.querySelector(link.getAttribute('href'));
      if (target) {
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });

  console.log('%c⬡ BloodIQ Showcase Loaded', 'color: #6366f1; font-weight: bold; font-size: 14px;');
  console.log('%cCluster-Adaptive Blood Report Analyzer · BCSE209L · VIT Chennai', 'color: #94a3b8;');
});
