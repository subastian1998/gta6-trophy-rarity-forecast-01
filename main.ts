// gta6-trophy-rarity-forecast-01
// Deterministic Node 22 (stdlib only) forecaster for GTA VI PS5 trophy
// earn rates on Exophase at each case's target instant.
//
// Approach, per trophy:
//   * Fit an ordinary-least-squares line of logit(rate) against
//     ln(1 + hours since 2026-11-19T00:00:00Z) over the reads of the
//     last 24 hours before the checkpoint. With fewer than 3 usable
//     reads, fall back to the trophy's most recent rate.
//   * Shrink each trophy's slope halfway toward the median slope
//     across all trophies (partial pooling: launch-night curves are
//     noisy and mostly flat, so a shared trend keeps wild slopes in
//     check).
//   * Read the shrunk line at the target instant for the median.
//   * Spread on the logit scale combines a random-walk term that
//     grows with the horizon (0.5 * sqrt(horizon in days)) and the
//     residual scatter of the trophy's own fit, added in quadrature.
//   * Clamp every logit to the logit of [0.01, 99.99], invert, clamp
//     to [0, 100], round to 4 decimals and force non-decreasing
//     quantiles.
//
// No randomness anywhere, so two runs are byte-identical.

import { readdirSync, readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";

const LAUNCH_MS = Date.parse("2026-11-19T00:00:00Z");
const HOUR_MS = 3_600_000;
const DAY_MS = 86_400_000;
const LO = 0.01;
const HI = 99.99;
const LO_LOGIT = Math.log(LO / (100 - LO));
const HI_LOGIT = Math.log(HI / (100 - HI));
const Z = [-1.6448536269514722, -0.6744897501960817, 0, 0.6744897501960817, 1.6448536269514722];

function clampRate(r) {
  if (!Number.isFinite(r)) return null;
  if (r < LO) return LO;
  if (r > HI) return HI;
  return r;
}

function toLogit(pct) {
  const p = pct / 100;
  return Math.log(p / (1 - p));
}

function fromLogit(l) {
  return 100 / (1 + Math.exp(-l));
}

function hoursSinceLaunch(iso) {
  return (Date.parse(iso) - LAUNCH_MS) / HOUR_MS;
}

// Ordinary least squares for y = a + b*x. Returns null on degenerate x.
function leastSquares(xs, ys) {
  const n = xs.length;
  let sx = 0;
  let sy = 0;
  for (let i = 0; i < n; i++) {
    sx += xs[i];
    sy += ys[i];
  }
  const mx = sx / n;
  const my = sy / n;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - mx;
    sxx += dx * dx;
    sxy += dx * (ys[i] - my);
  }
  if (sxx <= 1e-12) return null;
  const b = sxy / sxx;
  return { intercept: my - b * mx, slope: b };
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 1) return sorted[mid];
  return (sorted[mid - 1] + sorted[mid]) / 2;
}

function round4(v) {
  return Math.round(v * 10000) / 10000;
}

function forecastCase(c) {
  const checkpointMs = Date.parse(c.at);
  const targetX = Math.log1p(hoursSinceLaunch(c.target));
  const horizonDays = (Date.parse(c.target) - checkpointMs) / DAY_MS;
  const walkSigma = 0.5 * Math.sqrt(Math.max(horizonDays, 0));
  const windowStartMs = checkpointMs - DAY_MS;

  const perTrophy = [];
  const slopes = [];
  const reads = Array.isArray(c.history) ? c.history : [];

  for (let t = 0; t < c.trophies.length; t++) {
    const xs = [];
    const ys = [];
    const recentRates = [];
    for (const read of reads) {
      const rates = Array.isArray(read.rates) ? read.rates : [];
      const rate = rates[t];
      const clamped = clampRate(rate);
      if (clamped !== null) recentRates.push(clamped);
      const readMs = Date.parse(read.at);
      if (readMs < windowStartMs || readMs > checkpointMs) continue;
      if (clamped === null) continue;
      xs.push(Math.log1p(hoursSinceLaunch(read.at)));
      ys.push(toLogit(clamped));
    }
    let fit = null;
    if (xs.length >= 3) {
      fit = leastSquares(xs, ys);
      if (fit) slopes.push(fit.slope);
    }
    // Fallback centre: median of the last three usable rates (robust to a
    // single noisy read) instead of only the very last one.
    const lastRate = recentRates.length > 0 ? median(recentRates.slice(-3)) : null;
    perTrophy.push({ trophyIndex: t, xs, ys, fit, lastRate });
  }

  const sharedSlope = slopes.length > 0 ? median(slopes) : 0;

  const rows = [];
  for (const item of perTrophy) {
    const { fit, lastRate } = item;
    let centreLogit;
    let residSigma = 0.3;

    if (fit) {
      // Shrink the trophy's own slope halfway to the shared slope.
      const shrunkSlope = 0.5 * fit.slope + 0.5 * sharedSlope;
      const shrunkIntercept = fit.intercept + (fit.slope - shrunkSlope) * (item.xs.reduce((a, b) => a + b, 0) / item.xs.length);
      centreLogit = shrunkIntercept + shrunkSlope * targetX;
      // Residual scatter of the trophy's own fit around its own line.
      let ss = 0;
      for (let i = 0; i < item.xs.length; i++) {
        const d = item.ys[i] - (fit.intercept + fit.slope * item.xs[i]);
        ss += d * d;
      }
      const dof = Math.max(item.xs.length - 2, 1);
      residSigma = Math.sqrt(ss / dof);
      // A perfect fit on flat launch-night data still deserves some spread.
      if (residSigma < 0.05) residSigma = 0.05;
    } else if (lastRate !== null) {
      centreLogit = toLogit(lastRate);
      residSigma = 0.3;
    } else {
      // No usable history at all: centre on 1% with a wide band.
      centreLogit = toLogit(1);
      residSigma = 1.0;
    }

    const sigma = Math.sqrt(walkSigma * walkSigma + residSigma * residSigma);
    const qs = Z.map((z) => {
      const l = Math.min(HI_LOGIT, Math.max(LO_LOGIT, centreLogit + z * sigma));
      return round4(Math.min(100, Math.max(0, fromLogit(l))));
    });
    // Force non-decreasing quantiles (rounding can only tie, never invert,
    // but guard anyway).
    for (let i = 1; i < qs.length; i++) {
      if (qs[i] < qs[i - 1]) qs[i] = qs[i - 1];
    }
    rows.push({ id: c.trophies[item.trophyIndex].id, q: qs });
  }

  rows.sort((a, b) => a.id - b.id);
  return { trophies: rows };
}

function main() {
  const inputDir = process.env.MAKEFUN_INPUT_DIR || "/input/";
  const outputDir = process.env.MAKEFUN_OUTPUT_DIR || "/output/";
  mkdirSync(outputDir, { recursive: true });
  const names = readdirSync(inputDir)
    .filter((n) => n.endsWith(".in"))
    .sort();
  for (const name of names) {
    const c = JSON.parse(readFileSync(join(inputDir, name), "utf8"));
    const out = forecastCase(c);
    const outName = name.replace(/\.in$/, ".out");
    writeFileSync(join(outputDir, outName), JSON.stringify(out) + "\n", "utf8");
  }
}

main();
