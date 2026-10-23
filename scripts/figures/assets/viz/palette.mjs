/**
 * Palettes. Each one is six categorical series colors in a fixed order, a reference grey (always dashed),
 * and matching sequential and diverging colormaps. Items identify colors by slot (0-5, 'ref' or 'context').
 * 'context' is a neutral grey for background curves and secondary settings.
 * setPalette() writes --s1…--s6/--ref to :root, so HTML controls using var(--sN) follow automatically.
 * SVG marks receive concrete hex via recolor(), so exported files stay self-contained.
 * Facet or merge groups that exceed six categorical series.
 */
const d3 = window.d3;
const lab = stops => d3.piecewise(d3.interpolateLab, stops);

export const PALETTES = {
  // Print-safe: series differ in lightness, so they survive B/W printing. Name lines directly; the reference stays dashed.
  gray: {
    label: 'Grayscale',
    series: ['#767676', '#000000', '#454545', '#5e5e5e', '#262626', '#9a9a9a'], ref: '#a3a3a3',
    sequential: lab(['#fafafa', '#cfcfcf', '#969696', '#525252', '#141414']),
    // Dark shades indicate positive values, light shades negative values; zero contours are dashed.
    diverging: lab(['#ffffff', '#e0e0e0', '#b0b0b0', '#7a7a7a', '#454545', '#1a1a1a']),
  },
  // Okabe–Ito categorical (safe for protan/deutan/tritan), cividis sequential, and a blue–orange diverging map.
  colorblind: {
    label: 'Color-blind',
    series: ['#0072b2', '#d55e00', '#009e73', '#e69f00', '#cc79a7', '#56b4e9'], ref: '#6b7280',   // slot 1: vermilion
    sequential: d3.interpolateCividis,
    diverging: lab(['#00375a', '#0072b2', '#9ecae1', '#f7f7f7', '#f2b48a', '#d55e00', '#7a2e00']),
  },
  // Focus pairs orange (s2) with navy (s1); s3–s6 are muted comparison colors.
  // The diverging map runs from navy for negative values to orange for positive values.
  focus: {
    label: 'Focus',
    series: ['#1f3a5f', '#e5470c', '#7d8896', '#6f93ab', '#b59f7f', '#a28fb5'], ref: '#aab3bd',
    sequential: lab(['#fff5ee', '#ffcfb3', '#ff9a66', '#e5470c', '#7f2604']),
    diverging: lab(['#0f2340', '#1f3a5f', '#93a9c4', '#f6f6f4', '#ffb592', '#e5470c', '#7f2604']),
  },
  // Raspberry and pine highlights with muted comparison colors.
  raspberry: {
    label: 'Raspberry',
    series: ['#14453d', '#d1176b', '#879792', '#93abbd', '#bda88b', '#ae98bd'], ref: '#b2bcb9',
    sequential: lab(['#fff1f6', '#f9bdd3', '#ec6a9c', '#d1176b', '#6e0a37']),
    diverging: lab(['#0a2a25', '#14453d', '#8fb3aa', '#f6f6f4', '#f4a9c6', '#d1176b', '#6e0a37']),
  },
  // Cobalt and near-black highlights with warm grey comparisons.
  // The diverging map runs from grey for negative values to cobalt for positive values.
  cobalt: {
    label: 'Cobalt',
    series: ['#1c1c1c', '#2457ff', '#958d84', '#aab3c0', '#c4a98e', '#b3a6c4'], ref: '#bcb5ac',
    sequential: lab(['#f3f6ff', '#c3d1ff', '#7f9bff', '#2457ff', '#0c2a8a']),
    diverging: lab(['#1c1c1c', '#5c5a57', '#bdbab5', '#f6f6f4', '#aebfff', '#2457ff', '#0c2a8a']),
  },
};

let current = PALETTES.focus;
/** Live views of the active palette (mutated in place, so importers always see the current one). */
export const SERIES = [...current.series];
export const COLORMAPS = { sequential: current.sequential, diverging: current.diverging, viridis: d3.interpolateViridis };
export const CONTEXT = '#a8b1bd';                    // same as --context-light in ui.css; identical in every palette
export const colorOf = slot => (slot === 'ref' ? current.ref : slot === 'context' ? CONTEXT : current.series[slot]);
export const cssColor = slot => (slot === 'ref' ? 'var(--ref)' : slot === 'context' ? 'var(--context-light)' : `var(--s${slot + 1})`);
export const paletteName = () => Object.keys(PALETTES).find(k => PALETTES[k] === current);

export function setPalette(name) {
  current = PALETTES[name];
  const root = document.documentElement.style;
  current.series.forEach((c, i) => root.setProperty(`--s${i + 1}`, c));
  root.setProperty('--ref', current.ref);
  root.setProperty('--accent', current.series[1]);
  SERIES.splice(0, SERIES.length, ...current.series);
  Object.assign(COLORMAPS, { sequential: current.sequential, diverging: current.diverging });
}

/** Give every item with a slot its concrete color from the active palette. Pass every list the figures read. */
export function recolor(...lists) {
  for (const list of lists) for (const item of list) if (item.slot != null) item.color = colorOf(item.slot);
}

/** Diverging scales are symmetric about zero, so the neutral color means zero. */
export function divergingScale(values) {
  const m = d3.nice(0, d3.max(values, Math.abs) || 1, 5)[1];
  return d3.scaleDiverging(COLORMAPS.diverging).domain([-m, 0, m]);
}
export const sequentialScale = (values, map = COLORMAPS.sequential) => d3.scaleSequential(map).domain(d3.extent(values)).nice();
