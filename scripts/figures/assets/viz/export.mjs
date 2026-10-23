/** Export vector downloads from a fixed-size snapshot, preserving the display mode. */
function download(blob, name) {
  const url = URL.createObjectURL(blob);
  const anchor = Object.assign(document.createElement('a'), { href: url, download: name });
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function exportSVG(fig) {
  const svg = fig.snapshot();
  download(new Blob([new XMLSerializer().serializeToString(svg)], { type: 'image/svg+xml' }), `${fig.name}.svg`);
}

let libraries;
function pdfLibraries() {
  libraries ??= (async () => {
    await import('./vendor/jspdf.umd.min.js');
    await import('./vendor/svg2pdf.umd.min.js');
    return window.jspdf.jsPDF;
  })();
  return libraries;
}

export async function exportPDF(fig) {
  const svg = fig.snapshot();
  const [, , width, height] = svg.getAttribute('viewBox').split(/\s+/).map(Number);
  const jsPDF = await pdfLibraries();
  const pdf = new jsPDF({ unit: 'pt', format: [width, height],
    orientation: width > height ? 'landscape' : 'portrait', compress: true, putOnlyUsedFonts: true });
  pdf.setProperties({ title: `${document.title} — ${fig.name}`, subject: svg.querySelector('desc')?.textContent ?? '' });
  for (const text of svg.querySelectorAll('text, tspan')) {
    text.setAttribute('font-family', 'helvetica');
    const weight = text.getAttribute('font-weight');
    if (weight) text.setAttribute('font-weight', weight === 'bold' || +weight >= 600 ? 'bold' : 'normal');
  }
  const nodes = document.createTreeWalker(svg, NodeFilter.SHOW_TEXT);
  while (nodes.nextNode()) nodes.currentNode.textContent = nodes.currentNode.textContent.replaceAll('−', '-');
  const host = document.createElement('div');
  host.setAttribute('aria-hidden', 'true');
  host.style.cssText = 'position:fixed;left:-10000px;top:0;pointer-events:none';
  host.append(svg);
  document.body.append(host);
  try {
    await pdf.svg(svg, { x: 0, y: 0, width, height });
    download(pdf.output('blob'), `${fig.name}.pdf`);
  } finally { host.remove(); }
}
