const check = (condition, message) => { if (!condition) throw Error(message); };
const select = document.querySelector('#experiment');
const frame = document.querySelector('#figure-page');
const pages = JSON.parse(document.querySelector('#gallery-data').textContent);
check(select.options.length === pages.length && pages.length > 0, 'Missing gallery options');
const waitForPage = async (index, previous = null) => {
  for (let attempt = 0; attempt < 100; attempt++) {
    const doc = frame.contentDocument;
    if (doc !== previous && doc?.documentElement.dataset.ready === 'true' && doc.querySelector('#figure-title')?.textContent === pages[index].label) return doc;
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  throw Error(`Figure ${index} did not load`);
};
for (let index = 0; index < pages.length; index++) {
  const previous = frame.contentDocument;
  select.value = String(index);
  select.dispatchEvent(new Event('change', { bubbles: true }));
  const doc = await waitForPage(index, previous);
  const embedded = doc.querySelector('#figure-data');
  const manifest = embedded ? JSON.parse(embedded.textContent).manifest : await (await fetch(new URL('data/manifest.json', frame.src))).json();
  check(doc.querySelectorAll('.fig > svg').length === (manifest.kind === 'runtime' ? 1 : manifest.time_column ? 2 : 1), 'Embedded figures did not render');
  check(frame.title === pages[index].label, 'Wrong iframe title');
  const direct = document.querySelector('#open-figure').href;
  check(embedded ? direct.startsWith('blob:') : direct === frame.src, 'Wrong direct link');
  check(new URL(location.href).searchParams.get('figure') === String(index), 'Selection not reflected in URL');
  if (window.figureChecks) {
    await new frame.contentWindow.Function(`return (async () => { ${window.figureChecks} })()`)();
    continue;
  }
  if (manifest.kind === 'runtime') {
    const gamma = +doc.querySelector('#runtime').dataset.gamma;
    const cells = new Set(manifest.runs.filter(r => r.gamma === gamma).map(r => JSON.stringify([r.environment, r.label])));
    check(doc.querySelectorAll('.runtime-point').length === cells.size, 'Runtime median points missing');
    continue;
  }
  const logx = doc.querySelector('#controls input[data-axis="x"]');
  logx.click();
  check(doc.querySelector('#samples').dataset.xScale === 'log', 'Embedded axis control failed');
  check(doc.querySelectorAll('.seed').length === 0, 'Individual traces reappeared');
  check(doc.querySelectorAll('.percentile-band').length > 0, 'Missing percentile bands');
  check(doc.querySelector('#theme input[value="focus"]').checked, 'Embedded default palette is not Focus');
  check(!doc.querySelector('.panel-letter'), 'Unexpected embedded panel lettering');
  check(doc.querySelectorAll('.percentiles input[type="range"]').length === 2, 'Missing percentile handles');
}
await new Promise(resolve => setTimeout(resolve, 200));
check(frame.clientHeight > 300, 'Embedded page did not resize');
