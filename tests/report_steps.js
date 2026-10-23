const check = (condition, message) => { if (!condition) throw Error(message); };
const frame = document.querySelector('#figure-page');
const environment = document.querySelector('#environment'), discount = document.querySelector('#discount'), view = document.querySelector('#view');
check(document.documentElement.dataset.ready === 'true', 'Report loader did not finish');
check(!document.querySelector('#notice').textContent.trim(), 'Unexpected missing-data warning');
check(environment.options.length === 2, 'Environment dropdown missing');
const steps = await (await fetch('/tests/figure_steps.js')).text();
const wait = async previous => {
  for (let i = 0; i < 100; i++) {
    if (frame.contentDocument !== previous && frame.contentDocument?.documentElement.dataset.ready === 'true') return;
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  throw Error('Selected CSV figure did not load');
};
for (const option of environment.options) {
  for (const gamma of discount.options) {
    const previous = frame.contentDocument;
    environment.value = option.value; discount.value = gamma.value;
    environment.dispatchEvent(new Event('change'));
    await wait(previous);
    const embedded = JSON.parse(frame.contentDocument.querySelector('#figure-data').textContent);
    check(new Set(embedded.manifest.runs.map(r => r.seed)).size === 11, 'Wrong seed count');
    check(embedded.manifest.runs.every(r => r.gamma === +gamma.value), 'Discount filter failed');
    await new frame.contentWindow.Function(`return (async () => { ${steps} })()`)();
  }
}
const previous = frame.contentDocument;
view.value = 'runtime'; view.dispatchEvent(new Event('change'));
await wait(previous);
check(environment.disabled && discount.disabled, 'Runtime should retain both rows and its own discount selector');
await new frame.contentWindow.Function(`return (async () => { ${steps} })()`)();
check(frame.clientHeight > 300, 'Figure page did not resize');
