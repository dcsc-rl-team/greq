/** Two native, keyboard-accessible handles on one 0–100 percentile track. */
export function percentiles(container, { lower = 25, upper = 75, onChange }) {
  const field = document.createElement('fieldset');
  field.className = 'percentiles';
  field.innerHTML = `<legend>Percentile band <output aria-live="polite"></output></legend>
    <div class="percentile-slider">
      <div class="percentile-rail"></div>
      <input class="lower" type="range" min="0" max="50" step="1" aria-label="Lower percentile">
      <input class="upper" type="range" min="50" max="100" step="1" aria-label="Upper percentile">
    </div>
    <div class="percentile-scale" aria-hidden="true"><span>0%</span><span>50%</span><span>100%</span></div>`;
  container.append(field);
  const lo = field.querySelector('.lower'), hi = field.querySelector('.upper');
  const output = field.querySelector('output');
  const set = (lower, upper, emit = true) => {
    if (!Number.isFinite(lower) || !Number.isFinite(upper)) throw Error('Invalid percentiles');
    lower = Math.max(0, Math.min(50, Math.round(lower)));
    upper = Math.max(50, Math.min(100, Math.round(upper)));
    lo.value = lower; hi.value = upper;
    lo.setAttribute('aria-valuetext', `${lower}th percentile`);
    hi.setAttribute('aria-valuetext', `${upper}th percentile`);
    output.textContent = `${lower}–${upper}%`;
    field.style.setProperty('--lower', `${lower}%`);
    field.style.setProperty('--upper', `${upper}%`);
    field.classList.toggle('collapsed', lower === 50 && upper === 50);
    if (emit) onChange(lower, upper);
  };
  lo.addEventListener('input', () => set(+lo.value, +hi.value));
  hi.addEventListener('input', () => set(+lo.value, +hi.value));
  field.querySelector('.percentile-slider').addEventListener('pointerdown', event => {
    if (event.target instanceof HTMLInputElement) return;
    const bounds = field.querySelector('.percentile-rail').getBoundingClientRect();
    const value = Math.round(100 * (event.clientX - bounds.left) / bounds.width);
    const input = value <= 50 ? lo : hi;
    input.focus();
    set(input === lo ? value : +lo.value, input === hi ? value : +hi.value);
  });
  set(lower, upper, false);
  return { set };
}
