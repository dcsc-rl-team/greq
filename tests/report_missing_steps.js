const check = (condition, message) => { if (!condition) throw Error(message); };
check(document.documentElement.dataset.ready === 'true', 'Missing-data page did not finish');
const message = document.querySelector('#notice').textContent;
for (const name of ['tuning.csv', 'evaluation.csv', 'runtime.csv']) check(message.includes(name), `Missing ${name} warning`);
check(message.includes('python -m scripts.finetune'), 'Missing tuning command');
check(message.includes('python -m scripts.report'), 'Missing evaluation command');
check(message.includes('--runtime'), 'Missing runtime command');
check(message.includes('Expected:'), 'Missing expected data path');
check(document.querySelector('#figure-page').hidden, 'Missing data rendered an empty figure');
