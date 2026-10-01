import { SceneManager } from './SceneManager.js';
import { RobotManager } from './RobotManager.js';
import { Player } from './Player.js';

const $ = (id) => document.getElementById(id);
const store = { get: (k, d) => { try { return localStorage.getItem(k) ?? d; } catch { return d; } },
                set: (k, v) => { try { localStorage.setItem(k, v); } catch {} } };

let player, current = null, sample = 0;

function status(msg, kind = '') { const s = $('status'); s.textContent = msg; s.className = kind; }

function show(result, label = '') {
    current = result; sample = 0;
    $('idea').textContent = result.idea || ''; $('idea').parentElement.hidden = !result.idea;
    $('recipe').textContent = result.recipe || '—';
    const t = result.timing_ms;
    $('timing').textContent = t ? `${result.effort ? result.effort + ' effort · ' : ''}planner ${t.planner} ms · generator ${t.generator} ms · total ${t.total} ms` : (label || '');
    const tabs = $('samples'); tabs.innerHTML = '';
    result.moves.forEach((_, i) => {
        const b = document.createElement('button'); b.textContent = `${i + 1}`; b.className = i === 0 ? 'on' : '';
        b.onclick = () => { sample = i; [...tabs.children].forEach((c, j) => c.className = j === i ? 'on' : ''); play(); };
        tabs.appendChild(b);
    });
    $('result').hidden = false;
    play();
}

function play() {
    const d = player.load(current.moves[sample]);
    $('dur').textContent = `${d.toFixed(1)} s`;
    $('playbtn').textContent = '❚❚';
}

async function generate() {
    const url = $('endpoint').value.trim().replace(/\/$/, ''), prompt = $('prompt').value.trim();
    if (!prompt) return status('Type a prompt first.', 'err');
    if (!url) return status('Set the endpoint URL, or pick an example below.', 'err');
    store.set('endpoint', url);
    status('Generating…'); $('go').disabled = true;
    try {
        const r = await fetch(`${url}/generate-dense`, { method: 'POST', headers: { 'content-type': 'application/json' },
            body: JSON.stringify({ prompt, n: +$('n').value, effort: $('effort').value, seed: Math.floor(Math.random() * 1e6) }) });
        if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
        show(await r.json()); status(`${prompt}`, 'ok');
    } catch (e) {
        const hint = location.protocol === 'https:' && url.startsWith('http:')
            ? ' (this page is HTTPS, so the browser blocks plain-HTTP endpoints: serve the endpoint over HTTPS)' : '';
        status(`Request failed: ${e.message}${hint}`, 'err');
    } finally { $('go').disabled = false; }
}

/** Accept an endpoint response, a single move, or a list of moves. */
function fromJSON(obj, name) {
    if (obj.moves) return obj;
    if (obj.set_target_data) return { prompt: obj.description || name, moves: [obj] };
    if (Array.isArray(obj) && obj[0]?.set_target_data) return { prompt: name, moves: obj };
    throw new Error('not a Reachy move or /generate-dense response');
}

async function loadExamples() {
    try {
        const ex = await (await fetch('examples/examples.json')).json();
        const box = $('examples');
        ex.forEach((e) => {
            const b = document.createElement('button'); b.textContent = e.prompt.split('.')[0];
            b.title = e.prompt;
            b.onclick = () => { $('prompt').value = e.prompt; show(e, 'pre-generated example'); status(e.prompt, 'ok'); };
            box.appendChild(b);
        });
        if (ex.length) { $('prompt').value = ex[0].prompt; show(ex[0], 'pre-generated example'); status(ex[0].prompt, 'ok'); }
    } catch (e) { console.warn('no examples', e); }
}

async function main() {
    const scene = new SceneManager($('container'));
    const robot = new RobotManager((m) => status(m), scene.envMap);
    const robotObj = await robot.loadRobot(); scene.add(robotObj); window.__scene = scene; window.__robot = robotObj;
    scene.animate?.();
    player = window.__player = new Player(robot, (e, d) => { $('scrub').value = d ? (e / d) * 1000 : 0; $('clock').textContent = `${e.toFixed(1)}`; });

    const params = new URLSearchParams(location.search);
    $('endpoint').value = params.get('endpoint') || store.get('endpoint', '');
    $('effort').value = params.get('effort') || store.get('effort', 'high');
    $('effort').onchange = (e) => store.set('effort', e.target.value);
    $('go').onclick = generate;
    $('prompt').addEventListener('keydown', (e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) generate(); });
    $('playbtn').onclick = () => { player.toggle(); $('playbtn').textContent = player.playing() ? '❚❚' : '▶'; };
    $('restart').onclick = () => player.restart();
    $('speed').onchange = (e) => player.setSpeed(+e.target.value);
    $('loop').onchange = (e) => { player.loop = e.target.checked; };
    $('scrub').oninput = (e) => player.seek(e.target.value / 1000);
    $('download').onclick = () => {
        if (!current) return;
        const a = document.createElement('a');
        a.href = URL.createObjectURL(new Blob([JSON.stringify(current.moves[sample])], { type: 'application/json' }));
        a.download = `${(current.prompt || 'move').split('.')[0].replace(/\W+/g, '_')}_${sample + 1}.json`; a.click();
    };
    // drag & drop (or pick) a move JSON / endpoint response
    const openFile = async (f) => { try { show(fromJSON(JSON.parse(await f.text()), f.name)); status(f.name, 'ok'); } catch (e) { status(`${f.name}: ${e.message}`, 'err'); } };
    $('file').onchange = (e) => e.target.files[0] && openFile(e.target.files[0]);
    window.addEventListener('dragover', (e) => e.preventDefault());
    window.addEventListener('drop', (e) => { e.preventDefault(); e.dataTransfer.files[0] && openFile(e.dataTransfer.files[0]); });

    await loadExamples();
}

main();
