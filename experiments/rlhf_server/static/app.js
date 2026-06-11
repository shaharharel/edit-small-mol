/* Edit · RLHF — comparison flow
   Fetch a pair, render 2D structures, let the chemist pick (click / arrow keys),
   confirm, and advance. 3D Boltz pose in a modal with A/B tabs. */

const el = (id) => document.getElementById(id);
const stage = el("stage");
const cardL = el("cardL"), cardR = el("cardR");
const submitBtn = el("submitBtn"), skipBtn = el("skipBtn");

let pair = null;          // current pair payload
let selected = null;      // 'left' | 'right'
let busy = false;
const poseCache = {};     // mol_id -> {cif, conf}

// ---- rendering -------------------------------------------------------------
function confChip(node, conf) {
  const li = conf && conf.ligand_iptm != null ? conf.ligand_iptm : null;
  const span = node.querySelector("span:last-child");
  if (li == null) { node.classList.add("warn"); span.textContent = "pose n/a"; return; }
  node.classList.toggle("warn", li < 0.85);
  span.textContent = `pose conf ${li.toFixed(2)}`;
  node.title = `Boltz ligand-ipTM ${li.toFixed(2)} · pLDDT ${(conf.complex_plddt||0).toFixed(2)} — predicted, unvalidated`;
}

async function loadSvg(boxId, molId) {
  const box = el(boxId);
  box.classList.add("loading"); box.textContent = "drawing…";
  try {
    const r = await fetch(`/api/svg/${molId}?w=380&h=300`);
    box.innerHTML = await r.text();
    box.classList.remove("loading");
  } catch { box.textContent = "could not render"; }
}

function renderPair(p) {
  pair = p; selected = null; busy = false;
  el("cohort-id").textContent = (p.cohort || "—").replace("CHEMBL", "CHEMBL ");
  el("idL").textContent = p.left.chembl_id;
  el("idR").textContent = p.right.chembl_id;
  el("tabIdL").textContent = p.left.chembl_id;
  el("tabIdR").textContent = p.right.chembl_id;
  confChip(el("confL"), p.left.conf);
  confChip(el("confR"), p.right.conf);
  [cardL, cardR].forEach(c => c.classList.remove("selected", "dimmed"));
  submitBtn.disabled = true;
  el("pjudged").textContent = p.judged;
  el("ptotal").textContent = p.total;
  el("pfill").style.width = `${(p.judged / p.total) * 100}%`;
  loadSvg("molL", p.left.id);
  loadSvg("molR", p.right.id);
}

function showDone(n) {
  el("app").style.display = "none";
  el("doneCount").textContent = n;
  el("done").style.display = "block";
}

// ---- flow ------------------------------------------------------------------
async function nextPair() {
  stage.classList.add("swap");
  const r = await fetch("/api/next_pair");
  const data = await r.json();
  if (data.done) { showDone(data.judged); return; }
  // brief swap animation
  setTimeout(() => { renderPair(data); stage.classList.remove("swap"); }, 180);
}

function select(side) {
  if (busy || !pair) return;
  selected = side;
  cardL.classList.toggle("selected", side === "left");
  cardR.classList.toggle("selected", side === "right");
  cardL.classList.toggle("dimmed", side === "right");
  cardR.classList.toggle("dimmed", side === "left");
  submitBtn.disabled = false;
}

async function submit() {
  if (busy || !pair || !selected) return;
  busy = true;
  const chosen = selected === "left" ? pair.left.id : pair.right.id;
  await fetch("/api/submit", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ pair_id: pair.pair_id, chosen_id: chosen, action: "choice" }),
  });
  nextPair();
}

async function skip() {
  if (busy || !pair) return;
  busy = true;
  await fetch("/api/submit", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ pair_id: pair.pair_id, chosen_id: null, action: "skip" }),
  });
  nextPair();
}

// ---- 3D modal --------------------------------------------------------------
let viewer = null;
const modalBack = el("modalBack");

async function getPose(molId) {
  if (poseCache[molId]) return poseCache[molId];
  const r = await fetch(`/api/pose/${molId}`);
  const d = await r.json();
  poseCache[molId] = d;
  return d;
}

async function renderPose(side) {
  const mol = side === "left" ? pair.left : pair.right;
  el("tabL").classList.toggle("active", side === "left");
  el("tabR").classList.toggle("active", side === "right");
  const host = el("viewer3d");
  host.innerHTML = '<div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:#8a857a;font-family:JetBrains Mono,monospace;font-size:12px">loading pose…</div>';
  const d = await getPose(mol.id);
  if (!d.available) {
    host.innerHTML = '<div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:#8a857a;font-family:JetBrains Mono,monospace;font-size:12px">3D pose not available</div>';
    return;
  }
  host.innerHTML = "";
  if (!viewer) viewer = $3Dmol.createViewer(host, { backgroundColor: "#0e0d0a" });
  viewer.clear();
  viewer.addModel(d.cif, "cif");
  // protein cartoon, ligand sticks (hetero / non-polymer)
  viewer.setStyle({}, { cartoon: { color: "#7aa6c2", opacity: 0.9 } });
  viewer.setStyle({ hetflag: true }, { stick: { colorscheme: "default", radius: 0.18 } });
  viewer.addStyle({ hetflag: true }, { sphere: { scale: 0.22 } });
  // tint ligand carbons vermilion-ish by element default; emphasize via outline
  viewer.zoomTo({ hetflag: true });
  viewer.zoom(0.55);
  viewer.render();
  const c = d.conf || {};
  el("confLine").textContent =
    `Boltz-2 · ligand-ipTM ${(c.ligand_iptm||0).toFixed(2)} · pLDDT ${(c.complex_plddt||0).toFixed(2)} — predicted, experimentally unvalidated`;
}

function openModal(side) {
  if (!pair) return;
  modalBack.classList.add("open");
  renderPose(side);
}
function closeModal() { modalBack.classList.remove("open"); }

// ---- wiring ----------------------------------------------------------------
cardL.addEventListener("click", (e) => { if (!e.target.closest(".btn3d")) select("left"); });
cardR.addEventListener("click", (e) => { if (!e.target.closest(".btn3d")) select("right"); });
cardL.addEventListener("dblclick", () => { select("left"); submit(); });
cardR.addEventListener("dblclick", () => { select("right"); submit(); });
document.querySelectorAll(".btn3d").forEach(b =>
  b.addEventListener("click", (e) => { e.stopPropagation(); openModal(b.dataset.side); }));
el("tabL").addEventListener("click", () => renderPose("left"));
el("tabR").addEventListener("click", () => renderPose("right"));
el("modalX").addEventListener("click", closeModal);
modalBack.addEventListener("click", (e) => { if (e.target === modalBack) closeModal(); });
submitBtn.addEventListener("click", submit);
skipBtn.addEventListener("click", skip);

document.addEventListener("keydown", (e) => {
  if (modalBack.classList.contains("open")) {
    if (e.key === "Escape") closeModal();
    return;
  }
  if (e.key === "ArrowLeft") { select("left"); }
  else if (e.key === "ArrowRight") { select("right"); }
  else if (e.key === "Enter") { submit(); }
  else if (e.key === "3") { openModal(selected || "left"); }
  else if (e.key.toLowerCase() === "s") { skip(); }
});

nextPair();
