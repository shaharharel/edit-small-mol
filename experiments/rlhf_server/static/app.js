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
const POCKET_RADIUS = 5;          // Å, pocket residues / surface around ligand
const CYS_RESI = 346;             // catalytic cysteine in the anchor construct
let viewer = null;                // ONE persistent viewer, bound to #glhost (never wiped)
let curSide = "left";             // which molecule is loaded
let curConf = {};
const modalBack = el("modalBack");
const loadingEl = el("poseLoading");

async function getPose(molId) {
  if (poseCache[molId]) return poseCache[molId];
  const r = await fetch(`/api/pose/${molId}`);
  const d = await r.json();
  poseCache[molId] = d;
  return d;
}

function ensureViewer() {
  if (!viewer) {
    viewer = $3Dmol.createViewer(el("glhost"), { backgroundColor: "#0e0d0a" });
  }
  return viewer;
}

// Report-style styling: cartoon + pocket sticks + ligand + Cys346 + surface + H-bonds.
function applyPoseStyles() {
  if (!viewer) return;
  const cartoon = el("pc-cartoon").checked;
  const pocket = el("pc-pocket").checked;
  const surface = el("pc-surface").checked;
  const cys = el("pc-cys").checked;
  const hbonds = el("pc-hbonds").checked;

  viewer.setStyle({}, {});
  viewer.removeAllShapes();
  viewer.removeAllLabels();

  if (cartoon) viewer.setStyle({ chain: "A" }, { cartoon: { color: "spectrum", opacity: 0.78 } });
  if (pocket)
    viewer.addStyle(
      { chain: "A", byres: true, within: { distance: POCKET_RADIUS, sel: { chain: "B" } } },
      { stick: { radius: 0.12, colorscheme: "whiteCarbon" } }
    );
  // ligand
  viewer.setStyle({ chain: "B" }, { stick: { colorscheme: "magentaCarbon", radius: 0.2 } });
  // catalytic cysteine landmark
  if (cys) {
    viewer.addStyle(
      { chain: "A", resi: CYS_RESI },
      { stick: { colorscheme: "yellowCarbon", radius: 0.26 } }
    );
    viewer.addStyle({ chain: "A", resi: CYS_RESI, atom: "SG" }, { sphere: { radius: 0.5 } });
    viewer.addLabel(
      "Cys" + CYS_RESI,
      { backgroundColor: "rgba(242,192,55,0.9)", fontColor: "#1c1b17", fontSize: 11, inFront: true },
      { chain: "A", resi: CYS_RESI, atom: "CA" }
    );
  }
  // pocket surface (semi-transparent, pocket-local for speed)
  viewer.removeAllSurfaces();
  if (surface) {
    viewer.addSurface(
      $3Dmol.SurfaceType.VDW,
      { opacity: 0.4, color: "#9fb6c2" },
      { chain: "A", byres: true, within: { distance: POCKET_RADIUS, sel: { chain: "B" } } }
    );
  }
  // H-bonds: protein N/O <-> ligand N/O within 2.5–3.5 Å
  let nHb = 0;
  if (hbonds) {
    const m = viewer.getModel();
    const prot = m.selectedAtoms({ chain: "A", elem: ["N", "O"] });
    const lig = m.selectedAtoms({ chain: "B", elem: ["N", "O"] });
    for (const a of prot) {
      for (const b of lig) {
        const dd = Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
        if (dd > 2.5 && dd < 3.5) {
          nHb++;
          viewer.addCylinder({
            start: { x: a.x, y: a.y, z: a.z }, end: { x: b.x, y: b.y, z: b.z },
            radius: 0.05, dashed: true, fromCap: 1, toCap: 1, color: "#e8a33d",
          });
        }
      }
    }
  }
  viewer.render();
  const c = curConf || {};
  el("confLine").textContent =
    `Boltz-2 · ligand-ipTM ${(c.ligand_iptm || 0).toFixed(2)} · pLDDT ${(c.complex_plddt || 0).toFixed(2)}` +
    (hbonds ? ` · ${nHb} H-bond${nHb === 1 ? "" : "s"}` : "") +
    ` — predicted, unvalidated`;
}

async function renderPose(side) {
  if (!pair) return;
  curSide = side;
  const mol = side === "left" ? pair.left : pair.right;
  el("tabL").classList.toggle("active", side === "left");
  el("tabR").classList.toggle("active", side === "right");
  loadingEl.textContent = "loading pose…";
  loadingEl.classList.remove("hidden");

  const d = await getPose(mol.id);
  ensureViewer();
  viewer.clear();
  viewer.removeAllSurfaces();
  if (!d.available) {
    loadingEl.textContent = "3D pose not available";
    viewer.render();
    return;
  }
  curConf = d.conf || {};
  viewer.addModel(d.cif, "cif");
  viewer.resize();
  applyPoseStyles();
  viewer.zoomTo({ chain: "B" });
  viewer.zoom(0.55);
  viewer.render();
  loadingEl.classList.add("hidden");
}

function openModal(side) {
  if (!pair) return;
  modalBack.classList.add("open");
  // viewer must be created/resized while the modal is visible (correct sizing)
  requestAnimationFrame(() => renderPose(side));
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
["pc-cartoon", "pc-pocket", "pc-surface", "pc-cys", "pc-hbonds"].forEach(id =>
  el(id).addEventListener("change", applyPoseStyles));
el("pc-reset").addEventListener("click", () => {
  if (!viewer) return;
  viewer.zoomTo({ chain: "B" }); viewer.zoom(0.55); viewer.render();
});
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
