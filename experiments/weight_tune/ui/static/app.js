const state = {
  snapshots: [],
  mixes: [],
  mix: null,
  season: null,
  asOf: null,
  position: "",
  sort: "overall",
  selected: null,
  tab: "players",
  pred: "overall",
};

const $ = (id) => document.getElementById(id);

function badgeClass(action) {
  if (action === "STRONG BUY") return "strong";
  if (action === "BUY") return "buy";
  if (action === "HOLD") return "hold";
  return "avoid";
}
function fmt(n, d = 1) {
  return n == null || Number.isNaN(n) ? "—" : Number(n).toFixed(d);
}

async function loadMeta() {
  const meta = await (await fetch("/api/meta")).json();
  state.snapshots = meta.snapshots || [];
  state.mixes = meta.mixes || [];
  state.mix = meta.default_mix;
  const def = meta.default;
  state.season = def ? def.season : null;
  state.asOf = def ? def.as_of_gw : null;
  $("mix").innerHTML = state.mixes
    .map((m) => `<option value="${m}" ${m === state.mix ? "selected" : ""}>${m}</option>`)
    .join("");
  fillSeason();
  fillAsOf();
}

function fillSeason() {
  const seasons = [...new Set(state.snapshots.map((s) => s.season))];
  $("season").innerHTML = seasons
    .map((s) => `<option ${s === state.season ? "selected" : ""}>${s}</option>`)
    .join("");
}
function fillAsOf() {
  const gws = state.snapshots.filter((s) => s.season === state.season);
  $("as-of").innerHTML = gws
    .map(
      (s) =>
        `<option value="${s.as_of_gw}" ${Number(s.as_of_gw) === Number(state.asOf) ? "selected" : ""}>GW ${s.as_of_gw}</option>`
    )
    .join("");
}

function qs(extra = {}) {
  const p = new URLSearchParams({
    season: state.season || "",
    as_of: String(state.asOf || ""),
    mix: state.mix || "",
    ...extra,
  });
  return p.toString();
}

async function loadPlayers() {
  if (!state.season) return;
  const data = await (await fetch(`/api/players?${qs({ position: state.position, q: $("search").value, sort: state.sort })}`)).json();
  const players = data.players || [];
  $("empty").classList.toggle("hidden", players.length > 0);
  $("rows").innerHTML = players
    .map(
      (p, i) => `
      <div class="row ${state.selected === p.fpl_id ? "on" : ""}" data-id="${p.fpl_id}">
        <span class="c-rank">${i + 1}</span>
        <span class="c-name">${p.web_name || ""} <span style="color:var(--muted)">${p.team || ""} ${p.position || ""}</span></span>
        <span class="c-num">${fmt(p.overall)}</span>
        <span class="c-num">${fmt(p.upside)}</span>
        <span class="c-num">${fmt(p.xpts, 2)}</span>
        <span class="c-act"><span class="badge ${badgeClass(p.action)}">${p.action || ""}</span></span>
      </div>`
    )
    .join("");
  $("rows").querySelectorAll(".row").forEach((el) => {
    el.addEventListener("click", () => selectPlayer(Number(el.dataset.id)));
  });
}

function bar(label, value, cls = "") {
  const w = Math.max(0, Math.min(100, value || 0));
  return `<div class="bar-row"><label>${label}</label><div class="track"><div class="fill ${cls}" style="width:${w}%"></div></div><span class="n">${fmt(value)}</span></div>`;
}

async function selectPlayer(id) {
  state.selected = id;
  const p = await (await fetch(`/api/player?${qs({ id: String(id) })}`)).json();
  const parts = (p.card && p.card.upside_parts) || {};
  const st = p.stats || {};
  $("card").innerHTML = `
    <div class="hero">
      <div class="who">
        <h2>${p.web_name || ""}</h2>
        <p>${p.position || ""} · ${p.team_name || p.team || ""} · £${fmt(p.price)}</p>
        <p style="margin-top:.5rem"><span class="badge ${badgeClass(p.action)}">${p.action || ""}</span></p>
      </div>
      <div class="big">
        <div class="overall">${fmt(p.overall)}</div>
        <div class="upside">Upside ${fmt(p.upside)}</div>
        <div class="xpts">xPts ${fmt(p.xpts, 2)}</div>
      </div>
    </div>
    <div class="stats-grid">
      <div class="stat"><span>Pts</span><b>${fmt(st.points, 0)}</b></div>
      <div class="stat"><span>G</span><b>${fmt(st.goals, 0)}</b></div>
      <div class="stat"><span>A</span><b>${fmt(st.assists, 0)}</b></div>
      <div class="stat"><span>npxG/90</span><b>${fmt(st.npxg90, 2)}</b></div>
    </div>
    <div class="bars">
      ${bar("Quality Q", p.Q)}
      ${bar("Minutes M", p.M, "m")}
      ${bar("Fixtures O", p.O)}
      ${bar("Value V", p.V)}
      ${bar("Underlying U", p.U)}
      ${bar("FPL output F", p.F)}
      ${bar("Role R", p.R)}
      ${bar("Team T", p.T)}
      ${bar("Finishing", parts.finishing)}
      ${bar("Creation", parts.creation)}
    </div>`;
  loadPlayers();
}

function cellRank(rank, n, value, digits) {
  if (rank == null || value == null) return "—";
  const top = Number(rank) === 1 ? "rank-1" : "";
  return `<span class="${top}">${rank}/${n}</span> · ${fmt(value, digits)}`;
}

function renderRankBoard(board) {
  const n = board.n_mixes || (board.rows || []).length;
  const head = `
    <tr>
      <th>Mix</th>
      <th colspan="2">2024–25</th>
      <th colspan="2">2025–26</th>
    </tr>
    <tr>
      <th></th>
      <th class="num">Spearman</th>
      <th class="num">Top-20</th>
      <th class="num">Spearman</th>
      <th class="num">Top-20</th>
    </tr>`;
  const body = (board.rows || [])
    .map(
      (r) => `<tr>
        <td>${r.mix}</td>
        <td class="num">${cellRank(r.s2425_spearman_rank, n, r.s2425_spearman, 3)}</td>
        <td class="num">${cellRank(r.s2425_top20_rank, n, r.s2425_top20, 2)}</td>
        <td class="num">${cellRank(r.s2526_spearman_rank, n, r.s2526_spearman, 3)}</td>
        <td class="num">${cellRank(r.s2526_top20_rank, n, r.s2526_top20, 2)}</td>
      </tr>`
    )
    .join("");
  return `<div class="pos-board"><h3>${board.position} <span style="color:var(--muted);font-weight:400">rank / ${n} mixes</span></h3>
    <table>${head}${body}</table></div>`;
}

function renderCombined(rows, title, note) {
  if (!rows || !rows.length) return "";
  const n = rows[0].n_mixes || rows.length;
  const body = rows
    .map(
      (r) => `<tr>
        <td>${r.mix}</td>
        <td class="num ${Number(r.season_rank) === 1 ? "rank-1" : ""}">${r.season_rank}/${n}</td>
        <td class="num">${fmt(r.score, 4)}</td>
      </tr>`
    )
    .join("");
  return `<div class="pos-board combined-board">
    <h3>${title}</h3>
    <p class="rank-note">${note} Ranked 1/${n} on that score (higher is better).</p>
    <table>
      <tr><th>Mix</th><th class="num">Combined</th><th class="num">Score</th></tr>
      ${body}
    </table>
  </div>`;
}

function renderFinal(rows, title, note) {
  if (!rows || !rows.length) return "";
  const n = rows[0].n_mixes || rows.length;
  const body = rows
    .map(
      (r) => `<tr>
        <td>${r.mix}</td>
        <td class="num ${Number(r.final_rank) === 1 ? "rank-1" : ""}">${r.final_rank}/${n}</td>
        <td class="num">${fmt(r.final_u, 4)}</td>
        <td class="num">${fmt(r.u_2425, 4)}</td>
        <td class="num">${fmt(r.u_2526, 4)}</td>
      </tr>`
    )
    .join("");
  return `<div class="pos-board combined-board">
    <h3>${title}</h3>
    <p class="rank-note">${note} Ranked 1/${n} on that score (higher is better).</p>
    <table>
      <tr><th>Mix</th><th class="num">Final</th><th class="num">Weighted score</th><th class="num">2024–25</th><th class="num">2025–26</th></tr>
      ${body}
    </table>
  </div>`;
}

async function loadMetrics() {
  const m = await (await fetch("/api/metrics")).json();
  $("holdout").textContent = m.holdout || "";
  const xpts = state.pred === "xpts";
  const boards = xpts ? m.ranks_xpts : m.ranks_overall;
  const c24 = xpts ? m.combined_xpts_2425 : m.combined_overall_2425;
  const c25 = xpts ? m.combined_xpts_2526 : m.combined_overall_2526;
  const fin = xpts ? m.final_xpts : m.final_overall;
  const empty = !boards || !boards.length;
  $("rank-empty").classList.toggle("hidden", !empty);
  const rk = m.ranking || {};
  const w24 = rk.season_weights ? rk.season_weights["2024-25"] : 0.35;
  const w25 = rk.season_weights ? rk.season_weights["2025-26"] : 0.65;
  const spw = rk.spearman_weight ?? 0.7;
  const t20 = rk.top20_weight ?? 0.3;
  const seasonNote = `${spw}×Spearman + ${t20}×Top-20, mean over GK/DEF/MID/FWD. Same formula as Final, this season only.`;
  const finalNote = `${spw}×Spearman + ${t20}×Top-20, mean over positions, then ${w24}×2024–25 + ${w25}×2025–26. Only mixes that change this pill (Overall×Quality vs xPts recipes).`;
  $("rank-boards").innerHTML = empty
    ? ""
    : boards.map(renderRankBoard).join("") +
      renderCombined(c24, "2024–25 combined rank", seasonNote) +
      renderCombined(c25, "2025–26 combined rank", seasonNote) +
      renderFinal(fin, "Final (season-weighted)", finalNote);
}

async function loadXi() {
  $("xi-out").innerHTML = "<p class='empty'>Building…</p>";
  const data = await (
    await fetch(`/api/xi?${qs({ budget: $("xi-budget").value, objective: $("xi-obj").value })}`)
  ).json();
  if (!data.ok) {
    $("xi-out").innerHTML = `<p class="empty">${data.error || "Could not build XI"}</p>`;
    return;
  }
  const line = (p) =>
    `<div class="xi-line"><span>${p.position}</span><span>${p.web_name}</span><span>${p.team || ""}</span><span>£${fmt(p.price)}</span><span>${fmt(p.xpts, 2)} xPts</span></div>`;
  $("xi-out").innerHTML = `
    <p style="padding:1rem">${data.formation || ""} · spent £${fmt(data.spent)} · mix ${data.mix || ""}
    ${data.captain ? ` · C ${data.captain.web_name}` : ""}</p>
    ${(data.xi || []).map(line).join("")}
    <p style="padding:.5rem 1rem;color:var(--muted)">Bench</p>
    ${(data.bench || []).map(line).join("")}`;
}

function setTab(tab) {
  state.tab = tab;
  document.querySelectorAll(".tabs button").forEach((b) => b.classList.toggle("on", b.dataset.tab === tab));
  $("view-players").classList.toggle("hidden", tab !== "players");
  $("view-metrics").classList.toggle("hidden", tab !== "metrics");
  $("view-xi").classList.toggle("hidden", tab !== "xi");
  if (tab === "metrics") loadMetrics();
}

$("tabs").addEventListener("click", (e) => {
  const t = e.target.dataset.tab;
  if (t) setTab(t);
});
$("mix").addEventListener("change", () => {
  state.mix = $("mix").value;
  loadPlayers();
});
$("season").addEventListener("change", () => {
  state.season = $("season").value;
  fillAsOf();
  state.asOf = Number($("as-of").value);
  loadPlayers();
});
$("as-of").addEventListener("change", () => {
  state.asOf = Number($("as-of").value);
  loadPlayers();
});
$("sort").addEventListener("change", () => {
  state.sort = $("sort").value;
  loadPlayers();
});
$("search").addEventListener("input", () => loadPlayers());
$("pos-pills").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  state.position = b.dataset.pos || "";
  $("pos-pills").querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
  loadPlayers();
});
$("pred-pills").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b || !b.dataset.pred) return;
  state.pred = b.dataset.pred;
  $("pred-pills").querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
  loadMetrics();
});
$("xi-go").addEventListener("click", loadXi);

loadMeta().then(loadPlayers);
