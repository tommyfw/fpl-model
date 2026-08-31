const state = {
  snapshots: [],
  season: null,
  asOf: null,
  position: "",
  action: "",
  sort: "overall",
  selected: null,
  tab: "players",
  shortlist: new Set(),
  myTeam: [],
  lastXi: null,
  cache: new Map(),
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

function shortKey() {
  return `fpl-shortlist:${state.season || ""}`;
}
function teamKey() {
  return `fpl-myteam:${state.season || ""}`;
}

function loadShortlist() {
  try {
    state.shortlist = new Set(JSON.parse(localStorage.getItem(shortKey()) || "[]").map(Number));
  } catch {
    state.shortlist = new Set();
  }
}
function saveShortlist() {
  localStorage.setItem(shortKey(), JSON.stringify([...state.shortlist]));
}

function loadMyTeam() {
  try {
    const raw = JSON.parse(localStorage.getItem(teamKey()) || "{}");
    state.myTeam = (raw.ids || []).map(Number);
    if (raw.bank != null) $("team-bank").value = raw.bank;
    if (raw.ft != null) $("team-ft").value = raw.ft;
  } catch {
    state.myTeam = [];
  }
}
function saveMyTeam() {
  localStorage.setItem(
    teamKey(),
    JSON.stringify({
      ids: state.myTeam,
      bank: Number($("team-bank").value || 0),
      ft: Number($("team-ft").value || 1),
    })
  );
}

function toggleShort(id, ev) {
  if (ev) ev.stopPropagation();
  id = Number(id);
  if (state.shortlist.has(id)) state.shortlist.delete(id);
  else state.shortlist.add(id);
  saveShortlist();
  if (state.tab === "shortlist") renderShortlist();
  else if (state.tab === "players") loadPlayers();
}

function toggleTeamMember(id, ev) {
  if (ev) ev.stopPropagation();
  id = Number(id);
  const idx = state.myTeam.indexOf(id);
  if (idx >= 0) state.myTeam.splice(idx, 1);
  else if (state.myTeam.length < 15) state.myTeam.push(id);
  saveMyTeam();
  if (state.selected === id && state.tab === "players") selectPlayer(id);
  if (state.tab === "team") renderMyTeam();
}

async function loadMeta() {
  const meta = await (await fetch("/api/meta")).json();
  state.snapshots = meta.snapshots || [];
  const def = meta.default;
  state.season = def ? def.season : null;
  state.asOf = def ? def.as_of_gw : null;
  $("horizon-label").textContent = `Next ${meta.horizon_gws} GWs · as-of scores`;
  fillSeason();
  fillAsOf();
  loadShortlist();
  loadMyTeam();
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
        `<option value="${s.as_of_gw}" ${s.as_of_gw === state.asOf ? "selected" : ""}>GW ${s.as_of_gw} (${s.n})</option>`
    )
    .join("");
  if (gws.length && !gws.some((s) => s.as_of_gw === state.asOf)) {
    state.asOf = gws[0].as_of_gw;
    $("as-of").value = String(state.asOf);
  }
}

function rowHtml(p, i) {
  const starred = state.shortlist.has(p.fpl_id);
  return `
    <div class="row ${state.selected === p.fpl_id ? "sel" : ""}" data-id="${p.fpl_id}">
      <span class="c-star"><button type="button" class="${starred ? "on" : ""}" data-star="${p.fpl_id}" aria-label="Shortlist">${starred ? "★" : "☆"}</button></span>
      <span class="c-rank">${i + 1}</span>
      <span class="c-name">
        <div class="name">${p.web_name}</div>
        <div class="meta">${p.position} · ${p.team || "—"} · £${fmt(p.price)}</div>
      </span>
      <span class="c-num">${fmt(p.overall)}</span>
      <span class="c-num">${fmt(p.upside)}</span>
      <span class="c-num">${fmt(p.xpts, 1)}</span>
      <span class="c-act"><span class="badge ${badgeClass(p.action)}">${p.action}</span></span>
    </div>`;
}

async function fetchPlayers(extra = {}) {
  if (!state.season) return [];
  const params = new URLSearchParams({
    season: state.season,
    as_of: String(state.asOf),
    sort: extra.sort || state.sort,
  });
  if (extra.position || state.position) params.set("position", extra.position || state.position);
  if (extra.action || state.action) params.set("action", extra.action || state.action);
  if (extra.q) params.set("q", extra.q);
  if (extra.limit) params.set("limit", String(extra.limit));
  const data = await (await fetch(`/api/players?${params}`)).json();
  const players = data.players || [];
  players.forEach((p) => state.cache.set(p.fpl_id, p));
  return players;
}

async function loadPlayers() {
  if (!state.season) {
    $("empty").classList.remove("hidden");
    $("rows").innerHTML = "";
    return;
  }
  const q = $("search").value.trim();
  const players = await fetchPlayers({ q });
  $("empty").classList.toggle("hidden", players.length > 0);
  $("rows").innerHTML = players.map((p, i) => rowHtml(p, i)).join("");
  if (players.length && (state.selected == null || !players.some((p) => p.fpl_id === state.selected))) {
    await selectPlayer(players[0].fpl_id);
  } else if (state.selected) {
    highlight();
  }
}

async function renderShortlist() {
  const ids = [...state.shortlist];
  if (!ids.length) {
    $("short-rows").innerHTML = "";
    $("short-empty").classList.remove("hidden");
    return;
  }
  $("short-empty").classList.add("hidden");
  const all = await fetchPlayers({ sort: "overall" });
  const byId = new Map(all.map((p) => [p.fpl_id, p]));
  for (const id of ids.filter((x) => !byId.has(x))) {
    const p = await (await fetch(`/api/player?${new URLSearchParams({ season: state.season, as_of: String(state.asOf), id: String(id) })}`)).json();
    if (p && p.fpl_id) byId.set(p.fpl_id, p);
  }
  const players = ids.map((id) => byId.get(id)).filter(Boolean).sort((a, b) => (b.overall || 0) - (a.overall || 0));
  $("short-rows").innerHTML = players.map((p, i) => rowHtml(p, i)).join("");
}

function highlight() {
  document.querySelectorAll(".row").forEach((el) => {
    el.classList.toggle("sel", Number(el.dataset.id) === state.selected);
  });
}

async function selectPlayer(id) {
  state.selected = id;
  highlight();
  if (state.tab !== "players" && state.tab !== "shortlist") {
    setTab("players");
  }
  const params = new URLSearchParams({
    season: state.season,
    as_of: String(state.asOf),
    id: String(id),
  });
  const p = await (await fetch(`/api/player?${params}`)).json();
  renderCard(p);
}

function bar(label, value, cls = "") {
  const w = Math.max(0, Math.min(100, value || 0));
  return `<div class="bar-row"><label>${label}</label><div class="track"><div class="fill ${cls}" style="width:${w}%"></div></div><span class="n">${fmt(value)}</span></div>`;
}

function upTone(v) {
  if (v == null) return "";
  if (v > 52) return "plus";
  if (v < 48) return "minus";
  return "";
}

function easeClass(v, values) {
  if (v == null || !values.length) return "mid";
  const sorted = [...values].sort((a, b) => a - b);
  const lo = sorted[Math.floor(sorted.length * 0.33)];
  const hi = sorted[Math.floor(sorted.length * 0.66)];
  if (v >= hi) return "easy";
  if (v <= lo) return "hard";
  return "mid";
}

function fixtureChips(fixtures, mode = "attack") {
  if (!fixtures || !fixtures.length) return "";
  const key = mode === "def" ? "def_ease" : "attack_ease";
  const vals = fixtures.filter((f) => !f.blank).map((f) => f[key]);
  return `<div class="fx-row">${fixtures
    .map((f) => {
      const cls = f.blank ? "blank" : easeClass(f[key], vals);
      return `<div class="fx-chip ${cls}" title="${f.opp_name || f.opp}"><span class="gw">GW${f.event}</span>${f.label}</div>`;
    })
    .join("")}</div>`;
}

function statCell(label, value, d = 0) {
  return `<div class="stat"><span>${label}</span><b>${fmt(value, d)}</b></div>`;
}

function renderCard(p) {
  const full = [p.first_name, p.second_name].filter(Boolean).join(" ");
  const own = p.selected_by_percent != null ? `${fmt(p.selected_by_percent)}% owned` : "";
  const parts = (p.card && p.card.upside_parts) || {};
  const st = p.stats || {};
  const upRows = [
    ["Finishing", parts.finishing],
    ["Creation", parts.creation],
    ["Minutes trend", parts.minutes_trend],
    ["Team trend", parts.team_trend],
    ["Role change", parts.role_change],
    ["Price", parts.price],
    ["Ownership", parts.ownership],
  ];
  const starred = state.shortlist.has(p.fpl_id);
  const inTeam = state.myTeam.includes(p.fpl_id);
  const isDef = p.position === "DEF" || p.position === "GK"; // fixture chip ease mode
  const xgSrc = st.xg_source === "understat" ? "Understat" : "FPL";
  const rateNote =
    st.rate_weight != null && st.rate_weight < 0.5
      ? `Model /90 shrunk (${Math.round(st.rate_weight * 100)}% sample)` +
        (st.npxg90_obs != null ? ` · obs ${fmt(st.npxg90_obs, 2)}/${fmt(st.npxa90_obs, 2)} per 90` : "")
      : `${xgSrc} totals + model /90 (same as Underlying)`;
  const posRow =
    p.position === "GK"
      ? `${statCell("Mins", st.minutes)}${statCell("Starts", st.starts)}${statCell("CS", st.cs)}${statCell("Saves", st.saves)}`
      : p.position === "DEF"
        ? `${statCell("Mins", st.minutes)}${statCell("Starts", st.starts)}${statCell("CS", st.cs)}${statCell("DefCon%", (st.defcon_hit_rate || 0) * 100, 0)}`
        : `${statCell("Mins", st.minutes)}${statCell("Starts", st.starts)}${statCell("DefCon%", (st.defcon_hit_rate || 0) * 100, 0)}`;
  $("card").innerHTML = `
    <div class="hero">
      <div class="who">
        <h2>${p.web_name}</h2>
        <p>${full || ""} · ${p.position} · ${p.team_name || p.team || ""} · £${fmt(p.price)} · ${own}</p>
        <p style="margin-top:.6rem"><span class="badge ${badgeClass(p.action)}">${p.action}</span>
          ${p.status && p.status !== "a" ? ` · ${p.status}` : ""}</p>
        <div class="card-actions">
          <button type="button" data-star="${p.fpl_id}">${starred ? "★ Shortlisted" : "☆ Shortlist"}</button>
          <button type="button" data-team="${p.fpl_id}">${inTeam ? "− Remove from team" : "+ Add to team"}</button>
        </div>
      </div>
      <div class="big">
        <div class="overall">${fmt(p.overall)}</div>
        <div class="upside">Upside ${fmt(p.upside)}</div>
        <div class="xpts">xPts ${fmt(p.xpts, 2)} next window</div>
      </div>
    </div>
    <div class="stats-grid">
      ${statCell("Pts", st.points)}
      ${statCell("G", st.goals)}
      ${statCell("A", st.assists)}
      ${statCell("Bonus", st.bonus)}
    </div>
    <div class="stats-grid" style="margin-top:.4rem">
      ${posRow}
    </div>
    <div class="stats-grid" style="margin-top:.4rem">
      ${statCell("npxG", st.npxg, 2)}
      ${statCell("npxA", st.npxa, 2)}
      ${statCell("npxG/90", st.npxg90, 2)}
      ${statCell("npxA/90", st.npxa90, 2)}
    </div>
    <p class="stat-note">${rateNote}</p>
    <div style="margin-top:.85rem;color:var(--muted);font-size:.78rem">Upcoming fixtures (${isDef ? "CS ease" : "attack ease"})</div>
    ${fixtureChips(p.fixtures || [], isDef ? "def" : "attack")}
    <div class="bars">
      ${bar("Quality Q", p.Q)}
      ${bar("Minutes M", p.M, "m")}
      ${bar("Fixtures O", p.O, "o")}
      ${bar("Value V", p.V, "v")}
    </div>
    <div class="q-kids">
      <div class="kid"><span>Underlying U</span><b>${fmt(p.U)}</b></div>
      <div class="kid"><span>FPL output F</span><b>${fmt(p.F)}</b></div>
      <div class="kid"><span>Role R</span><b>${fmt(p.R)}</b></div>
      <div class="kid"><span>Team T</span><b>${fmt(p.T)}</b></div>
    </div>
    <div class="up-grid">
      ${upRows
        .map(
          ([lab, v]) =>
            `<div class="up-item"><span>${lab}</span><b class="${upTone(v)}">${fmt(v)}</b></div>`
        )
        .join("")}
    </div>
  `;
}

function chip(p, objective, captainId) {
  const cap = captainId === p.fpl_id ? " captain" : "";
  const metric = objective === "overall" ? fmt(p.overall) : fmt(p.xpts, 1);
  return `<div class="chip${cap}" data-id="${p.fpl_id}">
    <div class="nm">${p.web_name}${cap ? " (C)" : ""}</div>
    <div class="dt">${p.team || "—"} · £${fmt(p.price)} · ${metric}</div>
  </div>`;
}

function renderXi(data) {
  state.lastXi = data;
  if (!data.ok) {
    $("xi-board").innerHTML = `<p class="empty">Could not fill a legal 15-man squad at £${fmt(data.budget)}.</p>`;
    $("xi-summary").textContent = "";
    return;
  }
  const obj = data.objective;
  const capId = data.captain ? data.captain.fpl_id : null;
  const xiByPos = { GK: [], DEF: [], MID: [], FWD: [] };
  data.xi.forEach((p) => xiByPos[p.position].push(p));
  $("xi-summary").textContent =
    `${data.formation} · £${fmt(data.spent)} spent · £${fmt(data.remaining)} ITB · ` +
    `XI ${obj} ${fmt(data.xi_objective, 1)} · squad ${fmt(data.squad_objective, 1)}` +
    (data.captain ? ` · C ${data.captain.web_name}` : "");
  const squadBlocks = ["GK", "DEF", "MID", "FWD"]
    .map((pos) => {
      const rows = (data.by_position[pos] || [])
        .map(
          (p) =>
            `<div class="chip" data-id="${p.fpl_id}"><div class="nm">${p.web_name}</div><div class="dt">${p.team || "—"} · £${fmt(p.price)} · ${obj === "overall" ? fmt(p.overall) : fmt(p.xpts, 1)}</div></div>`
        )
        .join("");
      return `<div class="pos-block"><h4>${pos}</h4><div class="xi-line" style="justify-content:flex-start">${rows}</div></div>`;
    })
    .join("");
  $("xi-board").innerHTML = `
    <div class="xi-grid">
      <div class="pitch">
        <h3>Starting XI (${data.formation})</h3>
        <div class="xi-line">${xiByPos.GK.map((p) => chip(p, obj, capId)).join("")}</div>
        <div class="xi-line">${xiByPos.DEF.map((p) => chip(p, obj, capId)).join("")}</div>
        <div class="xi-line">${xiByPos.MID.map((p) => chip(p, obj, capId)).join("")}</div>
        <div class="xi-line">${xiByPos.FWD.map((p) => chip(p, obj, capId)).join("")}</div>
        <h3 style="margin-top:1rem">Bench</h3>
        <div class="xi-line">${data.bench.map((p) => chip(p, obj, null)).join("")}</div>
      </div>
      <div class="bench-col">
        <h3>Full squad (15)</h3>
        ${squadBlocks}
      </div>
    </div>`;
}

async function buildXi() {
  $("xi-summary").textContent = "Building…";
  const params = new URLSearchParams({
    season: state.season,
    as_of: String(state.asOf),
    objective: $("xi-objective").value,
    budget: $("xi-budget").value || "100",
  });
  const data = await (await fetch(`/api/xi?${params}`)).json();
  renderXi(data);
}

function saveXiAsTeam() {
  if (!state.lastXi || !state.lastXi.squad) return;
  state.myTeam = state.lastXi.squad.map((p) => p.fpl_id);
  $("team-bank").value = String(state.lastXi.remaining ?? 0);
  saveMyTeam();
  setTab("team");
}

async function loadPlayersByIds(ids) {
  const byId = new Map();
  const all = await fetchPlayers({ sort: "overall" });
  all.forEach((p) => byId.set(p.fpl_id, p));
  for (const id of ids) {
    if (byId.has(id)) continue;
    const params = new URLSearchParams({
      season: state.season,
      as_of: String(state.asOf),
      id: String(id),
    });
    const p = await (await fetch(`/api/player?${params}`)).json();
    if (p && p.fpl_id) byId.set(p.fpl_id, p);
  }
  return ids.map((id) => byId.get(id)).filter(Boolean);
}

async function renderMyTeam() {
  if (!state.myTeam.length) {
    $("team-board").innerHTML =
      `<p class="empty">No saved team yet. Build an XI and click “Save as my team”, or add players from cards.</p>`;
    $("team-summary").textContent = "";
    return;
  }
  const players = await loadPlayersByIds(state.myTeam);
  const spent = players.reduce((s, p) => s + (p.price || 0), 0);
  const obj = $("team-objective").value;
  const total = players.reduce((s, p) => s + (obj === "overall" ? p.overall || 0 : p.xpts || 0), 0);
  $("team-summary").textContent = `${players.length}/15 · £${fmt(spent)} spent · bank £${fmt(Number($("team-bank").value || 0))} · squad ${obj} ${fmt(total, 1)}`;
  const byPos = { GK: [], DEF: [], MID: [], FWD: [] };
  players.forEach((p) => byPos[p.position]?.push(p));
  $("team-board").innerHTML = ["GK", "DEF", "MID", "FWD"]
    .map((pos) => {
      const rows = (byPos[pos] || [])
        .map(
          (p) =>
            `<div class="chip" data-id="${p.fpl_id}"><div class="nm">${p.web_name}</div><div class="dt">${p.team || "—"} · £${fmt(p.price)} · ${obj === "overall" ? fmt(p.overall) : fmt(p.xpts, 1)}</div></div>`
        )
        .join("");
      return `<div class="pos-block"><h4>${pos} (${(byPos[pos] || []).length})</h4><div class="xi-line" style="justify-content:flex-start">${rows || "<span class='meta'>empty</span>"}</div></div>`;
    })
    .join("");
}

async function suggestTransfers() {
  saveMyTeam();
  if (state.myTeam.length !== 15) {
    $("transfer-board").innerHTML = `<p class="empty">Need a full 15-man squad (currently ${state.myTeam.length}).</p>`;
    return;
  }
  $("transfer-board").innerHTML = `<p class="empty">Finding transfers…</p>`;
  const params = new URLSearchParams({
    season: state.season,
    as_of: String(state.asOf),
    squad: state.myTeam.join(","),
    bank: $("team-bank").value || "0",
    ft: $("team-ft").value || "1",
    objective: $("team-objective").value,
  });
  const data = await (await fetch(`/api/transfers?${params}`)).json();
  if (!data.ok) {
    $("transfer-board").innerHTML = `<p class="empty">${data.error || "Could not suggest transfers."}</p>`;
    return;
  }
  if (!data.suggestions.length) {
    $("transfer-board").innerHTML = `<p class="empty">No improving transfers within bank / FT / team caps.</p>`;
    return;
  }
  const obj = data.objective;
  $("transfer-board").innerHTML =
    `<h3 style="margin:0 0 .7rem;font-size:.95rem">Suggestions (squad ${obj} ${fmt(data.squad_objective, 1)})</h3>` +
    data.suggestions
      .map((s) => {
        const moves = s.moves
          .map(
            (m) =>
              `<div class="transfer-move"><b>${m.out.web_name}</b> (£${fmt(m.out.price)}) → <b>${m.in.web_name}</b> (£${fmt(m.in.price)}) · ${m.in.team || ""} · ${obj} ${fmt(obj === "overall" ? m.in.overall : m.in.xpts, 1)}</div>`
          )
          .join("");
        const gainCls = s.net_gain >= 0 ? "gain" : "gain neg";
        return `<div class="transfer-card">
          <div><span class="${gainCls}">${s.net_gain >= 0 ? "+" : ""}${fmt(s.net_gain, 1)}</span>
            ${obj} · ${s.transfers} FT${s.hits ? ` · ${s.hits} hit` : ""} · cost £${fmt(s.cost)} · bank after £${fmt(s.bank_after)}</div>
          ${moves}
          <div class="card-actions" style="margin-top:.5rem">
            <button type="button" data-apply='${JSON.stringify(s.moves.map((m) => [m.out.fpl_id, m.in.fpl_id]))}'>Apply</button>
          </div>
        </div>`;
      })
      .join("");
}

function applyMoves(pairs) {
  let ids = [...state.myTeam];
  for (const [outId, inId] of pairs) {
    const i = ids.indexOf(outId);
    if (i >= 0) ids[i] = inId;
  }
  state.myTeam = ids;
  saveMyTeam();
  renderMyTeam();
  suggestTransfers();
}

async function renderFixtures() {
  $("fixture-board").innerHTML = `<p class="empty">Loading…</p>`;
  const params = new URLSearchParams({ season: state.season, as_of: String(state.asOf) });
  const data = await (await fetch(`/api/fixtures?${params}`)).json();
  const events = data.events || [];
  const teams = data.teams || [];
  if (!teams.length) {
    $("fixture-board").innerHTML = `<p class="empty">No fixtures found.</p>`;
    return;
  }
  const allEase = teams.flatMap((t) => t.fixtures.filter((f) => !f.blank).map((f) => f.attack_ease));
  const head = events.map((e) => `<th>GW${e}</th>`).join("");
  const body = teams
    .map((t) => {
      const cells = events
        .map((e) => {
          const f = t.fixtures.find((x) => x.event === e);
          if (!f) return `<td>-</td>`;
          const cls = f.blank ? "blank" : easeClass(f.attack_ease, allEase);
          return `<td><span class="fx-chip ${cls}" style="display:inline-block;min-width:3.2rem">${f.label}</span></td>`;
        })
        .join("");
      return `<tr>
        <td class="team">${t.team}</td>
        <td class="num">${fmt(t.avg_fixtures)}</td>
        <td class="num">${fmt(t.attack_ease_avg, 2)}</td>
        ${cells}
      </tr>`;
    })
    .join("");
  $("fixture-board").innerHTML = `
    <table class="fixture-table">
      <thead><tr><th>Team</th><th>Avg O</th><th>Att ease</th>${head}</tr></thead>
      <tbody>${body}</tbody>
    </table>
    <p class="empty" style="padding:.8rem 0 0">Green = easier attack fixtures (leakier opponents). High Avg O = model likes the run.</p>`;
}

function setTab(tab) {
  state.tab = tab;
  document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("on", b.dataset.tab === tab));
  ["players", "shortlist", "xi", "team", "fixtures"].forEach((name) => {
    $(`view-${name}`).classList.toggle("hidden", tab !== name);
  });
  if (tab === "shortlist") renderShortlist();
  if (tab === "xi" && !$("xi-board").dataset.ready) {
    buildXi().then(() => {
      $("xi-board").dataset.ready = "1";
    });
  }
  if (tab === "team") {
    renderMyTeam();
  }
  if (tab === "fixtures") renderFixtures();
}

function bindPills(id, attr, key) {
  $(id).addEventListener("click", (e) => {
    const btn = e.target.closest("button");
    if (!btn) return;
    state[key] = btn.dataset[attr] || "";
    $(id).querySelectorAll("button").forEach((b) => b.classList.toggle("on", b === btn));
    loadPlayers();
  });
}

function onRowClick(e) {
  const apply = e.target.closest("[data-apply]");
  if (apply) {
    applyMoves(JSON.parse(apply.dataset.apply));
    return;
  }
  const star = e.target.closest("[data-star]");
  if (star) {
    toggleShort(star.dataset.star, e);
    if (state.selected === Number(star.dataset.star) && state.tab === "players") selectPlayer(state.selected);
    return;
  }
  const teamBtn = e.target.closest("[data-team]");
  if (teamBtn) {
    toggleTeamMember(teamBtn.dataset.team, e);
    return;
  }
  const row = e.target.closest(".row");
  if (row) selectPlayer(Number(row.dataset.id));
  const chipEl = e.target.closest(".chip[data-id]");
  if (chipEl) selectPlayer(Number(chipEl.dataset.id));
}

$("rows").addEventListener("click", onRowClick);
$("short-rows").addEventListener("click", onRowClick);
$("card").addEventListener("click", onRowClick);
$("xi-board").addEventListener("click", onRowClick);
$("team-board").addEventListener("click", onRowClick);
$("transfer-board").addEventListener("click", onRowClick);

$("tabs").addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-tab]");
  if (btn) setTab(btn.dataset.tab);
});

$("season").addEventListener("change", () => {
  state.season = $("season").value;
  fillAsOf();
  loadShortlist();
  loadMyTeam();
  $("xi-board").dataset.ready = "";
  if (state.tab === "shortlist") renderShortlist();
  else if (state.tab === "xi") buildXi();
  else if (state.tab === "team") renderMyTeam();
  else if (state.tab === "fixtures") renderFixtures();
  else loadPlayers();
});
$("as-of").addEventListener("change", () => {
  state.asOf = Number($("as-of").value);
  $("xi-board").dataset.ready = "";
  if (state.tab === "shortlist") renderShortlist();
  else if (state.tab === "xi") buildXi();
  else if (state.tab === "team") renderMyTeam();
  else if (state.tab === "fixtures") renderFixtures();
  else loadPlayers();
});
$("sort").addEventListener("change", () => {
  state.sort = $("sort").value;
  loadPlayers();
});
let t;
const onSearch = () => {
  clearTimeout(t);
  t = setTimeout(loadPlayers, 150);
};
$("search").addEventListener("input", onSearch);
$("search").addEventListener("change", onSearch);
$("xi-build").addEventListener("click", buildXi);
$("xi-save-team").addEventListener("click", saveXiAsTeam);
$("team-suggest").addEventListener("click", suggestTransfers);
$("team-clear").addEventListener("click", () => {
  state.myTeam = [];
  saveMyTeam();
  renderMyTeam();
  $("transfer-board").innerHTML = `<p class="empty">Transfer suggestions appear here.</p>`;
});
["team-bank", "team-ft"].forEach((id) => {
  $(id).addEventListener("change", saveMyTeam);
});
bindPills("pos-pills", "pos", "position");
bindPills("action-pills", "action", "action");

loadMeta().then(loadPlayers);
