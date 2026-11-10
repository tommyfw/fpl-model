const state = {
  snapshots: [],
  season: null,
  asOf: null,
  position: "",
  action: "",
  team: "",
  sort: "xpts",
  selected: null,
  tab: "players",
  shortlist: new Set(),
  exclude: new Set(),
  excludeTeams: new Set(),
  include: new Set(),
  lock: new Set(),
  myTeam: [],
  teamLock: new Set(),
  plan: null,
  transferOut: null,
  lastXi: null,
  teamViewSeq: 0,
  cache: new Map(),
  horizonGws: 5,
  clubs: {},
  chipsUsed: [],
  horizonMode: "window",
  compareId: null,
  officialAsOf: null,
  betaAsOf: null,
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
function excludeKey() {
  return `fpl-exclude:${state.season || ""}`;
}
function excludeTeamKey() {
  return `fpl-exclude-teams:${state.season || ""}`;
}
function includeKey() {
  return `fpl-include:${state.season || ""}`;
}
function lockKey() {
  return `fpl-lock:${state.season || ""}`;
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

function loadExclude() {
  try {
    state.exclude = new Set(JSON.parse(localStorage.getItem(excludeKey()) || "[]").map(Number));
  } catch {
    state.exclude = new Set();
  }
}
function saveExclude() {
  localStorage.setItem(excludeKey(), JSON.stringify([...state.exclude]));
}

function loadExcludeTeams() {
  try {
    state.excludeTeams = new Set(JSON.parse(localStorage.getItem(excludeTeamKey()) || "[]").map(Number));
  } catch {
    state.excludeTeams = new Set();
  }
}
function saveExcludeTeams() {
  localStorage.setItem(excludeTeamKey(), JSON.stringify([...state.excludeTeams]));
}

function loadInclude() {
  try {
    state.include = new Set(JSON.parse(localStorage.getItem(includeKey()) || "[]").map(Number));
  } catch {
    state.include = new Set();
  }
}
function saveInclude() {
  localStorage.setItem(includeKey(), JSON.stringify([...state.include]));
}

function loadLock() {
  try {
    state.lock = new Set(JSON.parse(localStorage.getItem(lockKey()) || "[]").map(Number));
  } catch {
    state.lock = new Set();
  }
}
function saveLock() {
  localStorage.setItem(lockKey(), JSON.stringify([...state.lock]));
}

function loadMyTeam() {
  try {
    const raw = JSON.parse(localStorage.getItem(teamKey()) || "{}");
    state.myTeam = (raw.ids || []).map(Number);
    if (raw.bank != null) $("team-bank").value = raw.bank;
    if (raw.budget != null && $("team-budget")) $("team-budget").value = raw.budget;
    if (raw.wcMax != null && $("wc-max-value")) $("wc-max-value").value = raw.wcMax;
    if (raw.ft != null) $("team-ft").value = raw.ft;
    if (raw.want != null && $("team-want")) $("team-want").value = raw.want;
    if (raw.entry != null && $("team-entry")) $("team-entry").value = raw.entry;
    const teamIds = new Set(state.myTeam);
    state.teamLock = new Set((raw.keep || []).map(Number).filter((id) => teamIds.has(id)));
    state.plan = null;
    if (Array.isArray(raw.chipsUsed)) state.chipsUsed = raw.chipsUsed;
    if (raw.plan && Array.isArray(raw.plan.ids) && Array.isArray(raw.plan.moves)) {
      const rebuilt = applyPairs([...raw.plan.ids], raw.plan.moves);
      if (sameIds(rebuilt, state.myTeam)) state.plan = { ids: raw.plan.ids.map(Number), bank: Number(raw.plan.bank || 0), moves: raw.plan.moves.map((m) => ({ out: Number(m.out), in: Number(m.in) })) };
    }
  } catch {
    state.myTeam = [];
    state.teamLock = new Set();
    state.plan = null;
  }
}
function saveMyTeam() {
  const teamIds = new Set(state.myTeam);
  state.teamLock = new Set([...state.teamLock].filter((id) => teamIds.has(id)));
  localStorage.setItem(
    teamKey(),
    JSON.stringify({
      ids: state.myTeam,
      keep: [...state.teamLock],
      bank: Number($("team-bank").value || 0),
      budget: Number($("team-budget") && $("team-budget").value) || 100,
      wcMax: $("wc-max-value") && $("wc-max-value").value !== "" ? Number($("wc-max-value").value) : undefined,
      ft: Number($("team-ft").value || 1),
      want: Number($("team-want") && $("team-want").value) || 2,
      entry: Number($("team-entry") && $("team-entry").value) || undefined,
      chipsUsed: state.chipsUsed || [],
      plan: state.plan,
    })
  );
}

function sameIds(a, b) {
  if (!a || !b || a.length !== b.length) return false;
  const sa = [...a].map(Number).sort((x, y) => x - y).join(",");
  const sb = [...b].map(Number).sort((x, y) => x - y).join(",");
  return sa === sb;
}

function applyPairs(ids, moves) {
  const next = [...ids].map(Number);
  for (const m of moves || []) {
    const outId = Number(m.out != null ? m.out : m[0]);
    const inId = Number(m.in != null ? m.in : m[1]);
    const i = next.indexOf(outId);
    if (i >= 0) next[i] = inId;
  }
  return next;
}

function signed(n, d = 1) {
  if (n == null || Number.isNaN(Number(n))) return "—";
  const v = Number(n);
  return `${v >= 0 ? "+" : ""}${v.toFixed(d)}`;
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

function clubInfo(teamId) {
  const clubs = (state.clubs && state.clubs[state.season]) || [];
  return clubs.find((c) => Number(c.team_id) === Number(teamId));
}

function clubBanned(teamId) {
  return teamId != null && state.excludeTeams.has(Number(teamId));
}

function toggleExclude(id, ev) {
  if (ev) ev.stopPropagation();
  id = Number(id);
  const p = state.cache.get(id);
  const tid = p && p.team_id != null ? Number(p.team_id) : null;
  if (state.exclude.has(id)) {
    state.exclude.delete(id);
    if (clubBanned(tid)) state.include.add(id);
    saveExclude();
    saveInclude();
  } else if (clubBanned(tid) && !state.include.has(id)) {
    state.include.add(id);
    saveInclude();
  } else if (clubBanned(tid) && state.include.has(id)) {
    state.include.delete(id);
    saveInclude();
  } else {
    state.exclude.add(id);
    state.include.delete(id);
    state.lock.delete(id);
    saveExclude();
    saveInclude();
    saveLock();
  }
  renderExcludePanel();
  if (state.selected === id && (state.tab === "players" || state.tab === "shortlist")) {
    selectPlayer(state.selected);
  }
  if (state.tab === "players") loadPlayers();
  if (state.tab === "xi") {
    $("xi-board").dataset.ready = "";
    buildXi();
  }
  refreshWcCands();
}

function toggleExcludeTeam(teamId, ev) {
  if (ev) ev.stopPropagation();
  teamId = Number(teamId);
  if (!teamId) return;
  if (state.excludeTeams.has(teamId)) state.excludeTeams.delete(teamId);
  else state.excludeTeams.add(teamId);
  saveExcludeTeams();
  renderExcludePanel();
  if (state.selected && (state.tab === "players" || state.tab === "shortlist")) selectPlayer(state.selected);
  if (state.tab === "xi") {
    $("xi-board").dataset.ready = "";
    buildXi();
  }
  refreshWcCands();
}

async function toggleLock(id, ev) {
  if (ev) ev.stopPropagation();
  id = Number(id);
  if (state.lock.has(id)) state.lock.delete(id);
  else {
    state.lock.add(id);
    state.exclude.delete(id);
    saveExclude();
  }
  saveLock();
  await renderExcludePanel();
  if (state.selected === id && (state.tab === "players" || state.tab === "shortlist")) {
    await selectPlayer(state.selected);
  }
  if (state.tab === "players") await loadPlayers();
  if (state.tab === "xi") {
    $("xi-board").dataset.ready = "";
    await buildXi();
  }
  refreshWcCands();
}

function toggleTeamMember(id, ev) {
  if (ev) ev.stopPropagation();
  id = Number(id);
  const idx = state.myTeam.indexOf(id);
  if (idx >= 0) {
    state.myTeam.splice(idx, 1);
    state.teamLock.delete(id);
    state.plan = null;
    state.transferOut = null;
  } else if (state.myTeam.length < 15) {
    state.myTeam.push(id);
    state.plan = null;
    state.transferOut = null;
  }
  saveMyTeam();
  if (state.selected === id && state.tab === "players") selectPlayer(id);
  if (state.tab === "team") renderMyTeam();
}

function toggleKeep(id, ev) {
  if (ev) ev.stopPropagation();
  id = Number(id);
  if (!state.myTeam.includes(id)) return;
  if (state.teamLock.has(id)) state.teamLock.delete(id);
  else state.teamLock.add(id);
  saveMyTeam();
  renderExcludePanel();
  if (state.selected === id && (state.tab === "players" || state.tab === "shortlist")) {
    selectPlayer(id);
  }
  if (state.tab === "team") renderMyTeam();
}

async function loadMeta() {
  const meta = await (await fetch("/api/meta")).json();
  state.snapshots = meta.snapshots || [];
  const def = meta.default;
  state.season = def ? def.season : null;
  state.asOf = def ? def.as_of_gw : null;
  state.horizonGws = meta.horizon_gws || 5;
  state.clubs = meta.clubs || {};
  state.officialAsOf = meta.official_as_of || null;
  state.betaAsOf = meta.beta_as_of || null;
  fillSeason();
  fillAsOf();
  fillTeamEvent();
  fillClubs();
  setHorizonNote();
  loadShortlist();
  loadExclude();
  loadExcludeTeams();
  loadInclude();
  loadLock();
  loadMyTeam();
}

function setHorizonNote() {
  const el = $("horizon-label");
  if (!el) return;
  const cur = (state.snapshots || []).find((s) => s.season === state.season && Number(s.as_of_gw) === Number(state.asOf));
  const hz =
    state.horizonMode === "next"
      ? "This GW xPts (Overall still 5-GW)"
      : `Next ${state.horizonGws || 5} GWs`;
  let asof = `as-of GW${state.asOf}`;
  if (cur && cur.is_beta) {
    const off = state.officialAsOf && state.officialAsOf.as_of_gw;
    asof += off ? ` beta (official GW${off})` : " beta (FPL still on previous GW)";
  } else {
    asof += " official";
  }
  el.textContent = `${hz} · ${asof}`;
}

function fillSeason() {
  const seasons = [...new Set(state.snapshots.map((s) => s.season))];
  $("season").innerHTML = seasons
    .map((s) => `<option ${s === state.season ? "selected" : ""}>${s}</option>`)
    .join("");
}

function fillClubs() {
  const sel = $("club");
  if (!sel) return;
  const clubs = (state.clubs && state.clubs[state.season]) || [];
  const prev = state.team || sel.value;
  sel.innerHTML =
    `<option value="">All clubs</option>` +
    clubs
      .map(
        (c) =>
          `<option value="${c.team_id}" ${String(c.team_id) === String(prev) ? "selected" : ""}>${c.short_name}</option>`
      )
      .join("");
  if (![...sel.options].some((o) => o.value === String(prev))) {
    state.team = "";
    sel.value = "";
  } else {
    state.team = sel.value;
  }
  const xiSel = $("xi-exclude-club");
  const wcSel = $("wc-exclude-club");
  [xiSel, wcSel].forEach((box) => {
    if (!box) return;
    const keep = box.value;
    box.innerHTML =
      `<option value="">Club…</option>` +
      clubs
        .map((c) => `<option value="${c.team_id}">${c.short_name}</option>`)
        .join("");
    if ([...box.options].some((o) => o.value === keep)) box.value = keep;
  });
}

function fillAsOf() {
  const gws = state.snapshots.filter((s) => s.season === state.season);
  $("as-of").innerHTML = gws
    .map(
      (s) =>
        `<option value="${s.as_of_gw}" ${s.as_of_gw === state.asOf ? "selected" : ""}>GW ${s.as_of_gw}${s.is_beta ? " beta" : ""} (${s.n})</option>`
    )
    .join("");
  if (gws.length && !gws.some((s) => s.as_of_gw === state.asOf)) {
    state.asOf = gws[0].as_of_gw;
    $("as-of").value = String(state.asOf);
  }
  fillTeamEvent();
}

function fillTeamEvent() {
  const sel = $("team-event");
  if (!sel) return;
  const t = Number(state.asOf);
  const hz = Math.max(1, Number(state.horizonGws || 5));
  if (!Number.isFinite(t) || t < 1) {
    sel.innerHTML = "";
    return;
  }
  const prev = sel.value;
  sel.innerHTML = Array.from({ length: hz }, (_, k) => {
    const gw = t + k;
    const tag = k === 0 ? "this" : `T+${k}`;
    return `<option value="${gw}">GW ${gw} (${tag})</option>`;
  }).join("");
  if ([...sel.options].some((o) => o.value === prev)) sel.value = prev;
}

function flagHint(p) {
  const flags = (p.card && p.card.flags) || p.flags || {};
  const bits = [];
  if (flags.understat === false) bits.push("no Understat");
  if (flags.thin_sample) bits.push("thin");
  if (flags.no_fbref) bits.push("no FBref");
  return bits.length ? ` · ${bits.join(", ")}` : "";
}

function rowHtml(p, i) {
  const starred = state.shortlist.has(p.fpl_id);
  const banned = state.exclude.has(p.fpl_id) || (clubBanned(p.team_id) && !state.include.has(p.fpl_id));
  return `
    <div class="row ${state.selected === p.fpl_id ? "sel" : ""} ${banned ? "banned" : ""}" data-id="${p.fpl_id}">
      <span class="c-star"><button type="button" class="${starred ? "on" : ""}" data-star="${p.fpl_id}" aria-label="Shortlist">${starred ? "★" : "☆"}</button></span>
      <span class="c-rank">${i + 1}</span>
      <span class="c-name">
        <div class="name">${p.web_name}${state.lock.has(p.fpl_id) ? " · included" : banned ? (state.exclude.has(p.fpl_id) ? " · XI excluded" : " · club banned") : ""}</div>
        <div class="meta">${p.position} · ${p.team || "—"} · £${fmt(p.price)}${flagHint(p)}</div>
      </span>
      <span class="c-num">${fmt(p.overall)}</span>
      <span class="c-num">${fmt(p.upside)}</span>
      <span class="c-num">${fmt(state.horizonMode === "next" ? p.xpts_next : p.xpts, 1)}</span>
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
  if (extra.team) params.set("team", String(extra.team));
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
  const sort = state.horizonMode === "next" && state.sort === "xpts" ? "xpts_next" : state.sort;
  const players = await fetchPlayers({ q, team: state.team, sort });
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
  const all = await fetchPlayers({ sort: "xpts" });
  const byId = new Map(all.map((p) => [p.fpl_id, p]));
  for (const id of ids.filter((x) => !byId.has(x))) {
    const p = await (await fetch(`/api/player?${new URLSearchParams({ season: state.season, as_of: String(state.asOf), id: String(id) })}`)).json();
    if (p && p.fpl_id) byId.set(p.fpl_id, p);
  }
  const players = ids.map((id) => byId.get(id)).filter(Boolean).sort((a, b) => (b.xpts || 0) - (a.xpts || 0));
  $("short-rows").innerHTML = players.map((p, i) => rowHtml(p, i)).join("");
}

function highlight() {
  document.querySelectorAll(".row").forEach((el) => {
    el.classList.toggle("sel", Number(el.dataset.id) === state.selected);
  });
}

async function selectPlayer(id, { stay } = {}) {
  state.selected = id;
  highlight();
  if (!stay && state.tab !== "players" && state.tab !== "shortlist") {
    setTab("players");
  }
  if (stay && state.tab !== "players" && state.tab !== "shortlist") {
    return;
  }
  const params = new URLSearchParams({
    season: state.season,
    as_of: String(state.asOf),
    id: String(id),
  });
  const p = await (await fetch(`/api/player?${params}`)).json();
  if (state.compareId && Number(state.compareId) !== Number(id)) {
    const other = await (
      await fetch(
        `/api/player?${new URLSearchParams({
          season: state.season,
          as_of: String(state.asOf),
          id: String(state.compareId),
        })}`
      )
    ).json();
    $("card").innerHTML =
      compareDelta(p, other) + `<div class="compare-grid">${cardMarkup(p)}${cardMarkup(other)}</div>`;
    return;
  }
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

function fixtureTone(f, mode = "attack") {
  if (!f || f.blank) return "blank";
  const rating = mode === "def" ? f.def_rating ?? f.rating : f.rating;
  const n = Number(rating);
  if (n >= 1 && n <= 5) return `r${n}`;
  const band = mode === "def" ? f.def_band : f.attack_band;
  if (band === "easy") return "r2";
  if (band === "hard") return "r4";
  return "r3";
}

function fixtureTitle(f) {
  return f.title || f.opp_name || f.opp || "";
}

function fixtureChips(fixtures, mode = "attack") {
  if (!fixtures || !fixtures.length) return "";
  return `<div class="fx-row">${fixtures
    .map((f) => {
      const cls = [fixtureTone(f, mode), f.thin_venue ? "thin" : ""].filter(Boolean).join(" ");
      const detail =
        f.blank || f.attack_ease == null
          ? ""
          : mode === "def"
            ? `<span class="gw">P(CS) ${fmt((f.p_cs || 0) * 100, 0)}%</span>`
            : `<span class="gw">ease ${fmt(f.attack_ease, 2)}</span>`;
      const nBit = f.thin_venue && f.venue_n != null ? `<span class="n">n=${f.venue_n}</span>` : "";
      return `<div class="fx-chip ${cls}" title="${fixtureTitle(f)}"><span class="gw">GW${f.event}</span>${f.label}${detail}${nBit}</div>`;
    })
    .join("")}</div>`;
}

function statCell(label, value, d = 0) {
  return `<div class="stat"><span>${label}</span><b>${fmt(value, d)}</b></div>`;
}

function lastWeekLine(p) {
  const lw = p.last_week;
  if (!lw || (lw.pred == null && lw.actual == null)) return "";
  return `<p class="last-week">Last GW${lw.event}: raw forecast ${fmt(lw.pred, 1)} xPts · actual ${lw.actual == null ? "—" : lw.actual} pts</p>`;
}

function thisGwLine(p) {
  const shown = p.xpts_next;
  const raw = p.xpts_next_raw;
  const nudge = p.cal_nudge;
  if (shown == null) return `<div class="xpts">This GW —</div>`;
  if (raw != null && nudge && Math.abs(nudge) >= 0.05) {
    const sign = nudge > 0 ? "+" : "";
    return `<div class="xpts">This GW ${fmt(shown, 2)} <span style="font-size:.72rem;color:var(--muted)">(raw ${fmt(raw, 2)}, ${sign}${fmt(nudge, 2)} cal)</span></div>`;
  }
  return `<div class="xpts">This GW ${fmt(shown, 2)}</div>`;
}

function playerGwStrip(p) {
  const xs = p.xpts_gws || (p.card && p.card.xpts_gws) || [];
  if (!xs.length) return "";
  const t0 = Number(state.asOf) || 0;
  return `<p class="stat-note">Packed xPts: ${xs.map((x, i) => `GW${t0 + i} ${fmt(x, 1)}`).join(" · ")}</p>`;
}

function compareDelta(a, b) {
  const rows = [
    ["xPts 5GW", a.xpts, b.xpts, 1],
    ["This GW", a.xpts_next, b.xpts_next, 1],
    ["Overall", a.overall, b.overall, 1],
    ["Minutes M", a.M, b.M, 0],
    ["Fixtures O", a.O, b.O, 0],
    ["£", a.price, b.price, 1],
    ["Owned %", a.selected_by_percent, b.selected_by_percent, 1],
  ];
  const cells = rows
    .map(([lab, x, y, d]) => {
      const dx = x != null && y != null ? x - y : null;
      const cls = dx == null ? "" : dx > 0.05 ? "plus" : dx < -0.05 ? "minus" : "";
      return `<div class="up-item"><span>${lab}</span><b class="${cls}">${fmt(x, d)} / ${fmt(y, d)}${dx == null ? "" : ` (${dx > 0 ? "+" : ""}${fmt(dx, d)})`}</b></div>`;
    })
    .join("");
  return `<div class="compare-delta"><p class="stat-note">${a.web_name} vs ${b.web_name} (Δ = left − right)</p><div class="up-grid">${cells}</div></div>`;
}

function cardMarkup(p) {
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
  const hardBan = state.exclude.has(p.fpl_id);
  const teamBan = clubBanned(p.team_id) && !state.include.has(p.fpl_id);
  let excludeLabel = "Exclude from XI";
  if (hardBan) excludeLabel = "Allow in XI";
  else if (clubBanned(p.team_id) && state.include.has(p.fpl_id)) excludeLabel = "Use club ban";
  else if (teamBan) excludeLabel = "Allow in XI";
  const club = clubInfo(p.team_id);
  const clubBtn = p.team_id
    ? `<button type="button" data-exclude-team="${p.team_id}">${
        clubBanned(p.team_id) ? `Allow ${club ? club.short_name : "club"}` : `Exclude ${club ? club.short_name : "club"}`
      }</button>`
    : "";
  const isDef = p.position === "DEF" || p.position === "GK"; // fixture chip ease mode
  const flags = (p.card && p.card.flags) || p.flags || {};
  const flagBits = [
    flags.understat === false ? `<span class="badge hold">No Understat map</span>` : "",
    flags.thin_sample ? `<span class="badge hold">Thin sample</span>` : "",
    flags.no_fbref ? `<span class="badge hold">No FBref</span>` : "",
    p.position === "GK" && st.psxg_net90 == null && st.psxg == null
      ? `<span class="badge hold">PSxG missing</span>`
      : "",
    ...minutesRiskBadges(p),
  ].filter(Boolean).join(" ");
  const partsXp = (p.card && p.card.xpts_parts) || {};
  const xpRows = [
    ["Appear", partsXp.appear],
    ["Attack", partsXp.attack],
    ["CS", partsXp.cs],
    ["Saves", partsXp.saves],
    ["DefCon", partsXp.defcon],
    ["Bonus", partsXp.bonus],
  ].filter(([, v]) => v != null);
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
  const gkPsxg =
    p.position === "GK"
      ? `<div class="stats-grid" style="margin-top:.4rem">${statCell("PSxG", st.psxg, 2)}${statCell("PSxG±/90", st.psxg_net90, 2)}${statCell("Stop", st.shot_stopping, 2)}</div>`
      : "";
  return `
    <div class="hero">
      <div class="who">
        <h2>${p.web_name}</h2>
        <p>${full || ""} · ${p.position} · ${p.team_name || p.team || ""} · £${fmt(p.price)} · ${own}</p>
        <p style="margin-top:.6rem"><span class="badge ${badgeClass(p.action)}">${p.action}</span>
          ${flagBits ? ` ${flagBits}` : ""}
          ${p.status && p.status !== "a" ? ` · ${p.status}` : ""}</p>
        ${p.news ? `<p class="stat-note">${p.news}</p>` : ""}
        <div class="card-actions">
          <button type="button" data-star="${p.fpl_id}">${starred ? "★ Shortlisted" : "☆ Shortlist"}</button>
          <button type="button" data-team="${p.fpl_id}">${inTeam ? "− Remove from team" : "+ Add to team"}</button>
          ${
            inTeam
              ? `<button type="button" data-keep="${p.fpl_id}">${
                  state.teamLock.has(p.fpl_id) ? "Unlock for transfers" : "Lock (don't transfer out)"
                }</button>`
              : ""
          }
          ${
            state.transferOut && !inTeam && state.cache.get(state.transferOut) && state.cache.get(state.transferOut).position === p.position
              ? `<button type="button" data-bring="${p.fpl_id}">Bring in for ${state.cache.get(state.transferOut).web_name}</button>`
              : inTeam
                ? `<button type="button" data-transfer-out="${p.fpl_id}">${
                    state.transferOut === p.fpl_id ? "Cancel transfer" : "Transfer out"
                  }</button>`
                : ""
          }
          <button type="button" data-exclude="${p.fpl_id}">${excludeLabel}</button>
          <button type="button" data-lock="${p.fpl_id}">${state.lock.has(p.fpl_id) ? "Remove from include" : "Include in XI"}</button>
          <button type="button" data-compare="${p.fpl_id}">${
            Number(state.compareId) === Number(p.fpl_id) ? "Clear compare" : "Compare"
          }</button>
          ${clubBtn}
        </div>
        ${lastWeekLine(p)}
      </div>
      <div class="big">
        <div class="overall">${fmt(p.overall)}</div>
        <div class="upside">Upside ${fmt(p.upside)}</div>
        <div class="xpts">xPts ${fmt(p.xpts, 2)} next window</div>
        ${thisGwLine(p)}
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
    ${gkPsxg}
    <div class="stats-grid" style="margin-top:.4rem">
      ${statCell("npxG", st.npxg, 2)}
      ${statCell("npxA", st.npxa, 2)}
      ${statCell("npxG/90", st.npxg90, 2)}
      ${statCell("npxA/90", st.npxa90, 2)}
    </div>
    <p class="stat-note">${rateNote}</p>
    ${
      xpRows.length
        ? `<div class="up-grid" style="margin-top:.6rem">${xpRows
            .map(([lab, v]) => `<div class="up-item"><span>xPts ${lab}</span><b>${fmt(v, 2)}</b></div>`)
            .join("")}</div>`
        : ""
    }
    ${playerGwStrip(p)}
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

function renderCard(p) {
  $("card").innerHTML = cardMarkup(p);
}

function weeklyXiStrip(data) {
  const xi = data.xi || [];
  const n = Math.max(0, ...xi.map((p) => (p.xpts_gws || []).length));
  if (!n) return "";
  const bits = [];
  for (let t = 0; t < n; t += 1) {
    const tot = xi.reduce((s, p) => s + Number((p.xpts_gws || [])[t] || 0), 0);
    bits.push(`GW${Number(data.as_of_gw) + t} ${tot.toFixed(1)}`);
  }
  return `<p class="stat-note">Weekly stacked XI xPts (no captain): ${bits.join(" · ")}</p>`;
}

function minutesRiskOf(p) {
  return (p && (p.minutes_risk || (p.card && p.card.minutes_risk))) || null;
}

function minutesRiskBadges(p) {
  const r = minutesRiskOf(p);
  if (!r || r.level === "ok") return [];
  const lab = (r.labels && r.labels[0]) || (r.level === "risk" ? "OUT / minutes" : "minutes watch");
  return [`<span class="badge ${r.level}">${lab}</span>`];
}

function chip(p, objective, captainId, opts = {}) {
  const cap = captainId === p.fpl_id ? " captain" : "";
  const dim = p.fixture === "BLANK" ? " blank" : "";
  const risk = minutesRiskOf(p);
  const riskCls = risk && risk.level !== "ok" ? ` ${risk.level}` : "";
  const keep = opts.team && state.teamLock.has(p.fpl_id) ? " keep" : "";
  const out = opts.team && state.transferOut === p.fpl_id ? " out" : "";
  const metric =
    objective === "combined"
      ? `${fmt(p.xpts, 1)} / ${fmt(p.overall)}`
      : objective === "overall"
        ? fmt(p.overall)
        : fmt(p.xpts_gw != null ? p.xpts_gw : p.xpts, 1);
  const fx = p.fixture ? ` · ${p.fixture}` : "";
  const pin = p.locked ? " · included" : "";
  const lockBit = opts.team
    ? ` · <button type="button" data-transfer-out="${p.fpl_id}">${state.transferOut === p.fpl_id ? "Cancel" : "Out"}</button> · <button type="button" data-keep="${p.fpl_id}">${state.teamLock.has(p.fpl_id) ? "Unlock" : "Lock"}</button>`
    : "";
  const riskBit =
    p.fixture === "BLANK"
      ? ` · <span class="badge blank">BLANK</span>`
      : risk && risk.level === "risk"
        ? ` · <span class="badge risk">${(risk.labels && risk.labels[0]) || "OUT"}</span>`
        : risk && risk.level === "watch"
          ? ` · <span class="badge watch">${(risk.labels && risk.labels[0]) || "watch"}</span>`
          : "";
  const capLab =
    Number(captainId) === Number(p.fpl_id)
      ? opts.triple
        ? " (TC)"
        : " (C)"
      : opts.viceId != null && Number(opts.viceId) === Number(p.fpl_id)
        ? " (VC)"
        : "";
  const loud = p.fixture === "BLANK" ? " BLANK" : risk && risk.level === "risk" ? " OUT" : "";
  return `<div class="chip${cap}${dim}${riskCls}${keep}${out}" data-id="${p.fpl_id}">
    <div class="nm">${p.web_name}${capLab}${loud}${pin}${opts.team && state.teamLock.has(p.fpl_id) ? " · locked" : ""}</div>
    <div class="dt">${p.team || "—"} · £${fmt(p.price)} · ${metric}${fx}${riskBit}${lockBit}</div>
  </div>`;
}

function renderXi(data) {
  state.lastXi = data;
  if (!data.ok) {
    const msg =
      data.error ||
      `Could not fill a legal 15-man squad at £${fmt(data.budget)}.`;
    $("xi-board").innerHTML = `<p class="empty">${msg}</p>`;
    $("xi-summary").textContent = "";
    return;
  }
  const obj = data.objective;
  const capId = data.captain ? data.captain.fpl_id : null;
  const xiByPos = { GK: [], DEF: [], MID: [], FWD: [] };
  data.xi.forEach((p) => xiByPos[p.position].push(p));
  const mixBit =
    obj === "combined"
      ? `XI mix ${fmt(data.xi_objective, 1)} · xPts ${fmt(data.xi_xpts, 1)} · ovr ${fmt(data.xi_overall)}`
      : `XI ${obj} ${fmt(data.xi_objective, 1)}`;
  $("xi-summary").textContent =
    `${data.formation} · £${fmt(data.spent)} spent · £${fmt(data.remaining)} ITB · ` +
    `${mixBit} · squad ${fmt(data.squad_objective, 1)}` +
    (data.captain ? ` · C ${data.captain.web_name}` : "");
  const squadBlocks = ["GK", "DEF", "MID", "FWD"]
    .map((pos) => {
      const rows = (data.by_position[pos] || [])
        .map((p) => {
          const metric =
            obj === "combined"
              ? `${fmt(p.xpts, 1)} / ${fmt(p.overall)}`
              : obj === "overall"
                ? fmt(p.overall)
                : fmt(p.xpts, 1);
          const pin = p.locked ? " · included" : "";
          return `<div class="chip" data-id="${p.fpl_id}"><div class="nm">${p.web_name}${pin}</div><div class="dt">${p.team || "—"} · £${fmt(p.price)} · ${metric}</div></div>`;
        })
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
        <h3 style="margin-top:1rem">Bench (auto-sub order)</h3>
        <div class="xi-line">${data.bench.map((p) => chip(p, obj, null)).join("")}</div>
        ${weeklyXiStrip(data)}
      </div>
      <div class="bench-col">
        <h3>Full squad (15)</h3>
        ${squadBlocks}
      </div>
    </div>`;
}

let xiAbort = null;

async function buildXi() {
  $("xi-summary").textContent = "Building…";
  const params = new URLSearchParams({
    season: state.season,
    as_of: String(state.asOf),
    objective: $("xi-objective").value,
    budget: $("xi-budget").value || "100",
  });
  if (state.exclude.size) params.set("exclude", [...state.exclude].join(","));
  if (state.excludeTeams.size) params.set("exclude_teams", [...state.excludeTeams].join(","));
  if (state.include.size) params.set("include", [...state.include].join(","));
  if (state.lock.size) params.set("lock", [...state.lock].join(","));
  if (xiAbort) xiAbort.abort();
  xiAbort = new AbortController();
  try {
    const data = await (await fetch(`/api/xi?${params}`, { signal: xiAbort.signal })).json();
    await renderExcludePanel();
    renderXi(data);
  } catch (err) {
    if (err && err.name === "AbortError") return;
    $("xi-summary").textContent = "Could not build squad";
  }
}

function saveXiAsTeam() {
  if (!state.lastXi || !state.lastXi.squad) return;
  state.myTeam = state.lastXi.squad.map((p) => p.fpl_id);
  $("team-bank").value = String(state.lastXi.remaining ?? 0);
  state.plan = null;
  state.transferOut = null;
  saveMyTeam();
  setTab("team");
}

async function loadPlayersByIds(ids) {
  const byId = new Map();
  const all = await fetchPlayers({ sort: "xpts" });
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

async function renderExcludePanel() {
  const mounts = [
    {
      empty: "xi-exclude-empty",
      list: "xi-exclude-list",
      teams: "xi-exclude-teams",
      inc: "xi-include-list",
      lock: "xi-lock-list",
    },
    {
      empty: "wc-exclude-empty",
      list: "wc-exclude-list",
      teams: "wc-exclude-teams",
      inc: "wc-include-list",
      lock: "wc-lock-list",
      keep: "wc-keep-list",
    },
  ];
  const teamIds = [...state.excludeTeams];
  const ids = [...state.exclude];
  const allowIds = [...state.include];
  const lockIds = [...state.lock];
  const keepIds = [...state.teamLock];
  const playersById = new Map();
  const need = [...new Set([...ids, ...allowIds, ...lockIds, ...keepIds])];
  if (need.length && state.season) {
    (await loadPlayersByIds(need)).forEach((p) => playersById.set(p.fpl_id, p));
  }
  for (const m of mounts) {
    const empty = $(m.empty);
    const list = $(m.list);
    if (!list) continue;
    const hasAny =
      teamIds.length || ids.length || allowIds.length || lockIds.length || (m.keep && keepIds.length);
    if (empty) empty.classList.toggle("hidden", !!hasAny);
    if ($(m.teams)) {
      $(m.teams).innerHTML = teamIds
        .map((tid) => {
          const c = clubInfo(tid);
          const name = c ? c.short_name : `Team ${tid}`;
          return `<div class="chip excluded"><div class="nm">${name}</div><div class="dt">club banned · <button type="button" data-exclude-team="${tid}">Allow</button></div></div>`;
        })
        .join("");
    }
    list.innerHTML = ids
      .map((id) => playersById.get(id))
      .filter(Boolean)
      .map(
        (p) =>
          `<div class="chip excluded" data-id="${p.fpl_id}">
          <div class="nm">${p.web_name}</div>
          <div class="dt">${p.position} · ${p.team || "—"} · <button type="button" data-exclude="${p.fpl_id}">Allow</button></div>
        </div>`
      )
      .join("");
    if ($(m.inc)) {
      $(m.inc).innerHTML = allowIds
        .map((id) => playersById.get(id))
        .filter(Boolean)
        .map(
          (p) =>
            `<div class="chip" data-id="${p.fpl_id}">
            <div class="nm">${p.web_name}</div>
            <div class="dt">club exception · <button type="button" data-exclude="${p.fpl_id}">Use club ban</button></div>
          </div>`
        )
        .join("");
    }
    if ($(m.lock)) {
      $(m.lock).innerHTML = lockIds
        .map((id) => playersById.get(id))
        .filter(Boolean)
        .map(
          (p) =>
            `<div class="chip keep" data-id="${p.fpl_id}">
            <div class="nm">${p.web_name}</div>
            <div class="dt">must include · <button type="button" data-lock="${p.fpl_id}">Remove</button></div>
          </div>`
        )
        .join("");
    }
    if (m.keep && $(m.keep)) {
      $(m.keep).innerHTML = keepIds
        .map((id) => playersById.get(id))
        .filter(Boolean)
        .map(
          (p) =>
            `<div class="chip keep" data-id="${p.fpl_id}">
            <div class="nm">${p.web_name}</div>
            <div class="dt">keep from your 15 · <button type="button" data-keep="${p.fpl_id}">Unlock</button></div>
          </div>`
        )
        .join("");
    }
  }
}

async function refreshWcCands() {
  const box = $("wc-player-hits");
  const qEl = $("wc-player-q");
  if (!box || !qEl) return;
  const q = (qEl.value || "").trim();
  if (q.length < 2 || !state.season) {
    box.innerHTML = "";
    return;
  }
  const params = new URLSearchParams({
    season: state.season,
    as_of: String(state.asOf),
    q,
    sort: "xpts",
  });
  try {
    const data = await (await fetch(`/api/players?${params}`)).json();
    const rows = (data.players || []).slice(0, 8);
    box.innerHTML = rows
      .map((p) => {
        const banned = state.exclude.has(p.fpl_id);
        const must = state.lock.has(p.fpl_id);
        const inTeam = state.myTeam.includes(p.fpl_id);
        const keep = state.teamLock.has(p.fpl_id);
        return `<div class="chip${banned ? " excluded" : must || keep ? " keep" : ""}" data-id="${p.fpl_id}">
          <div class="nm">${p.web_name}</div>
          <div class="dt">${p.team || "—"} · £${fmt(p.price)} · ${fmt(p.xpts, 1)}
            · <button type="button" data-exclude="${p.fpl_id}">${banned ? "Allow" : "Ban"}</button>
            · <button type="button" data-lock="${p.fpl_id}">${must ? "Drop include" : "Must include"}</button>
            ${inTeam ? ` · <button type="button" data-keep="${p.fpl_id}">${keep ? "Unlock" : "Keep"}</button>` : ""}
          </div>
        </div>`;
      })
      .join("");
  } catch {
    box.innerHTML = "";
  }
}

async function renderMyTeam() {
  const seq = ++state.teamViewSeq;
  if (!state.myTeam.length) {
    $("team-board").innerHTML =
      `<p class="empty">No saved team yet. Build an XI and click “Save as my team”, or add players from cards.</p>`;
    $("team-summary").textContent = "";
    return;
  }
  const players = await loadPlayersByIds(state.myTeam);
  if (seq !== state.teamViewSeq) return;
  const spent = players.reduce((s, p) => s + (p.price || 0), 0);
  const budget = Number($("team-budget") && $("team-budget").value) || 100;
  const over = spent - budget;
  const obj = $("team-objective").value;
  const total = players.reduce((s, p) => s + (obj === "overall" ? p.overall || 0 : p.xpts || 0), 0);
  const capBit = over > 0.05 ? ` · £${fmt(over)} over cap` : ` · cap £${fmt(budget)}`;
  const keepBit = state.teamLock.size ? ` · ${state.teamLock.size} locked` : "";
  $("team-summary").textContent = `${players.length}/15 · £${fmt(spent)} / £${fmt(budget)} spent · bank £${fmt(Number($("team-bank").value || 0))}${capBit}${keepBit} · squad ${obj} ${fmt(total, 1)}`;
  const byPos = { GK: [], DEF: [], MID: [], FWD: [] };
  players.forEach((p) => byPos[p.position]?.push(p));
  $("team-board").innerHTML = ["GK", "DEF", "MID", "FWD"]
    .map((pos) => {
      const rows = (byPos[pos] || [])
        .map(
          (p) =>
            `<div class="chip${state.teamLock.has(p.fpl_id) ? " keep" : ""}${state.transferOut === p.fpl_id ? " out" : ""}${minutesRiskOf(p) && minutesRiskOf(p).level !== "ok" ? " " + minutesRiskOf(p).level : ""}" data-id="${p.fpl_id}"><div class="nm">${p.web_name}${state.teamLock.has(p.fpl_id) ? " · locked" : ""}${state.transferOut === p.fpl_id ? " · out" : ""}</div><div class="dt">${p.team || "—"} · £${fmt(p.price)} · ${obj === "overall" ? fmt(p.overall) : fmt(p.xpts, 1)}${minutesRiskOf(p) && minutesRiskOf(p).level !== "ok" ? " · " + ((minutesRiskOf(p).labels && minutesRiskOf(p).labels[0]) || "mins") : ""} · <button type="button" data-transfer-out="${p.fpl_id}">${state.transferOut === p.fpl_id ? "Cancel" : "Out"}</button> · <button type="button" data-keep="${p.fpl_id}">${state.teamLock.has(p.fpl_id) ? "Unlock" : "Lock"}</button></div></div>`
        )
        .join("");
      return `<div class="pos-block"><h4>${pos} (${(byPos[pos] || []).length})</h4><div class="xi-line" style="justify-content:flex-start">${rows || "<span class='meta'>empty</span>"}</div></div>`;
    })
    .join("");
}

function renderLineup(data) {
  if (!data.ok) {
    $("team-board").innerHTML = `<p class="empty">${data.error || "Could not pick a starting XI."}</p>`;
    return;
  }
  const capId = data.captain ? data.captain.fpl_id : null;
  const viceId = data.vice_captain ? data.vice_captain.fpl_id : null;
  const xiByPos = { GK: [], DEF: [], MID: [], FWD: [] };
  data.xi.forEach((p) => xiByPos[p.position].push(p));
  const chipBit =
    data.chip === "bboost"
      ? " · bench boost on"
      : data.chip === "3xc"
        ? " · triple captain on"
        : "";
  const evBit =
    data.captain_ev != null
      ? ` · C EV ${fmt(data.captain_ev, 1)} (VC-as-C ${fmt(data.vice_ev, 1)}) / 3XC ${fmt(data.triple_ev, 1)}`
      : "";
  $("team-summary").textContent =
    `GW${data.event} XI ${data.formation} · ${fmt(data.xi_objective, 1)} xPts` +
    (data.captain ? ` · ${data.triple ? "TC" : "C"} ${data.captain.web_name}` : "") +
    (data.vice_captain ? ` · VC ${data.vice_captain.web_name}` : "") +
    evBit +
    chipBit +
    (data.offset ? ` · T+${data.offset} fixtures` : " · this GW fixtures");
  const benchLabel = data.include_bench ? "Bench (playing — BB)" : "Bench";
  const chipOpts = { team: true, triple: !!data.triple, viceId };
  const scan = chipScanHtml(data.chip_scan, data.template);
  $("team-board").innerHTML = `
    <div class="pitch">
      <h3>GW${data.event} starting XI (${data.formation})${chipBit}</h3>
      <div class="xi-line">${xiByPos.GK.map((p) => chip(p, "xpts_gw", capId, chipOpts)).join("")}</div>
      <div class="xi-line">${xiByPos.DEF.map((p) => chip(p, "xpts_gw", capId, chipOpts)).join("")}</div>
      <div class="xi-line">${xiByPos.MID.map((p) => chip(p, "xpts_gw", capId, chipOpts)).join("")}</div>
      <div class="xi-line">${xiByPos.FWD.map((p) => chip(p, "xpts_gw", capId, chipOpts)).join("")}</div>
      <h3 style="margin-top:1rem">${benchLabel}</h3>
      <div class="xi-line">${data.bench.map((p) => chip(p, "xpts_gw", null, { team: true })).join("")}</div>
      ${scan}
    </div>`;
}

function freehitMeter(fh) {
  if (!fh || fh.ok === false) return "";
  const s = fh.score == null ? 0 : fh.score;
  const cls = s >= 7 ? "plus" : s <= 3 ? "minus" : "";
  const w = Math.max(0, Math.min(100, s * 10));
  return `<div class="fh-meter">
    <p><b class="${cls}">Free hit ${s}/10</b> · ${fh.verdict || ""} this GW
      · yours ${fmt(fh.current_xi, 1)} → FH ${fmt(fh.fh_xi, 1)} (${signed(fh.xi_gain)})
      · ${fh.n_out == null ? "—" : fh.n_out} changes</p>
    <div class="fh-meter-bar" aria-hidden="true"><span style="width:${w}%"></span></div>
    ${fh.reason ? `<p class="stat-note">${fh.reason}</p>` : ""}
  </div>`;
}

function chipScanHtml(scan, template) {
  const weeks = (scan && scan.weeks) || [];
  const best = (scan && scan.best) || [];
  const fh = freehitMeter((scan && scan.freehit) || null);
  const rec = best.length ? `<p>${best.map((b) => b.reason).join(" · ")}</p>` : "";
  const table = weeks.length
    ? `<table><thead><tr><th>GW</th><th>XI</th><th>C</th><th>C EV</th><th>3XC</th><th>bench</th><th>BB</th></tr></thead><tbody>${weeks
        .map(
          (w) =>
            `<tr><td>${w.event || w.offset}</td><td>${fmt(w.xi, 1)}</td><td>${w.captain || "—"}</td><td>${fmt(w.c_ev, 1)}</td><td>${fmt(w.triple_ev, 1)}</td><td>${fmt(w.bench, 1)}</td><td>${fmt(w.bb, 1)}</td></tr>`
        )
        .join("")}</tbody></table>`
    : "";
  const tmpl = template
    ? `<p>Template overlap ${template.overlap}/${template.template_n || 15} · your XI owned ${fmt(template.you_owned_avg, 1)}% vs template ${fmt(template.template_owned_avg, 1)}%${
        (template.differentials || []).length
          ? ` · diffs: ${template.differentials.map((d) => `${d.web_name} ${fmt(d.selected_by_percent, 1)}%`).join(", ")}`
          : ""
      }</p>`
    : "";
  if (!fh && !rec && !table && !tmpl) return "";
  return `<div class="chip-scan">${fh}${rec}${tmpl}${table}</div>`;
}

async function scanChips() {
  saveMyTeam();
  if (state.myTeam.length !== 15) {
    $("transfer-board").innerHTML = `<p class="empty">Need a full 15-man squad to scan chips.</p>`;
    return;
  }
  const params = new URLSearchParams({
    season: state.season,
    as_of: String(state.asOf),
    squad: state.myTeam.join(","),
  });
  if (state.chipsUsed && state.chipsUsed.length) params.set("chips_used", state.chipsUsed.join(","));
  const data = await (await fetch(`/api/chips?${params}`)).json();
  $("transfer-board").innerHTML = data.ok
    ? chipScanHtml(data)
    : `<p class="empty">${data.error || "Could not scan chips."}</p>`;
}

async function optimiseLineup(chip) {
  saveMyTeam();
  const seq = ++state.teamViewSeq;
  if (state.myTeam.length !== 15) {
    $("team-board").innerHTML = `<p class="empty">Need a full 15-man squad (currently ${state.myTeam.length}).</p>`;
    return;
  }
  $("team-board").innerHTML = `<p class="empty">Picking GW XI…</p>`;
  const params = new URLSearchParams({
    season: state.season,
    as_of: String(state.asOf),
    squad: state.myTeam.join(","),
  });
  const ev = $("team-event") && $("team-event").value;
  if (ev) params.set("event", ev);
  if (chip) params.set("chip", chip);
  try {
    const res = await fetch(`/api/lineup?${params}`);
    const text = await res.text();
    let data;
    try {
      data = JSON.parse(text);
    } catch {
      throw new Error(
        res.status === 404
          ? "Restart fpl-model ui — this server has no /api/lineup yet."
          : `HTTP ${res.status}`
      );
    }
    if (seq !== state.teamViewSeq) return;
    renderLineup(data);
  } catch (err) {
    if (seq !== state.teamViewSeq) return;
    $("team-board").innerHTML = `<p class="empty">${err.message || "Could not pick a starting XI."}</p>`;
  }
}

async function importFplTeam() {
  const entry = $("team-entry").value;
  if (!entry) {
    $("team-board").innerHTML = `<p class="empty">Enter the number from fantasy.premierleague.com/entry/THIS/…</p>`;
    return;
  }
  $("team-board").innerHTML = `<p class="empty">Loading FPL team ${entry}…</p>`;
  const data = await (await fetch(`/api/fpl-team?entry=${encodeURIComponent(entry)}`)).json();
  if (!data.ok) {
    $("team-board").innerHTML = `<p class="empty">${data.error || "Could not import team."}</p>`;
    return;
  }
  state.myTeam = data.ids.map(Number);
  state.chipsUsed = data.chips_used || [];
  $("team-bank").value = String(data.bank ?? 0);
  $("team-ft").value = String(data.free_transfers ?? 1);
  state.plan = null;
  state.transferOut = null;
  saveMyTeam();
  $("team-summary").textContent = `${data.name || "Team"} imported · ${data.chips_used && data.chips_used.length ? "chips used: " + data.chips_used.join(", ") : "no chips used yet"}`;
  renderMyTeam();
}

async function suggestTransfers() {
  saveMyTeam();
  if (state.myTeam.length !== 15) {
    $("transfer-board").innerHTML =
      transferPickerShell() +
      `<p class="empty">Need a full 15-man squad (currently ${state.myTeam.length}).</p>`;
    refreshXferCands();
    return;
  }
  $("transfer-board").innerHTML = `<p class="empty">Finding transfers…</p>`;
  const baseline = state.plan ? state.plan.ids : state.myTeam;
  const bank = state.plan ? state.plan.bank : $("team-bank").value || "0";
  const params = new URLSearchParams({
    season: state.season,
    as_of: String(state.asOf),
    squad: baseline.join(","),
    bank: String(bank),
    budget: ($("team-budget") && $("team-budget").value) || "100",
    ft: $("team-ft").value || "1",
    want: ($("team-want") && $("team-want").value) || "2",
    objective: $("team-objective").value,
  });
  if (state.plan && state.plan.moves.length) {
    params.set("moves", state.plan.moves.map((m) => `${m.out}-${m.in}`).join(","));
  }
  if (state.exclude.size) params.set("exclude", [...state.exclude].join(","));
  if (state.excludeTeams.size) params.set("exclude_teams", [...state.excludeTeams].join(","));
  if (state.include.size) params.set("include", [...state.include].join(","));
  if (state.teamLock.size) params.set("keep", [...state.teamLock].join(","));
  const data = await (await fetch(`/api/transfers?${params}`)).json();
  if (!data.ok) {
    $("transfer-board").innerHTML = `<p class="empty">${data.error || "Could not suggest transfers."}</p>`;
    return;
  }
  if (data.bank_now != null && $("team-bank")) $("team-bank").value = String(data.bank_now);
  const picker = transferPickerShell();
  const committedHtml = committedCardHtml(data);
  const restore = data.restore || {};
  const capNote =
    restore.needed
      ? `<p class="empty" style="padding:0 0 .6rem">${restore.reason}</p>`
      : data.overspend > 0.05
        ? `<p class="empty" style="padding:0 0 .6rem">£${fmt(data.spent_now != null ? data.spent_now : data.spent)} vs max £${fmt(data.budget)} (£${fmt(data.overspend)} over).</p>`
        : data.bank_now < -0.05
          ? `<p class="empty" style="padding:0 0 .6rem">Bank £${fmt(data.bank_now)}.</p>`
          : "";
  const ftNote = data.plan
    ? `<p class="empty" style="padding:0 0 .6rem">${data.plan.action === "bank" ? "Bank FT: " : ""}${data.plan.reason}</p>`
    : "";
  const chips = (data.chips || []).length
    ? `<p class="empty" style="padding:0 0 .6rem">${data.chips.map((c) => `${c.chip}: ${c.reason}`).join(" · ")}</p>`
    : "";
  const scanNote = data.chip_scan ? chipScanHtml(data.chip_scan) : "";
  let body;
  if (!data.suggestions.length) {
    const over =
      restore.needed && restore.reason
        ? restore.reason
        : data.overspend > 0.05
          ? `Squad is £${fmt(data.spent_now != null ? data.spent_now : data.spent)} vs max £${fmt(data.budget)} (£${fmt(data.overspend)} over). No path under the cap.`
          : "No improving transfers within bank / squad cap / team caps.";
    body = `<p class="empty">${over}</p>${ftNote}${chips}${scanNote}`;
  } else {
    const obj = data.objective;
    body =
      `<h3 style="margin:0 0 .7rem;font-size:.95rem">Suggestions (squad ${obj} ${fmt(data.squad_objective_now != null ? data.squad_objective_now : data.squad_objective, 1)})</h3>` +
      capNote +
      ftNote +
      chips +
      scanNote +
      data.suggestions.map((s) => transferCardHtml(s, data.committed ? "Add to plan" : "Apply")).join("");
  }
  $("transfer-board").innerHTML = picker + committedHtml + body;
  refreshXferCands();
}

function transferPickerShell() {
  if (!state.transferOut) {
    return `<p class="empty" style="padding:0 0 .55rem">Click <b>Out</b> on a player to make a manual transfer. <b>Transfers</b> is extra moves to search after manuals; extra vs remaining FT cost −4 xPts each.</p>`;
  }
  const sell = state.cache.get(state.transferOut);
  const name = sell ? sell.web_name : "player";
  const q = (state.xferQ || "").replace(/"/g, "&quot;");
  return `<div class="transfer-card">
    <div>Replace <b>${name}</b>
      <input id="xfer-search" type="search" placeholder="Search replacement" value="${q}" />
    </div>
    <div id="xfer-cands" class="xfer-cands"></div>
    <div class="card-actions"><button type="button" data-transfer-out="${state.transferOut}">Cancel</button></div>
  </div>`;
}

async function refreshXferCands() {
  const box = $("xfer-cands");
  if (!box || !state.transferOut) return;
  let sell = state.cache.get(state.transferOut);
  if (!sell) {
    const loaded = await loadPlayersByIds([state.transferOut]);
    sell = loaded[0];
  }
  if (!sell) {
    box.innerHTML = `<p class="empty">Could not load that player.</p>`;
    return;
  }
  const obj = $("team-objective").value;
  const cands = await fetchPlayers({
    position: sell.position,
    q: state.xferQ || "",
    sort: obj === "overall" ? "overall" : "xpts",
    limit: 40,
  });
  const inSquad = new Set(state.myTeam);
  const rows = cands
    .filter((p) => !inSquad.has(p.fpl_id))
    .slice(0, 16)
    .map((p) => {
      const dx = Number(p.xpts || 0) - Number(sell.xpts || 0);
      const dov = Number(p.overall || 0) - Number(sell.overall || 0);
      const net = Number(p.price || 0) - Number(sell.price || 0);
      return `<div class="xfer-cand">
        <button type="button" data-bring="${p.fpl_id}">${p.web_name}</button>
        <span>${p.team || "—"} · £${fmt(p.price)} (${net >= 0 ? "+" : ""}£${fmt(net)}) · xPts ${signed(dx)} · Overall ${signed(dov)}</span>
      </div>`;
    });
  box.innerHTML = rows.length ? rows.join("") : `<p class="empty">No other ${sell.position} replacements.</p>`;
}

function committedCardHtml(data) {
  const c = data.committed;
  if (!c) return "";
  const notes = (c.notes || []).length
    ? `<div class="transfer-move">${c.notes.map((n) => `⚠ ${n}`).join(" · ")}</div>`
    : "";
  return `<div class="transfer-card">
    <div><b>Your plan</b> · ${scoreLine(c)}</div>
    ${movesHtml(c.moves)}
    ${notes}
    <div class="card-actions">
      <button type="button" data-undo-plan>Undo last</button>
      <button type="button" data-reset-plan>Reset plan</button>
    </div>
  </div>`;
}

function scoreLine(s) {
  const hits = s.hits ? ` · ${s.hits} hit (−${4 * s.hits} xPts)` : "";
  const capBit = s.fixes_budget
    ? ` · under cap £${fmt(s.spent_after)}`
    : s.spent_after != null
      ? ` · spent £${fmt(s.spent_after)}`
      : "";
  const hz = s.horizon_gws ? ` · ${s.horizon_gws}GW ${signed(s.gain_horizon)}` : "";
  const xiBit =
    s.xi_after && s.xi_after.formation
      ? ` · XI ${s.xi_after.formation} ${fmt(s.xi_after.xi_objective, 1)} (${signed(s.xi_gain)})`
      : "";
  return `<span class="${s.net_gain >= 0 ? "gain" : "gain neg"}">${signed(s.net_gain)}</span>
    xPts ${signed(s.gain_xpts)} (next ${signed(s.gain_next)}${hz}) · Overall ${signed(s.gain_overall)}${hits}
    · ${s.transfers} transfer${s.transfers === 1 ? "" : "s"} · cost £${fmt(s.cost)} · bank £${fmt(s.bank_after)}${capBit}${xiBit}`;
}

function movesHtml(moves) {
  return (moves || [])
    .map(
      (m) =>
        `<div class="transfer-move"><b>${m.out.web_name}</b> (£${fmt(m.out.price)}) → <b>${m.in.web_name}</b> (£${fmt(m.in.price)}) · ${m.in.team || ""}</div>`
    )
    .join("");
}

function transferCardHtml(s, applyLabel) {
  const shown = s.combined || s;
  const extra = s.combined ? `<div class="transfer-move" style="margin-top:.35rem">Add:</div>${movesHtml(s.moves)}` : movesHtml(s.moves);
  const xiNames =
    s.xi_after && s.xi_after.xi && s.xi_after.xi.length
      ? `<div class="transfer-move">XI after: ${s.xi_after.xi.join(", ")}${s.xi_after.captain ? ` · C ${s.xi_after.captain}` : ""}</div>`
      : "";
  const notes = (s.notes || []).length
    ? `<div class="transfer-move" style="margin-top:.25rem">${s.notes.map((n) => `⚠ ${n}`).join(" · ")}</div>`
    : "";
  return `<div class="transfer-card">
    <div>${scoreLine(shown)}</div>
    ${extra}
    ${xiNames}
    ${notes}
    <div class="card-actions" style="margin-top:.5rem">
      <button type="button" data-apply='${JSON.stringify(s.moves.map((m) => [m.out.fpl_id, m.in.fpl_id]))}' data-bank-after="${s.bank_after}">${applyLabel}</button>
    </div>
  </div>`;
}

function applyMoves(pairs, bankAfter) {
  if (!state.plan) {
    state.plan = { ids: [...state.myTeam], bank: Number($("team-bank").value || 0), moves: [] };
  }
  let ids = [...state.myTeam];
  for (const [outId, inId] of pairs) {
    const i = ids.indexOf(Number(outId));
    if (i >= 0) ids[i] = Number(inId);
    state.plan.moves.push({ out: Number(outId), in: Number(inId) });
  }
  state.myTeam = ids;
  state.transferOut = null;
  if (bankAfter != null && bankAfter !== "" && $("team-bank")) $("team-bank").value = String(bankAfter);
  saveMyTeam();
  renderMyTeam();
  suggestTransfers();
}

function startTransferOut(id) {
  id = Number(id);
  state.transferOut = state.transferOut === id ? null : id;
  state.xferQ = "";
  renderMyTeam();
  if (state.selected === id && (state.tab === "players" || state.tab === "shortlist")) selectPlayer(id);
  suggestTransfers();
}

function bringIn(id) {
  id = Number(id);
  if (!state.transferOut) return;
  applyMoves([[state.transferOut, id]]);
}

function undoPlan() {
  if (!state.plan || !state.plan.moves.length) return;
  state.plan.moves.pop();
  state.myTeam = applyPairs(state.plan.ids, state.plan.moves);
  if (!state.plan.moves.length) {
    if ($("team-bank")) $("team-bank").value = String(state.plan.bank);
    state.plan = null;
  }
  state.transferOut = null;
  saveMyTeam();
  renderMyTeam();
  suggestTransfers();
}

function resetPlan() {
  if (state.plan) {
    state.myTeam = [...state.plan.ids];
    if ($("team-bank")) $("team-bank").value = String(state.plan.bank);
  }
  state.plan = null;
  state.transferOut = null;
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
  const head = events.map((e) => `<th>GW${e}</th>`).join("");
  const body = teams
    .map((t) => {
      const cells = events
        .map((e) => {
          const f = t.fixtures.find((x) => x.event === e);
          if (!f) return `<td>-</td>`;
          const cls = [fixtureTone(f, "attack"), f.thin_venue ? "thin" : ""].filter(Boolean).join(" ");
          return `<td><span class="fx-chip ${cls}" style="display:inline-block;min-width:3.2rem" title="${fixtureTitle(f)}">${f.label}${f.thin_venue ? "*" : ""}</span></td>`;
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
      <thead><tr><th>Team</th><th>Avg O</th><th>Att</th>${head}</tr></thead>
      <tbody>${body}</tbody>
    </table>
    <p class="empty" style="padding:.8rem 0 0">Five colours match the 1–5 rating (dark green = 1 very easy … dark red = 5 very hard). Att ease = opponent leak × your team’s venue xG. Hover for details. High Avg O = model likes the run. * = thin venue sample.</p>`;
  renderCalibrate();
}

async function renderCalibrate() {
  const box = $("calibrate-board");
  if (!box) return;
  box.innerHTML = "";
  try {
    const params = new URLSearchParams({ season: state.season, as_of: String(state.asOf) });
    const data = await (await fetch(`/api/calibrate?${params}`)).json();
    if (data.empty) {
      box.innerHTML = `<p class="empty" style="padding:0">Calibration: ${data.reason || "no stored GW actuals yet. Run fpl-model update after a GW finishes."}</p>`;
      return;
    }
    const rows = (data.by_position || []).map(
      (r) =>
        `<tr><td>${r.position || r.label}</td><td>${r.n}</td><td>${fmt(r.mae, 2)}</td><td>${signed(r.bias)}</td></tr>`
    );
    const extra = [...(data.by_venue || []), ...(data.by_opponent || [])]
      .map((r) => `<tr><td>${r.label}</td><td>${r.n}</td><td>${fmt(r.mae, 2)}</td><td>${signed(r.bias)}</td></tr>`)
      .join("");
    const bins = (data.table || [])
      .map((r) => {
        const bias =
          r.bias != null
            ? r.bias
            : r.pred_mean != null && r.actual_mean != null
              ? r.pred_mean - r.actual_mean
              : null;
        return `<tr><td>${r.bin != null ? r.bin : "bin"}</td><td>${r.n}</td><td>${fmt(r.pred_mean, 2)}</td><td>${fmt(r.actual_mean, 2)}</td><td>${signed(bias)}</td></tr>`;
      })
      .join("");
    const binTable = bins
      ? `<p class="empty" style="padding:.45rem 0 .2rem">Reliability (predicted xPts quantile vs actual)</p>
         <table class="calibrate-table"><thead><tr><th>bin</th><th>n</th><th>pred</th><th>actual</th><th>bias</th></tr></thead><tbody>${bins}</tbody></table>`
      : `<p class="empty" style="padding:.45rem 0 0">Need more GWs before quantile bins fill in.</p>`;
    box.innerHTML = `<div>
      <p class="empty" style="padding:0 0 .35rem">xPts-next vs actual GW${data.as_of_gw} · n=${data.n} · MAE ${fmt(data.mae, 2)} · bias ${signed(data.bias)} · this-GW cards use −½ last bias (clipped)</p>
      <table class="calibrate-table"><thead><tr><th></th><th>n</th><th>MAE</th><th>bias</th></tr></thead><tbody>${rows.join("")}${extra}</tbody></table>
      ${binTable}
    </div>`;
  } catch {
    box.innerHTML = "";
  }
}

async function rebuildChip(chip) {
  if (state.myTeam.length !== 15) {
    $("transfer-board").innerHTML = `<p class="empty">Need a full 15-man squad for ${chip}.</p>`;
    return;
  }
  $("transfer-board").innerHTML = `<p class="empty">Building ${chip} 15…</p>`;
  const params = new URLSearchParams({
    season: state.season,
    as_of: String(state.asOf),
    squad: state.myTeam.join(","),
    chip,
    bank: $("team-bank").value || "0",
    budget: ($("team-budget") && $("team-budget").value) || "100",
    objective: $("team-objective").value,
  });
  applyBuilderConstraints(params);
  const data = await (await fetch(`/api/rebuild?${params}`)).json();
  if (!data.ok) {
    $("transfer-board").innerHTML = `<p class="empty">${data.error || "Could not rebuild squad."}</p>`;
    return;
  }
  const names = (data.squad || []).map((p) => p.web_name).join(", ");
  $("transfer-board").innerHTML = `<div class="transfer-card">
    <div><b>${chip === "freehit" ? "Free hit" : "Wildcard"} 15</b> · £${fmt(data.spent)} · bank £${fmt(data.bank_after)} · XI ${data.formation || ""} ${fmt(data.xi_objective, 1)} · squad ${signed(data.gain)} · XI ${signed(data.xi_gain)}</div>
    <div class="transfer-move">${names}</div>
    <div class="card-actions"><button type="button" data-apply-rebuild='${JSON.stringify(data.ids)}' data-bank-after="${data.bank_after}">Apply ${chip === "freehit" ? "FH" : "WC"} 15</button></div>
  </div>`;
}

async function loadWildcardAdvice() {
  saveMyTeam();
  const board = $("wildcard-board");
  if (!board) return;
  if (state.myTeam.length !== 15) {
    board.innerHTML = `<p class="empty">Need a full 15-man squad for wildcard advice (currently ${state.myTeam.length}).</p>`;
    return;
  }
  board.innerHTML = `<p class="empty">Scoring rest-of-season look-aheads and building wildcard 15s… this can take a bit.</p>`;
  const params = new URLSearchParams({
    season: state.season,
    as_of: String(state.asOf),
    squad: state.myTeam.join(","),
    bank: $("team-bank").value || "0",
    budget: ($("team-budget") && $("team-budget").value) || "100",
    ft: $("team-ft").value || "1",
  });
  applyBuilderConstraints(params);
  if (state.chipsUsed && state.chipsUsed.length) params.set("chips", state.chipsUsed.join(","));
  try {
    const res = await fetch(`/api/wildcard?${params}`);
    const data = await res.json();
    renderWildcardAdvice(data);
  } catch (err) {
    board.innerHTML = `<p class="empty">${err.message || "Wildcard advice failed."}</p>`;
  }
}

function renderWildcardAdvice(data) {
  const board = $("wildcard-board");
  if (!board) return;
  if (!data.ok) {
    board.innerHTML = `<p class="empty">${data.error || "Could not score a wildcard."}</p>`;
    return;
  }
  const v = data.verdict || "consider";
  const rows = (data.windows || [])
    .map(
      (w) => `<tr class="${w.rest_of_season ? "wc-ros" : ""}">
        <td>${w.label || w.gws + " GW"}</td>
        <td>${fmt(w.current_xi, 1)}</td>
        <td>${fmt(w.wc_xi, 1)}</td>
        <td>${signed(w.gain)}</td>
        <td>${signed(w.gain_adj)}</td>
        <td>${fmt(w.ft_path_xi, 1)}</td>
        <td>${signed(w.ft_gap)}</td>
        <td class="wc-conf">${w.confidence || ""}</td>
      </tr>`
    )
    .join("");
  const reasons = (data.reasons || []).map((r) => `<li>${r}</li>`).join("");
  const sells = (data.must_sell || [])
    .map((p) => `<span class="chip"><span class="nm">${p.web_name}</span><span class="dt">${p.reason}</span></span>`)
    .join("");
  const near = (data.stress || []).slice(0, 8);
  const later = (data.stress || [])
    .slice(8)
    .filter((s) => s.blanks >= 2)
    .sort((a, b) => b.blanks - a.blanks)
    .slice(0, 4);
  const stressBits = [...near, ...later.filter((s) => !near.some((n) => n.event === s.event))];
  const stress = stressBits
    .map((s) => `<span class="fx-chip ${s.blanks >= 3 ? "hard" : s.blanks >= 2 ? "mid" : "easy"}"><span class="gw">GW${s.event}</span>${fmt(s.xi_xpts, 1)}${s.blanks ? ` · ${s.blanks} weak` : ""}</span>`)
    .join("");
  const delayRow = (data.delays || [])
    .map(
      (d) =>
        `<span class="fx-chip ${d.best ? "easy" : "mid"}">${d.delay === 0 ? "Now" : "GW" + d.play_event}${d.best ? " · best" : ""} ${signed(d.delta_vs_now)}</span>`
    )
    .join("");
  const stints = (data.stints || [])
    .map(
      (s) =>
        `<span class="chip"><span class="nm">${s.web_name}</span><span class="dt">${s.reason || "stint"} · £${fmt(s.price)}</span></span>`
    )
    .join("");
  const names = (data.squad || []).map((p) => p.web_name).join(", ");
  const apply =
    data.ids && data.ids.length === 15 && v !== "used"
      ? `<div class="card-actions"><button type="button" data-apply-rebuild='${JSON.stringify(data.ids)}' data-bank-after="${data.bank_after || 0}">Apply WC 15</button></div>`
      : "";
  const waitBit = data.wait_until_gw ? ` · hold until GW${data.wait_until_gw}` : "";
  const extra =
    data.only_long && data.only_long.length
      ? `<p class="stat-note">8GW core names not in this 15: ${data.only_long.join(", ")}</p>`
      : "";
  board.innerHTML = `
    <div class="wc-verdict ${v}">
      <b>${data.headline || "Wildcard"}</b>${waitBit}
      <ul class="wc-reasons">${reasons}</ul>
    </div>
    <p class="stat-note">${data.note || ""} · ${data.n_out != null ? data.n_out + " moves vs this WC 15" : ""}</p>
    <table class="wc-table">
      <thead><tr><th>Look-ahead</th><th>Your XI</th><th>WC XI</th><th>Gain</th><th>Adj gain</th><th>FT-only</th><th>WC vs FTs</th><th>Trust</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <p class="stat-note">WC XI is this fixed 15, week by week (stints stay in the squad; later FTs are not scripted). Play/wait uses ~5 GWs. 8+ GW is core shape. FT-only = 1 transfer/week from your current 15, no hits.</p>
    ${delayRow ? `<h4 class="wc-h">When to fire (weighted XI vs playing now)</h4><div class="fx-row">${delayRow}</div>` : ""}
    ${stress ? `<h4 class="wc-h">Your XI by week</h4><div class="fx-row">${stress}</div>` : ""}
    ${sells ? `<h4 class="wc-h">Likely outs</h4><div class="xi-line" style="justify-content:flex-start">${sells}</div>` : ""}
    ${stints ? `<h4 class="wc-h">Fixture stints in this 15</h4><div class="xi-line" style="justify-content:flex-start">${stints}</div>` : ""}
    ${extra}
    ${names ? `<div class="transfer-card" style="margin-top:.8rem"><div><b>Wildcard 15</b> · ${data.hold_gws || 3}GW hold · ${data.core_gws || 8}GW core · £${fmt(data.spent)}${data.max_value != null ? ` / max £${fmt(data.max_value)}` : ""} · bank £${fmt(data.bank_after)} · ${data.formation || ""}</div><div class="transfer-move">${names}</div>${apply}</div>` : ""}
  `;
}

function applyBuilderConstraints(params) {
  if (state.exclude.size) params.set("exclude", [...state.exclude].join(","));
  if (state.excludeTeams.size) params.set("exclude_teams", [...state.excludeTeams].join(","));
  if (state.include.size) params.set("include", [...state.include].join(","));
  if (state.lock.size) params.set("lock", [...state.lock].join(","));
  if (state.teamLock.size) params.set("keep", [...state.teamLock].join(","));
  const mv = $("wc-max-value") && $("wc-max-value").value;
  if (mv) params.set("max_value", mv);
}

function applyRebuild(ids, bankAfter) {
  state.myTeam = ids.map(Number);
  state.plan = null;
  state.transferOut = null;
  if (bankAfter != null && $("team-bank")) $("team-bank").value = String(bankAfter);
  saveMyTeam();
  renderMyTeam();
}

function setTab(tab) {
  state.tab = tab;
  document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("on", b.dataset.tab === tab));
  ["players", "shortlist", "xi", "team", "fixtures"].forEach((name) => {
    $(`view-${name}`).classList.toggle("hidden", tab !== name);
  });
  if (tab === "shortlist") renderShortlist();
  if (tab === "xi") renderExcludePanel();
  if (tab === "xi" && !$("xi-board").dataset.ready) {
    buildXi().then(() => {
      $("xi-board").dataset.ready = "1";
    });
  }
  if (tab === "team") {
    renderMyTeam();
    renderExcludePanel();
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
  const applyRebuildBtn = e.target.closest("[data-apply-rebuild]");
  if (applyRebuildBtn) {
    applyRebuild(JSON.parse(applyRebuildBtn.dataset.applyRebuild), applyRebuildBtn.dataset.bankAfter);
    return;
  }
  const apply = e.target.closest("[data-apply]");
  if (apply) {
    applyMoves(JSON.parse(apply.dataset.apply), apply.dataset.bankAfter);
    return;
  }
  const undo = e.target.closest("[data-undo-plan]");
  if (undo) {
    undoPlan();
    return;
  }
  const reset = e.target.closest("[data-reset-plan]");
  if (reset) {
    resetPlan();
    return;
  }
  const bring = e.target.closest("[data-bring]");
  if (bring) {
    bringIn(bring.dataset.bring);
    return;
  }
  const outBtn = e.target.closest("[data-transfer-out]");
  if (outBtn) {
    startTransferOut(outBtn.dataset.transferOut);
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
  const banBtn = e.target.closest("[data-exclude]");
  if (banBtn) {
    toggleExclude(banBtn.dataset.exclude, e);
    return;
  }
  const clubBan = e.target.closest("[data-exclude-team]");
  if (clubBan) {
    toggleExcludeTeam(clubBan.dataset.excludeTeam, e);
    return;
  }
  const keepBtn = e.target.closest("[data-keep]");
  if (keepBtn) {
    toggleKeep(keepBtn.dataset.keep, e);
    return;
  }
  const lockBtn = e.target.closest("[data-lock]");
  if (lockBtn) {
    toggleLock(lockBtn.dataset.lock, e);
    return;
  }
  const compareBtn = e.target.closest("[data-compare]");
  if (compareBtn) {
    const cid = Number(compareBtn.dataset.compare);
    state.compareId = Number(state.compareId) === cid ? null : cid;
    if (state.selected) selectPlayer(state.selected);
    return;
  }
  const row = e.target.closest(".row");
  if (row) selectPlayer(Number(row.dataset.id));
  const chipEl = e.target.closest(".chip[data-id]");
  if (chipEl) {
    if (state.tab === "players" || state.tab === "shortlist") selectPlayer(Number(chipEl.dataset.id));
    return;
  }
}

$("rows").addEventListener("click", onRowClick);
$("short-rows").addEventListener("click", onRowClick);
$("card").addEventListener("click", onRowClick);
$("xi-board").addEventListener("click", onRowClick);
$("xi-exclude-list").addEventListener("click", onRowClick);
if ($("xi-exclude-teams")) $("xi-exclude-teams").addEventListener("click", onRowClick);
if ($("xi-include-list")) $("xi-include-list").addEventListener("click", onRowClick);
if ($("xi-lock-list")) $("xi-lock-list").addEventListener("click", onRowClick);
$("team-board").addEventListener("click", onRowClick);
$("transfer-board").addEventListener("click", onRowClick);
if ($("wildcard-board")) $("wildcard-board").addEventListener("click", onRowClick);
if ($("wc-constraints")) $("wc-constraints").addEventListener("click", onRowClick);
$("transfer-board").addEventListener("input", (e) => {
  if (e.target.id !== "xfer-search") return;
  state.xferQ = e.target.value;
  clearTimeout(state.xferT);
  state.xferT = setTimeout(refreshXferCands, 150);
});

$("tabs").addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-tab]");
  if (btn) setTab(btn.dataset.tab);
});

$("season").addEventListener("change", () => {
  state.season = $("season").value;
  fillAsOf();
  fillClubs();
  loadShortlist();
  loadExclude();
  loadExcludeTeams();
  loadInclude();
  loadLock();
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
  setHorizonNote();
  fillTeamEvent();
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
if ($("club")) {
  $("club").addEventListener("change", () => {
    state.team = $("club").value;
    loadPlayers();
  });
}
let t;
const onSearch = () => {
  clearTimeout(t);
  t = setTimeout(loadPlayers, 150);
};
$("search").addEventListener("input", onSearch);
$("search").addEventListener("change", onSearch);
$("xi-build").addEventListener("click", buildXi);
$("xi-save-team").addEventListener("click", saveXiAsTeam);
if ($("xi-exclude-club-add")) {
  $("xi-exclude-club-add").addEventListener("click", () => {
    const sel = $("xi-exclude-club");
    if (!sel || !sel.value) return;
    const tid = Number(sel.value);
    if (!state.excludeTeams.has(tid)) toggleExcludeTeam(tid);
  });
}
if ($("wc-exclude-club-add")) {
  $("wc-exclude-club-add").addEventListener("click", () => {
    const sel = $("wc-exclude-club");
    if (!sel || !sel.value) return;
    const tid = Number(sel.value);
    if (!state.excludeTeams.has(tid)) toggleExcludeTeam(tid);
  });
}
if ($("wc-player-q")) {
  $("wc-player-q").addEventListener("input", () => {
    clearTimeout(state.wcT);
    state.wcT = setTimeout(refreshWcCands, 150);
  });
}
$("team-import").addEventListener("click", importFplTeam);
$("team-lineup").addEventListener("click", optimiseLineup);
if ($("team-event")) {
  $("team-event").addEventListener("change", () => {
    if (state.tab === "team" && state.myTeam.length === 15) optimiseLineup();
  });
}
$("team-suggest").addEventListener("click", suggestTransfers);
if ($("team-wildcard")) $("team-wildcard").addEventListener("click", () => rebuildChip("wildcard"));
if ($("team-wildcard-advice")) $("team-wildcard-advice").addEventListener("click", loadWildcardAdvice);
if ($("team-freehit")) $("team-freehit").addEventListener("click", () => rebuildChip("freehit"));
if ($("team-bboost")) $("team-bboost").addEventListener("click", () => optimiseLineup("bboost"));
if ($("team-3xc")) $("team-3xc").addEventListener("click", () => optimiseLineup("3xc"));
if ($("team-chips")) $("team-chips").addEventListener("click", scanChips);
$("team-clear").addEventListener("click", () => {
  state.myTeam = [];
  state.teamLock = new Set();
  state.plan = null;
  state.transferOut = null;
  saveMyTeam();
  renderMyTeam();
  $("transfer-board").innerHTML = `<p class="empty">Transfer suggestions appear here.</p>`;
  if ($("wildcard-board"))
    $("wildcard-board").innerHTML = `<p class="empty">Wildcard advice builds a 15 for the next 3–5 GWs, with a soft 8GW core. Later weeks are left to ordinary FTs.</p>`;
});
["team-bank", "team-budget", "team-ft", "team-want", "team-entry", "wc-max-value"].forEach((id) => {
  const el = $(id);
  if (el) el.addEventListener("change", saveMyTeam);
});
bindPills("pos-pills", "pos", "position");
bindPills("action-pills", "action", "action");
if ($("horizon-pills")) {
  $("horizon-pills").addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-horizon]");
    if (!btn) return;
    state.horizonMode = btn.dataset.horizon || "window";
    $("horizon-pills").querySelectorAll("button").forEach((b) => b.classList.toggle("on", b === btn));
    setHorizonNote();
    if (state.sort === "xpts" || state.sort === "xpts_next") {
      state.sort = state.horizonMode === "next" ? "xpts_next" : "xpts";
      if ($("sort")) $("sort").value = state.sort;
    }
    if (state.tab === "players") loadPlayers();
    else if (state.tab === "shortlist") renderShortlist();
  });
}

loadMeta().then(loadPlayers);
