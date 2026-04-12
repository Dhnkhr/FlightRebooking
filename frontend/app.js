const state = {
  tasks: [],
  selectedTask: "easy",
  sessionId: "default",
  observation: null,
  done: false,
  grade: null,
  logs: [],
  latest: null,
  showRawLatest: false,
};

const refs = {
  taskSelect: document.getElementById("taskSelect"),
  resetBtn: document.getElementById("resetBtn"),
  autoBtn: document.getElementById("autoBtn"),
  finalizeBtn: document.getElementById("finalizeBtn"),
  suggestBtn: document.getElementById("suggestBtn"),
  runStepBtn: document.getElementById("runStepBtn"),
  clearLogBtn: document.getElementById("clearLogBtn"),
  actionForm: document.getElementById("actionForm"),
  actionType: document.getElementById("actionType"),
  passengerId: document.getElementById("passengerId"),
  flightId: document.getElementById("flightId"),
  sessionBadge: document.getElementById("sessionBadge"),
  phaseBadge: document.getElementById("phaseBadge"),
  scoreBadge: document.getElementById("scoreBadge"),
  taskMeta: document.getElementById("taskMeta"),
  budgetRemaining: document.getElementById("budgetRemaining"),
  budgetSpent: document.getElementById("budgetSpent"),
  progressValue: document.getElementById("progressValue"),
  invalidValue: document.getElementById("invalidValue"),
  stepCountBadge: document.getElementById("stepCountBadge"),
  pendingList: document.getElementById("pendingList"),
  flightsList: document.getElementById("flightsList"),
  latestResult: document.getElementById("latestResult"),
  latestRaw: document.getElementById("latestRaw"),
  toggleRawBtn: document.getElementById("toggleRawBtn"),
  logList: document.getElementById("logList"),
};

function money(value) {
  return `$${Number(value || 0).toFixed(2)}`;
}

function safeJson(data) {
  return JSON.stringify(data, null, 2);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function latestRow(label, value) {
  return `
    <div class="latest-row">
      <span class="latest-label">${escapeHtml(label)}</span>
      <span class="latest-value">${escapeHtml(value)}</span>
    </div>
  `;
}

function buildLatestSummary() {
  const data = state.latest;
  if (!data) {
    return '<div class="latest-empty">No actions yet.</div>';
  }

  if (data.error) {
    return `
      <div class="latest-status latest-status-error">Action failed</div>
      <div class="latest-note latest-note-error">${escapeHtml(data.error)}</div>
    `;
  }

  if (data.event === "reset") {
    return `
      <div class="latest-status latest-status-reset">Session reset</div>
      ${latestRow("Task", data.task || state.selectedTask)}
      ${latestRow("Session", data.session_id || state.sessionId)}
      <div class="latest-note">Environment is ready for your first step.</div>
    `;
  }

  const summary = [];
  const observation = data.observation || {};
  const rewardValue = Number(data.reward?.value ?? 0);
  const rewardNotes = Array.isArray(data.reward?.notes) ? data.reward.notes.filter(Boolean) : [];
  const latestAction = state.logs.length ? state.logs[state.logs.length - 1].action : null;

  summary.push(
    `<div class="latest-status ${data.done ? "latest-status-done" : "latest-status-active"}">${
      data.done ? "Step completed" : "Step applied"
    }</div>`
  );

  if (latestAction) {
    summary.push(latestRow("Action", latestAction.action_type));
    summary.push(latestRow("Passenger", latestAction.passenger_id || "na"));
    summary.push(latestRow("Flight", latestAction.flight_id || "na"));
  }

  summary.push(latestRow("Reward", rewardValue.toFixed(4)));
  summary.push(latestRow("Done", data.done ? "Yes" : "No"));

  if (typeof data.final_score === "number") {
    summary.push(latestRow("Final score", Number(data.final_score).toFixed(4)));
  }

  if (typeof observation.processed_count === "number" && typeof observation.total_passengers === "number") {
    summary.push(latestRow("Progress", `${observation.processed_count}/${observation.total_passengers}`));
  }

  if (Array.isArray(observation.pending_passengers)) {
    summary.push(latestRow("Pending left", String(observation.pending_passengers.length)));
  }

  if (typeof observation.budget_remaining === "number") {
    summary.push(latestRow("Budget remaining", money(observation.budget_remaining)));
  }

  if (typeof observation.budget_spent === "number") {
    summary.push(latestRow("Budget spent", money(observation.budget_spent)));
  }

  if (typeof observation.invalid_actions === "number") {
    summary.push(latestRow("Invalid actions", String(observation.invalid_actions)));
  }

  if (rewardNotes.length) {
    summary.push(`<div class="latest-note">Notes: ${escapeHtml(rewardNotes.join(" | "))}</div>`);
  }

  if (data.info?.warning) {
    summary.push(`<div class="latest-note">Warning: ${escapeHtml(data.info.warning)}</div>`);
  }

  if (data.info?.error) {
    summary.push(`<div class="latest-note latest-note-error">Error: ${escapeHtml(data.info.error)}</div>`);
  }

  if (Array.isArray(data.info?.unresolved_passengers) && data.info.unresolved_passengers.length) {
    summary.push(
      `<div class="latest-note">Unresolved passengers: ${escapeHtml(data.info.unresolved_passengers.join(", "))}</div>`
    );
  }

  return summary.join("");
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });

  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = payload.detail || `Request failed: ${response.status}`;
    throw new Error(detail);
  }
  return payload;
}

function taskByKey(taskKey) {
  return state.tasks.find((task) => task.task_key === taskKey);
}

function tierWeight(tier) {
  return { Platinum: 4, Gold: 3, Silver: 2, Standard: 1 }[tier] || 1;
}

function requiresFlight(actionType) {
  return ["rebook_passenger", "offer_downgrade", "rebook_on_partner"].includes(actionType);
}

function chooseSuggestion(observation) {
  const pending = [...(observation.pending_passengers || [])];
  if (!pending.length) return { action_type: "finalize" };

  pending.sort(
    (a, b) =>
      tierWeight(b.priority_tier) - tierWeight(a.priority_tier) ||
      (a.connection_deadline_hrs ?? 1e9) - (b.connection_deadline_hrs ?? 1e9)
  );

  const passenger = pending[0];
  const flights = [...(observation.available_flights || [])].sort(
    (a, b) => a.departure_hrs - b.departure_hrs
  );

  const hasSeat = (flight, cabinClass) =>
    cabinClass === "Business" ? flight.business_seats > 0 : flight.economy_seats > 0;

  for (const flight of flights) {
    if (!flight.is_partner && hasSeat(flight, passenger.cabin_class)) {
      return {
        action_type: "rebook_passenger",
        passenger_id: passenger.id,
        flight_id: flight.id,
      };
    }
  }

  if (passenger.cabin_class === "Business") {
    for (const flight of flights) {
      if (!flight.is_partner && flight.economy_seats > 0 && observation.budget_remaining >= 500) {
        return {
          action_type: "offer_downgrade",
          passenger_id: passenger.id,
          flight_id: flight.id,
        };
      }
    }
  }

  for (const flight of flights) {
    if (flight.is_partner && hasSeat(flight, passenger.cabin_class) && observation.budget_remaining >= 800) {
      return {
        action_type: "rebook_on_partner",
        passenger_id: passenger.id,
        flight_id: flight.id,
      };
    }
  }

  if (observation.budget_remaining >= 250) {
    return { action_type: "book_hotel", passenger_id: passenger.id };
  }

  return { action_type: "mark_no_solution", passenger_id: passenger.id };
}

function setSelectOptions(selectEl, options, selectedValue) {
  const current = selectedValue ?? selectEl.value;
  selectEl.innerHTML = "";
  options.forEach((option) => {
    const el = document.createElement("option");
    el.value = option.value;
    el.textContent = option.label;
    if (current && current === option.value) {
      el.selected = true;
    }
    selectEl.appendChild(el);
  });
}

function renderTaskMeta() {
  const task = taskByKey(state.selectedTask);
  if (!task) {
    refs.taskMeta.textContent = "No task loaded.";
    return;
  }

  refs.taskMeta.textContent = `${task.task_id} | ${task.difficulty.toUpperCase()} | ${task.passenger_count} passengers | Budget ${money(task.max_budget)}`;
}

function renderMetrics() {
  const obs = state.observation;
  if (!obs) {
    refs.budgetRemaining.textContent = "-";
    refs.budgetSpent.textContent = "-";
    refs.progressValue.textContent = "-";
    refs.invalidValue.textContent = "-";
    refs.stepCountBadge.textContent = "step 0";
    return;
  }

  refs.budgetRemaining.textContent = money(obs.budget_remaining);
  refs.budgetSpent.textContent = money(obs.budget_spent);
  refs.progressValue.textContent = `${obs.processed_count}/${obs.total_passengers}`;
  refs.invalidValue.textContent = String(obs.invalid_actions ?? 0);
  refs.stepCountBadge.textContent = `step ${obs.step_count ?? 0}`;
}

function renderPassengers() {
  const list = refs.pendingList;
  list.innerHTML = "";

  if (!state.observation?.pending_passengers?.length) {
    const empty = document.createElement("div");
    empty.className = "empty empty-passengers";
    empty.textContent = state.done ? "No pending passengers. Episode is complete." : "No pending passengers.";
    list.appendChild(empty);
    return;
  }

  state.observation.pending_passengers.forEach((passenger) => {
    const card = document.createElement("article");
    card.className = "card card-passenger";
    card.innerHTML = `
      <div class="card-top">
        <div class="card-title">${passenger.id} - ${passenger.name}</div>
        <span class="tier-pill tier-${passenger.priority_tier}">${passenger.priority_tier}</span>
      </div>
      <div class="card-sub">Cabin ${passenger.cabin_class} | Flight ${passenger.original_flight}</div>
      <div class="card-sub">Connection deadline: ${passenger.connection_deadline_hrs ?? "none"} hrs</div>
    `;
    list.appendChild(card);
  });
}

function renderFlights() {
  const list = refs.flightsList;
  list.innerHTML = "";

  if (!state.observation?.available_flights?.length) {
    const empty = document.createElement("div");
    empty.className = "empty empty-flights";
    empty.textContent = "No flights in observation.";
    list.appendChild(empty);
    return;
  }

  state.observation.available_flights.forEach((flight) => {
    const card = document.createElement("article");
    card.className = "card card-flight";
    card.innerHTML = `
      <div class="card-top">
        <div class="card-title">${flight.id} ${flight.is_partner ? "(partner)" : "(same airline)"}</div>
        <span class="chip">T+${flight.departure_hrs}h</span>
      </div>
      <div class="card-sub">${flight.destination}</div>
      <div class="card-sub">Economy ${flight.economy_seats} | Business ${flight.business_seats}</div>
    `;
    list.appendChild(card);
  });
}

function renderActionOptions() {
  const obs = state.observation;
  if (!obs) return;

  const passengerOptions = [{ value: "", label: "Select passenger" }].concat(
    obs.pending_passengers.map((p) => ({ value: p.id, label: `${p.id} - ${p.name}` }))
  );
  setSelectOptions(refs.passengerId, passengerOptions, refs.passengerId.value);

  const flightOptions = [{ value: "", label: "Select flight" }].concat(
    obs.available_flights.map((f) => ({
      value: f.id,
      label: `${f.id} | E${f.economy_seats} B${f.business_seats} | ${f.is_partner ? "partner" : "same"}`,
    }))
  );
  setSelectOptions(refs.flightId, flightOptions, refs.flightId.value);

  const actionType = refs.actionType.value;
  refs.passengerId.disabled = actionType === "finalize";
  refs.flightId.disabled = !requiresFlight(actionType);
}

function renderStatus() {
  refs.sessionBadge.textContent = `Session: ${state.sessionId}`;
  refs.phaseBadge.textContent = `State: ${state.done ? "done" : "active"}`;
  refs.scoreBadge.textContent = `Grade: ${state.grade == null ? "-" : Number(state.grade).toFixed(4)}`;
}

function renderLatest() {
  if (!refs.latestResult) {
    return;
  }

  refs.latestResult.innerHTML = buildLatestSummary();

  if (refs.latestRaw) {
    refs.latestRaw.textContent = state.latest ? safeJson(state.latest) : "";
    refs.latestRaw.classList.toggle("hidden", !state.showRawLatest);
  }

  if (refs.toggleRawBtn) {
    refs.toggleRawBtn.textContent = state.showRawLatest ? "Hide Raw JSON" : "Show Raw JSON";
  }
}

function renderLog() {
  refs.logList.innerHTML = "";
  if (!state.logs.length) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "No step logs yet.";
    refs.logList.appendChild(empty);
    return;
  }

  [...state.logs].reverse().forEach((entry) => {
    const item = document.createElement("article");
    item.className = "log-item";
    item.innerHTML = `
      <div><strong>${entry.action.action_type}</strong> | reward ${Number(entry.reward).toFixed(4)} | done ${entry.done ? "1" : "0"}</div>
      <div class="log-meta">passenger ${entry.action.passenger_id || "na"} | flight ${entry.action.flight_id || "na"}</div>
      <div class="log-meta">${entry.notes || ""}</div>
    `;
    refs.logList.appendChild(item);
  });
}

function renderAll() {
  renderTaskMeta();
  renderMetrics();
  renderPassengers();
  renderFlights();
  renderActionOptions();
  renderStatus();
  renderLatest();
  renderLog();
}

async function refreshGrade() {
  try {
    const result = await api(`/state?session_id=${encodeURIComponent(state.sessionId)}`);
    state.grade = result.grade;
  } catch {
    state.grade = null;
  }
}

async function loadTasks() {
  const result = await api("/tasks");
  state.tasks = result.tasks || [];

  setSelectOptions(
    refs.taskSelect,
    state.tasks.map((task) => ({ value: task.task_key, label: `${task.task_key.toUpperCase()} | ${task.passenger_count} pax` })),
    state.selectedTask
  );
}

async function resetSession() {
  const selectedTask = refs.taskSelect.value || "easy";
  const result = await api("/reset", {
    method: "POST",
    body: JSON.stringify({ task: selectedTask }),
  });

  state.selectedTask = result.task_key;
  state.sessionId = result.session_id;
  state.observation = result.observation;
  state.done = false;
  state.latest = { event: "reset", task: result.task_key, session_id: result.session_id };
  state.logs = [];

  await refreshGrade();
  renderAll();
}

function readActionFromForm() {
  const actionType = refs.actionType.value;
  const action = { action_type: actionType };

  if (actionType !== "finalize") {
    if (!refs.passengerId.value) {
      throw new Error("Passenger is required for this action.");
    }
    action.passenger_id = refs.passengerId.value;
  }

  if (requiresFlight(actionType)) {
    if (!refs.flightId.value) {
      throw new Error("Flight is required for this action.");
    }
    action.flight_id = refs.flightId.value;
  }

  return action;
}

async function runStep(action) {
  const result = await api("/step", {
    method: "POST",
    body: JSON.stringify({ session_id: state.sessionId, action }),
  });

  state.observation = result.observation;
  state.done = !!result.done;
  state.latest = result;

  const rewardValue = Number(result.reward?.value ?? 0);
  const notes = Array.isArray(result.reward?.notes) ? result.reward.notes.join(" | ") : "";
  state.logs.push({
    action,
    reward: rewardValue,
    done: state.done,
    notes,
  });

  if (typeof result.final_score === "number") {
    state.grade = result.final_score;
  } else {
    await refreshGrade();
  }

  renderAll();
}

async function autoStep() {
  if (!state.observation || state.done) return;
  const action = chooseSuggestion(state.observation);
  refs.actionType.value = action.action_type;
  refs.passengerId.value = action.passenger_id || "";
  refs.flightId.value = action.flight_id || "";
  renderActionOptions();
  await runStep(action);
}

function useSuggestion() {
  if (!state.observation || state.done) return;
  const action = chooseSuggestion(state.observation);
  refs.actionType.value = action.action_type;
  refs.passengerId.value = action.passenger_id || "";
  refs.flightId.value = action.flight_id || "";
  renderActionOptions();
}

function bindEvents() {
  refs.taskSelect.addEventListener("change", () => {
    state.selectedTask = refs.taskSelect.value;
    renderTaskMeta();
  });

  refs.actionType.addEventListener("change", renderActionOptions);
  refs.resetBtn.addEventListener("click", () => resetSession().catch(showError));
  refs.autoBtn.addEventListener("click", () => autoStep().catch(showError));
  refs.finalizeBtn.addEventListener("click", () => runStep({ action_type: "finalize" }).catch(showError));
  refs.suggestBtn.addEventListener("click", useSuggestion);
  refs.clearLogBtn.addEventListener("click", () => {
    state.logs = [];
    renderLog();
  });

  if (refs.toggleRawBtn) {
    refs.toggleRawBtn.addEventListener("click", () => {
      state.showRawLatest = !state.showRawLatest;
      renderLatest();
    });
  }

  refs.actionForm.addEventListener("submit", (event) => {
    event.preventDefault();
    try {
      const action = readActionFromForm();
      runStep(action).catch(showError);
    } catch (err) {
      showError(err);
    }
  });
}

function showError(err) {
  const message = err instanceof Error ? err.message : String(err);
  state.latest = { error: message };
  renderLatest();
}

async function init() {
  bindEvents();
  await loadTasks();
  await resetSession();
}

init().catch(showError);
