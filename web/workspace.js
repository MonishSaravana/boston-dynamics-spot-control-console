/* One shell around the existing sensor, target and control adapters. */
let workspaceState = null;
let currentEntity = null;
let artifacts = [];
let selectedArtifact = null;
let mapSignature = "";
let moduleSignature = "";
let candidateSignature = "";
let policySignature = "";
let confirmedState = null;
const NS = "http://www.w3.org/2000/svg";
const modeCopy = {
  observe: "Sensors and perception only. No movement authorization.",
  dry_run:
    "Full target pipeline. GO records the destination without moving Spot.",
  robot_control:
    "Command authority required. Confirm a target, inspect the destination, then GO.",
};

async function workspaceAction(name, extra = {}) {
  try {
    await api("/api/workspace/action", {
      name,
      epoch: workspaceState?.epoch,
      revision: workspaceState?.revision,
      controller,
      ...extra,
    });
    await refresh();
  } catch (error) {
    toast(error.message);
    await refresh();
  }
}

function details(element, pairs) {
  element.replaceChildren();
  for (const [label, value] of pairs) {
    const dt = document.createElement("dt");
    dt.textContent = label;
    const dd = document.createElement("dd");
    dd.textContent = value;
    element.append(dt, dd);
  }
}
const fixed = (value, digits = 2) =>
  value == null ? "Unavailable" : Number(value).toFixed(digits);
const xyz = (values) =>
  values ? values.map((v) => fixed(v)).join(", ") + " m" : "No world geometry";
function el(tag, attributes = {}, text) {
  const item = document.createElementNS(NS, tag);
  for (const [name, value] of Object.entries(attributes))
    item.setAttribute(name, value);
  if (text != null) item.textContent = text;
  return item;
}

function drawMap(svg, ws, large = false) {
  svg.replaceChildren();
  const mapping = ws?.mapping;
  if (!mapping) {
    svg.append(
      el(
        "text",
        { x: 30, y: 80, class: "map-empty" },
        "Map requires a verified RGB-D pair",
      ),
    );
    return;
  }
  const width = large ? 700 : 360,
    height = large ? 620 : 360;
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  const config = mapping.config,
    [nx, ny] = config.shape;
  const pad = large ? 64 : 16;
  const scale = Math.min((width - pad * 2) / nx, (height - pad * 2) / ny);
  const ox = (width - nx * scale) / 2,
    oy = (height - ny * scale) / 2;
  const point = (x, y) => [
    ox + ((x - config.origin[0]) / config.voxel_m) * scale,
    height - oy - ((y - config.origin[1]) / config.voxel_m) * scale,
  ];
  const enabled = (key) =>
    !large || document.querySelector(`[data-layer="${key}"]`).checked;
  const content = el("g");
  svg.append(content);
  if (large && $("map-view").value === "iso")
    content.setAttribute(
      "transform",
      "translate(70 -45) matrix(.82 .15 -.22 .72 90 55)",
    );
  if (enabled("occupancy")) {
    const colors = ["#142023", "#30423d", "#766853"];
    // Combine each row's equal cells into spans; bounded SVG work, no canvas loop.
    for (let i = 0; i < nx; i++) {
      let j = 0;
      while (j < ny) {
        const type = mapping.topdown[i][j];
        let end = j + 1;
        while (end < ny && mapping.topdown[i][end] === type) end++;
        content.append(
          el("rect", {
            x: ox + i * scale,
            y: height - oy - end * scale,
            width: scale + 0.1,
            height: (end - j) * scale + 0.1,
            fill: colors[type],
          }),
        );
        j = end;
      }
    }
    for (let i = 0; i <= nx; i += 10)
      content.append(
        el("line", {
          x1: ox + i * scale,
          y1: oy,
          x2: ox + i * scale,
          y2: height - oy,
          class: "grid-line",
        }),
      );
    for (let j = 0; j <= ny; j += 10)
      content.append(
        el("line", {
          x1: ox,
          y1: oy + j * scale,
          x2: width - ox,
          y2: oy + j * scale,
          class: "grid-line",
        }),
      );
  }
  if (enabled("entities"))
    for (const entity of ws.entities || []) {
      const [x1, y1] = point(entity.bounds_low_m[0], entity.bounds_low_m[1]);
      const [x2, y2] = point(entity.bounds_high_m[0], entity.bounds_high_m[1]);
      const selected =
        ws.selected?.entity_id === entity.entity_id ||
        (large && currentEntity === entity.entity_id);
      const group = el("g", {
        class: "map-entity",
        ...(large ? { tabindex: "0" } : {}),
        role: "button",
        "aria-label": `Select ${entity.entity_id}`,
      });
      group.append(
        el("rect", {
          x: x1,
          y: y2,
          width: Math.max(3, x2 - x1),
          height: Math.max(3, y1 - y2),
          fill: selected ? "#edc76e33" : "#11191966",
          stroke: selected ? "#edc76e" : "#afc2bc",
          "stroke-width": selected ? 2 : 1,
        }),
      );
      // Close neighbors use opposing label anchors to preserve readable names.
      const anchor = entity.entity_id.toLowerCase().includes("backpack")
        ? "end"
        : "start";
      const labelY = entity.entity_id.toLowerCase().includes("backpack")
        ? y2 - 14
        : y1 + 16;
      group.append(
        el(
          "text",
          {
            x: anchor === "end" ? x2 - 2 : x1 + 2,
            y: labelY,
            "text-anchor": anchor,
          },
          entity.entity_id,
        ),
      );
      const select = () => {
        currentEntity = entity.entity_id;
        renderEntity(ws);
        mapSignature = "";
        if (!large) workspaceAction("select", { entity_id: entity.entity_id });
        else {
          drawMap(svg, ws, true);
        }
      };
      group.addEventListener("click", select);
      group.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          select();
        }
      });
      content.append(group);
    }
  if (enabled("route") && ws.destination) {
    const p = ws.destination;
    content.append(
      el("path", {
        d: p.route_xy_m
          .map((xy, i) => (i ? "L" : "M") + point(...xy).join(" "))
          .join(" "),
        class: "route-path",
      }),
    );
    const [x, y] = point(p.x_m, p.y_m);
    content.append(el("circle", { cx: x, cy: y, r: 6, fill: "#9aebd2" }));
    const hx = x + 24 * Math.cos(p.yaw_rad),
      hy = y - 24 * Math.sin(p.yaw_rad);
    content.append(
      el("line", { x1: x, y1: y, x2: hx, y2: hy, class: "map-heading" }),
    );
    content.append(
      el("path", {
        d: `M${hx - 6 * Math.cos(p.yaw_rad - 0.5)} ${hy + 6 * Math.sin(p.yaw_rad - 0.5)} L${hx} ${hy} L${hx - 6 * Math.cos(p.yaw_rad + 0.5)} ${hy + 6 * Math.sin(p.yaw_rad + 0.5)}`,
        class: "map-heading",
        fill: "none",
      }),
    );
    if (ws.selected?.center_m) {
      const target = point(...ws.selected.center_m);
      content.append(
        el("line", {
          x1: x,
          y1: y,
          x2: target[0],
          y2: target[1],
          class: "target-ray",
        }),
      );
    }
  }
  if (enabled("robot") && ws.robot) {
    const r = ws.robot,
      [x, y] = point(r.x_m, r.y_m);
    content.append(el("circle", { cx: x, cy: y, r: 6, fill: "#eef3f1" }));
    content.append(
      el("line", {
        x1: x,
        y1: y,
        x2: x + 22 * Math.cos(r.yaw_rad),
        y2: y - 22 * Math.sin(r.yaw_rad),
        class: "robot-heading",
      }),
    );
  }
  if (large)
    svg.append(
      el(
        "text",
        {
          x: 12,
          y: large ? height - 22 : height - 10,
          fill: "#acb7b8",
          "font-size": 12,
        },
        `1 grid interval = ${(config.voxel_m * 10).toFixed(1)} m · ${ws.world_frame}`,
      ),
    );
}

function renderEntity(ws) {
  const entity = (ws?.entities || []).find(
    (item) => item.entity_id === currentEntity,
  );
  $("entity-title").textContent = entity?.entity_id || "Select an entity";
  $("entity-target").disabled = !entity;
  if (!entity) {
    details($("entity-details"), []);
    return;
  }
  const center = entity.bounds_low_m.map(
    (v, i) => (v + entity.bounds_high_m[i]) / 2,
  );
  const dimensions = entity.bounds_high_m.map(
    (v, i) => v - entity.bounds_low_m[i],
  );
  details($("entity-details"), [
    ["Class proposal", entity.label],
    ["Position", xyz(center)],
    ["Dimensions", xyz(dimensions)],
    ["World frame", ws.world_frame],
    ["Evidence score", fixed(entity.confidence)],
    [
      "Last seen",
      entity.last_seen_s == null
        ? "Synthetic fixture"
        : `${fixed(entity.last_seen_s)} source s`,
    ],
  ]);
  $("entity-history").textContent =
    JSON.stringify(entity, null, 2) +
    "\n\nPersisted entity history and change events are available in recorded memory sessions under Runs.";
  for (const button of $("entity-list").children)
    button.classList.toggle(
      "selected",
      button.dataset.entity === currentEntity,
    );
}

function updateWorkspace(ws, control) {
  if (!ws) return;
  workspaceState = ws;
  $("control-mode").value = ws.mode;
  $("mode-description").textContent = control.demo
    ? ws.mode === "observe"
      ? "Synthetic sensors only. GO is disabled."
      : "Synthetic room. GO moves a virtual robot; manual Spot controls stay disabled."
    : modeCopy[ws.mode];
  const readable = {
    SIMULATED_ARRIVAL:
      "Virtual robot arrived. Preview again before another GO.",
    WOULD_EXECUTE_NO_MOTION:
      "Dry run recorded the SE2 destination. No movement command sent.",
    NOT_FOUND: "No target found. Try another object phrase.",
  };
  $("interaction-message").textContent = readable[ws.message] || ws.message;
  if (ws.available && !control.demo) {
    $("mode-pill").textContent =
      ws.mode === "robot_control"
        ? "ROBOT CONTROL"
        : ws.mode === "dry_run"
          ? "DRY RUN"
          : "OBSERVE";
    $("connection-message").textContent = control.connected
      ? control.status
      : "Spot sensors connected · no command lease";
    $("connect-button").textContent = "Disconnect";
  }
  if (ws.available) {
    $("image-badge").textContent = control.demo
      ? "SYNTHETIC ROOM · NOT SPOT FOOTAGE"
      : "SPOT CAMERA";
    const view = $("camera-select").value;
    if (view === "evidence") {
      $("image-title").textContent = ws.visual_source;
      const camera = ws.cameras?.[ws.visual_source];
      $("image-meta").textContent = control.demo
        ? "Simulated RGB-D pair"
        : `Age ${fixed(camera?.age_s, 1)} s · ${camera?.health || "Unavailable"}`;
      $("camera-caption").textContent =
        `Processing pair: ${ws.visual_source} + ${ws.depth_source || "no depth"}${ws.alignment_verified ? "" : " · alignment unverified"}`;
      $("camera-empty").textContent =
        "Perception display disabled or waiting for fresh frames";
      if (camera && (!camera.displayed || !camera.acquired)) {
        $("camera-image").hidden = true;
        $("camera-empty").hidden = false;
      }
    }
  }
  const c = ws.selected;
  $("target-stage").textContent = ws.confirmed
    ? ws.destination
      ? "Confirmed"
      : "Preview required"
    : c
      ? "Candidate"
      : ws.candidates?.length
        ? "Choose candidate"
        : "No target";
  $("target-column")?.setAttribute("data-confirmed", String(ws.confirmed));
  document.querySelector(".target-column").dataset.confirmed = String(
    ws.confirmed,
  );
  if (confirmedState !== ws.confirmed) {
    confirmedState = ws.confirmed;
    $("target-inputs").open = !ws.confirmed;
  }
  $("destination-empty").textContent = ws.confirmed
    ? "Movement authorization cleared. Preview again before GO."
    : "Confirm a target to propose a destination.";
  $("target-label").textContent = c ? c.entity_id : "No target selected";
  $("target-explanation").textContent = c
    ? control.demo
      ? "Synthetic fixture evidence"
      : `${c.label} · ${c.track_state} · ${c.source.toLowerCase()} selection`
    : "Type an object, use pointing, or select a map entity.";
  details(
    $("target-details"),
    c
      ? [
          ["World position", xyz(c.center_m)],
          ["World frame", c.world_frame || "Unavailable"],
          ["Evidence age", `${fixed(c.age_s, 1)} s`],
          ["Evidence score", `${fixed(c.score)} · uncalibrated`],
          ["Camera source", c.camera || ws.visual_source],
        ]
      : [],
  );
  $("target-raw").textContent = c
    ? JSON.stringify(c.evidence, null, 2)
    : "No candidate evidence";
  $("confirm-target").disabled = !ws.can_confirm;
  $("reject-target").disabled = !c;
  $("preview-button").disabled = !ws.confirmed;
  $("find-target").disabled = !ws.available;
  $("point-button").textContent = control.demo
    ? "Simulate pointing"
    : "Use pointing";
  $("point-button").disabled =
    !ws.available || (!control.demo && !ws.human_pose);
  $("point-note").textContent = control.demo
    ? "Fixture example; not a detected person"
    : ws.human_pose
      ? "Uses visible human geometry; uncertain rays may abstain"
      : "Enable human pose when connecting";
  const signature = JSON.stringify([
    ws.candidates?.map((item) => [item.entity_id, item.label]),
    c?.entity_id,
  ]);
  if (signature !== candidateSignature) {
    candidateSignature = signature;
    $("candidate-list").replaceChildren();
    for (const candidate of ws.candidates || []) {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = candidate.entity_id;
      button.className = c?.entity_id === candidate.entity_id ? "selected" : "";
      button.addEventListener("click", () =>
        workspaceAction("select", { entity_id: candidate.entity_id }),
      );
      $("candidate-list").append(button);
    }
  }
  const p = ws.destination;
  $("route-scale").textContent = ws.mapping
    ? `Grid interval ${(ws.mapping.config.voxel_m * 10).toFixed(1)} m · ${ws.world_frame}`
    : "Scale unavailable until a verified map is present";
  $("destination-empty").hidden = !!p;
  $("route-readout").textContent = p
    ? `Destination ${fixed(p.x_m)}, ${fixed(p.y_m)} m. Heading ${fixed((p.yaw_rad * 180) / Math.PI, 0)}° faces the target. Standoff ${fixed(p.stand_off_m)} m. ${p.route_xy_m.length} observed-free samples.`
    : ws.confirmed
      ? "Movement authorization cleared. Preview again to review a route."
      : "World-frame poses. Camera view does not limit travel direction.";
  details(
    $("destination-details"),
    p
      ? [
          ["Destination", `${fixed(p.x_m)}, ${fixed(p.y_m)} m`],
          [
            "Heading",
            `${fixed((p.yaw_rad * 180) / Math.PI, 0)}° · faces target`,
          ],
          [
            "Standoff / clearance",
            `${fixed(p.stand_off_m)} / ${fixed(p.clearance_m)} m`,
          ],
          ["Route evidence", `${p.route_xy_m.length} observed-free samples`],
        ]
      : [],
  );
  $("go-button").textContent = control.demo
    ? "GO · virtual"
    : ws.mode === "dry_run"
      ? "GO · dry run"
      : "GO · robot";
  $("go-button").disabled = !ws.can_go;
  $("go-note").textContent =
    ws.mode === "observe"
      ? "Observe has no movement authorization."
      : control.demo
        ? "Virtual destination only."
        : ws.mode === "dry_run"
          ? "Exact SE2 pose is recorded. No motion."
          : "Single-use GO. Keep the class E-stop ready.";
  if (ws.mode !== "robot_control" || ws.navigation_active) {
    for (const id of [
      "power-button",
      "stand-button",
      "apply-button",
      "gesture-toggle",
    ])
      $(id).disabled = true;
    for (const button of document.querySelectorAll(".direction"))
      button.disabled = true;
    $("drive-state").textContent = control.demo
      ? "DEMO · DISABLED"
      : ws.navigation_active
        ? "GO ACTIVE"
        : "LOCKED";
  }
  $("model-status").textContent = control.model_status;
  const mapKey = JSON.stringify([
    ws.mapping?.revision,
    ws.entities,
    ws.robot?.x_m,
    ws.robot?.y_m,
    ws.robot?.yaw_rad,
    c?.entity_id,
    p?.proposed_host_s,
    currentEntity,
    $("map-view").value,
    [...document.querySelectorAll("[data-layer]")].map((box) => box.checked),
  ]);
  if (mapKey !== mapSignature) {
    mapSignature = mapKey;
    drawMap($("route-map"), ws);
    drawMap($("main-map"), ws, true);
  }
  const entityKey = JSON.stringify(ws.entities?.map((item) => item.entity_id));
  if ($("entity-list").dataset.key !== entityKey) {
    $("entity-list").dataset.key = entityKey;
    $("entity-list").replaceChildren();
    for (const entity of ws.entities || []) {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = `${entity.entity_id} · ${entity.label}`;
      button.dataset.entity = entity.entity_id;
      button.addEventListener("click", () => {
        currentEntity = entity.entity_id;
        renderEntity(workspaceState);
        mapSignature = "";
        drawMap($("main-map"), workspaceState, true);
      });
      $("entity-list").append(button);
    }
  }
  renderEntity(ws);
  $("map-source").textContent = ws.available
    ? `${control.demo ? "SYNTHETIC ROOM" : "LIVE ODOM"} · ${ws.mapping?.revision || 0} map updates`
    : "No map available";
  $("map-frame").textContent = ws.available
    ? `${ws.visual_source} + ${ws.depth_source || "no depth"} · ${ws.world_frame}`
    : "Waiting for a selected RGB-D pair";
  const moduleRows = Object.entries(ws.modules || {});
  $("module-summary").textContent = control.demo
    ? "Synthetic fixtures · live modules disabled"
    : moduleRows.length
      ? `${moduleRows.length} module states`
      : "No live perception pipeline";
  const health = control.demo
    ? [
        ["Cameras", "SIMULATED"],
        ["Mapping", "SIMULATED"],
        ["Entities", "SIMULATED"],
        ["Human pose", "DISABLED"],
        ["Network", "OFFLINE"],
      ]
    : moduleRows.slice(0, 8).map(([key, value]) => [key, value.status]);
  $("map-health").replaceChildren();
  for (const [name, status] of health) {
    const line = document.createElement("div");
    line.className = "health-line";
    const label = document.createElement("span");
    label.textContent = name;
    const value = document.createElement("span");
    value.textContent = status;
    line.append(label, value);
    $("map-health").append(line);
  }
  if ($("page-operate").querySelector(".developer").open)
    $("telemetry").textContent = JSON.stringify(
      {
        control_mode: ws.mode,
        destination_preview: ws.destination,
        stages: ws.timings,
      },
      null,
      2,
    );
  const configKey = JSON.stringify(
    moduleRows.map(([key, value]) => [key, value.config]),
  );
  if (configKey !== moduleSignature) {
    moduleSignature = configKey;
    buildModules(moduleRows);
  }
  for (const row of $("module-controls").children) {
    const item = ws.modules[row.dataset.module];
    if (item) {
      row.querySelector(".module-status").textContent = item.status;
      row.querySelector(".module-latency").textContent =
        `p95 ${fixed(item.latency?.p95_ms, 1)} ms`;
    }
  }
  if ($("settings-dialog").open) buildPolicies(ws);
}

function buildModules(rows) {
  $("module-controls").replaceChildren();
  for (const [name, item] of rows) {
    const row = document.createElement("div");
    row.className = "module-row";
    row.dataset.module = name;
    const label = document.createElement("label");
    const enabled = document.createElement("input");
    enabled.type = "checkbox";
    enabled.checked = item.config.enabled;
    label.append(enabled, document.createTextNode(name));
    const status = document.createElement("span");
    status.className = "module-status";
    status.textContent = item.status;
    const latency = document.createElement("span");
    latency.className = "module-latency";
    const rate = document.createElement("input");
    rate.type = "number";
    rate.min = ".1";
    rate.max = "30";
    rate.step = ".1";
    rate.value = item.config.max_hz || 5;
    rate.setAttribute("aria-label", `${name} max Hz`);
    const apply = document.createElement("button");
    apply.type = "button";
    apply.textContent = "Apply Hz";
    enabled.addEventListener("change", () =>
      workspaceAction("module", {
        module: name,
        enabled: enabled.checked,
        max_hz: Number(rate.value),
      }),
    );
    apply.addEventListener("click", () =>
      workspaceAction("module", {
        module: name,
        enabled: enabled.checked,
        max_hz: Number(rate.value),
      }),
    );
    row.append(label, status, latency, rate, apply);
    $("module-controls").append(row);
  }
}
function buildPolicies(ws) {
  const keys = Object.keys(ws.cameras || {}),
    signature = JSON.stringify(keys);
  if (signature === policySignature) return;
  policySignature = signature;
  for (const id of ["visual-source", "depth-source"]) {
    $(id).replaceChildren();
    if (id === "depth-source") {
      const none = document.createElement("option");
      none.value = "";
      none.textContent = "No depth (query only)";
      $(id).append(none);
    }
    for (const source of keys) {
      const option = document.createElement("option");
      option.value = source;
      option.textContent = source;
      $(id).append(option);
    }
  }
  // Synthetic depth is part of the fixed fixture, not a separately acquired Spot source.
  if (ws.simulated) {
    const option = document.createElement("option");
    option.value = "synthetic-depth";
    option.textContent = "synthetic-depth";
    $("depth-source").append(option);
  }
  $("visual-source").value = ws.visual_source || "";
  $("depth-source").value = ws.depth_source || "";
  $("alignment-verified").checked = ws.alignment_verified;
  for (const id of [
    "visual-source",
    "depth-source",
    "alignment-verified",
    "apply-pair",
  ])
    $(id).disabled = ws.simulated || !ws.available;
  $("camera-policies").replaceChildren();
  for (const source of keys) {
    const status = ws.cameras[source];
    const row = document.createElement("div");
    row.className = "camera-policy";
    const name = document.createElement("span");
    name.className = "camera-policy-name";
    name.textContent = source;
    const health = document.createElement("small");
    health.textContent = status.health;
    name.append(health);
    row.append(name);
    for (const [labelText, key] of [
      ["Acquire", "acquire"],
      ["Display", "display"],
    ]) {
      const label = document.createElement("label");
      const toggle = document.createElement("input");
      toggle.type = "checkbox";
      toggle.checked = status[key === "acquire" ? "acquired" : "displayed"];
      toggle.disabled = ws.simulated;
      label.append(toggle, document.createTextNode(labelText));
      toggle.addEventListener("change", () =>
        workspaceAction("camera", { source, [key]: toggle.checked }),
      );
      row.append(label);
    }
    const hz = document.createElement("input");
    hz.type = "number";
    hz.min = ".2";
    hz.max = "30";
    hz.step = ".2";
    hz.value = status.max_hz;
    hz.setAttribute("aria-label", `${source} acquisition Hz`);
    hz.disabled = ws.simulated;
    hz.addEventListener("change", () =>
      workspaceAction("camera", { source, max_hz: Number(hz.value) }),
    );
    row.append(hz);
    $("camera-policies").append(row);
  }
  $("settings-message").textContent = ws.simulated
    ? "Synthetic pair is fixed. Live source policies become available after connection."
    : "Changing the processing pair clears target approval and destination.";
}

function route() {
  const name = ["operate", "runs", "maps", "evaluate"].includes(
    location.hash.slice(1),
  )
    ? location.hash.slice(1)
    : "operate";
  clearMovement();
  for (const page of document.querySelectorAll(".page"))
    page.hidden = page.id !== `page-${name}`;
  for (const link of document.querySelectorAll(".topbar nav a")) {
    if (link.hash === `#${name}`) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
  if (name === "runs") loadRuns();
  if (name === "maps" && workspaceState)
    drawMap($("main-map"), workspaceState, true);
}
window.addEventListener("hashchange", route);
$("query-form").addEventListener("submit", (e) => {
  e.preventDefault();
  workspaceAction("query", { phrase: $("target-query").value });
});
$("point-button").addEventListener("click", () => workspaceAction("point"));
for (const [id, name] of [
  ["confirm-target", "confirm"],
  ["reject-target", "reject"],
  ["preview-button", "preview"],
  ["go-button", "go"],
])
  $(id).addEventListener("click", () => workspaceAction(name));
$("entity-target").addEventListener("click", async () => {
  await workspaceAction("select", { entity_id: currentEntity });
  location.hash = "operate";
});
$("control-mode").addEventListener("change", async () => {
  clearMovement();
  const mode = $("control-mode").value;
  if (mode === "robot_control" && !state?.connected) {
    $("control-mode").value = workspaceState.mode;
    toast(
      state?.demo
        ? "Leave the offline demo, then connect with command authority."
        : "Disconnect sensors, then connect in Robot control with command authority.",
    );
    return;
  }
  await workspaceAction("mode", { mode });
});
$("connect-mode").addEventListener("change", () => {
  $("authority-row").hidden = $("connect-mode").value !== "robot_control";
});
$("settings-button").addEventListener("click", () => {
  clearMovement();
  policySignature = "";
  buildPolicies(workspaceState || {});
  $("settings-dialog").showModal();
});
$("close-settings").addEventListener("click", () =>
  $("settings-dialog").close(),
);
$("pair-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  await workspaceAction("pair", {
    visual_source: $("visual-source").value,
    depth_source: $("depth-source").value,
    alignment_verified: $("alignment-verified").checked,
  });
  policySignature = "";
  buildPolicies(workspaceState);
});
$("rerun-help").addEventListener("click", () => $("rerun-dialog").showModal());
$("close-rerun").addEventListener("click", () => $("rerun-dialog").close());
$("rerun-to-runs").addEventListener("click", () => $("rerun-dialog").close());
$("map-runs-button").addEventListener("click", () => {
  location.hash = "runs";
});
for (const input of document.querySelectorAll("[data-layer]"))
  input.addEventListener("change", () => {
    mapSignature = "";
    drawMap($("main-map"), workspaceState, true);
  });
$("map-view").addEventListener("change", () => {
  mapSignature = "";
  drawMap($("main-map"), workspaceState, true);
});
setInterval(() => {
  if (
    workspaceState?.navigation_active &&
    !document.hidden &&
    document.hasFocus()
  )
    api("/api/workspace/presence", { controller }).catch(() => command("stop"));
}, 100);

async function loadRuns() {
  try {
    artifacts = (await api("/api/runs")).artifacts;
    renderRuns();
  } catch (error) {
    $("run-list").textContent = error.message;
  }
}
function renderRuns() {
  const phrase = $("run-filter").value.toLowerCase();
  const rows = artifacts.filter((item) =>
    `${item.name} ${item.run} ${item.kind}`.toLowerCase().includes(phrase),
  );
  $("run-list").replaceChildren();
  if (!rows.length) {
    const p = document.createElement("p");
    p.textContent = artifacts.length
      ? "No artifacts match this filter."
      : "No local artifacts found. Use an existing CLI to record a run, or launch with --runs-dir PATH.";
    $("run-list").append(p);
    return;
  }
  for (const item of rows) {
    const button = document.createElement("button");
    button.type = "button";
    button.className =
      "run-row" + (selectedArtifact?.id === item.id ? " selected" : "");
    const first = document.createElement("span");
    const strong = document.createElement("strong");
    strong.textContent = item.name;
    const small = document.createElement("small");
    small.textContent = `${item.run === "." ? "Run root" : item.run} · ${(item.bytes / 1024).toFixed(0)} KB`;
    first.append(strong, small);
    const kind = document.createElement("span");
    kind.textContent = item.kind;
    button.append(first, kind);
    button.addEventListener("click", () => inspectArtifact(item));
    $("run-list").append(button);
  }
}
async function inspectArtifact(item) {
  selectedArtifact = item;
  renderRuns();
  $("artifact-title").textContent = item.name;
  $("artifact-description").textContent =
    `${item.run} · ${item.kind} · read only`;
  $("open-recording").hidden = !item.name.endsWith(".rrd");
  $("artifact-content").textContent = "Reading local artifact…";
  try {
    $("artifact-content").textContent = JSON.stringify(
      await api("/api/runs/inspect", { id: item.id }),
      null,
      2,
    );
  } catch (error) {
    $("artifact-content").textContent = error.message;
  }
}
$("open-recording").addEventListener("click", async () => {
  try {
    toast((await api("/api/runs/open", { id: selectedArtifact.id })).message);
  } catch (error) {
    toast(error.message);
  }
});
$("refresh-runs").addEventListener("click", loadRuns);
$("run-filter").addEventListener("input", renderRuns);
const evaluationCommands = [
  [
    "Perception",
    "Local query and tracking runtime. Uses installed model bundles.",
    ".venv/bin/python benchmarks/perception_runtime.py --tum TUM_DIRECTORY --output runs/perception-check",
  ],
  [
    "Mapping",
    "Existing synthetic RGB-D reconstruction and saved-map checks.",
    ".venv/bin/python -m scope map synthetic --frames 4 --output runs/mapping-check --no-viewer",
  ],
  [
    "Pointing & humans",
    "Existing multi-view geometry, pointing and telemetry evaluation.",
    ".venv/bin/python -m scope benchmark-humans --output runs/pointing-check",
  ],
  [
    "Persistent memory",
    "Existing repeat-visit identity and change-event evaluation.",
    ".venv/bin/python -m scope benchmark-memory --output runs/memory-check",
  ],
  [
    "Semantic entities",
    "Projection, fusion, association and localization metrics.",
    ".venv/bin/python -m scope benchmark-entities --output runs/entity-check",
  ],
  [
    "Integration & safety",
    "Run the repository test suite without a physical robot.",
    ".venv/bin/python -m unittest discover -v",
  ],
  [
    "Spot sensor diagnostic",
    "Physical source names, timing and geometry. Read-only; requires robot access.",
    ".venv/bin/python -m scope spot-sensors --hostname ROBOT_IP --samples 10",
  ],
];
const legacyCommands = [
  [
    "Qt manual console",
    "Existing dockable cameras, model, posture and gesture controls.",
    ".venv/bin/python spot_control_gui.py --demo",
  ],
  [
    "Qt target interaction",
    "Same target/destination adapters as this browser.",
    ".venv/bin/python -m scope.m5_console --demo",
  ],
  [
    "Common perception",
    "Model, segmentation, tracking, performance and source options.",
    ".venv/bin/python -m scope query-live --help",
  ],
  [
    "Multi-camera humans",
    "3D skeletons, pointing, camera frustums, health and Rerun.",
    ".venv/bin/python -m scope map synthetic --humans --entities --oracle-keypoints --frames 4 --output runs/humans-preview",
  ],
  [
    "Map reload / viewer",
    "Replace RUN_DIRECTORY with an existing recorded run.",
    ".venv/bin/python -m scope view RUN_DIRECTORY",
  ],
  [
    "Memory episode & history",
    "Replace MEMORY_DB and ENTITY_ID with recorded values.",
    ".venv/bin/python -m scope memory list --db MEMORY_DB\n.venv/bin/python -m scope history ENTITY_ID --db MEMORY_DB",
  ],
  [
    "Memory replay",
    "Persisted landmarks, changes and episode timeline in Rerun.",
    ".venv/bin/python -m scope replay-memory --db MEMORY_DB",
  ],
  [
    "All offline commands",
    "Original mapping, objects, human, memory and query CLIs remain available.",
    ".venv/bin/python -m scope --help",
  ],
];
function commandList(id, commands) {
  for (const [name, note, code] of commands) {
    const row = document.createElement("div");
    row.className = "command-row";
    const copy = document.createElement("div");
    const title = document.createElement("h2");
    title.textContent = name;
    const p = document.createElement("p");
    p.textContent = note;
    copy.append(title, p);
    const pre = document.createElement("pre");
    pre.textContent = code;
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = "Copy";
    button.setAttribute("aria-label", `Copy ${name} command`);
    button.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(code);
        toast("Command copied");
      } catch (_) {
        toast("Select and copy the command text");
      }
    });
    row.append(copy, pre, button);
    $(id).append(row);
  }
}
commandList("evaluation-commands", evaluationCommands);
commandList("legacy-commands", legacyCommands);
route();
