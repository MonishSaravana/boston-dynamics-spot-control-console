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
let worldZoom = 1;
let reviewFixture = true;
let evidenceVersion = null;
let priorSelected = null;
let inspectorPhase = null;
const NS = "http://www.w3.org/2000/svg";
const modeCopy = {
  observe: "Sensors and perception only. No movement authorization.",
  dry_run:
    "Full target pipeline. GO records the destination without moving Spot.",
  robot_control:
    "Command authority required. Manual drive and posture only; GO for the real robot is not included in this version.",
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
  const width = svg.clientWidth || 900,
    height = svg.clientHeight || 650;
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  const mapping = ws?.mapping;
  if (!mapping) {
    svg.append(
      el(
        "text",
        {
          x: width / 2,
          y: height / 2,
          "text-anchor": "middle",
          class: "map-empty",
        },
        "No verified world geometry",
      ),
    );
    return;
  }
  const config = mapping.config,
    [nx, ny] = config.shape,
    pad = Math.min(
      large ? 74 : 58,
      Math.max(18, Math.min(width, height) * 0.12),
    );
  const zoom = large ? worldZoom : 1;
  // Fit short canvases between the fixed caption bands.
  const verticalPad = Math.max(40, pad);
  const scale =
    Math.min((width - pad * 2) / nx, (height - verticalPad * 2) / ny) * zoom;
  const ox = (width - nx * scale) / 2,
    oy = (height - ny * scale) / 2;
  const point = (x, y) => [
    ox + ((x - config.origin[0]) / config.voxel_m) * scale,
    height - oy - ((y - config.origin[1]) / config.voxel_m) * scale,
  ];
  const enabled = (key) =>
    !large || document.querySelector(`[data-layer="${key}"]`)?.checked;
  const content = el("g");
  const entityLabels = [];
  const length =
    Math.max(1, Math.round(Math.min(100, width * 0.15) / scale)) *
    config.voxel_m;
  const pixels = (length / config.voxel_m) * scale;
  // Keep labels clear of the docked captions and the metric scale.
  const labelBounds = [
    [0, 0, width, 52],
    [0, height - 42, width, height],
    [18, height - 98, Math.max(78, 30 + pixels), height - 60],
  ];
  const footprintBounds = [];
  const reserveLabel = (x, y, text, size = 12) => {
    const box = [x - 3, y - size - 2, x + text.length * size * 0.6 + 3, y + 4];
    labelBounds.push(box);
    return box;
  };
  svg.append(content);
  if (large && $("map-view").value === "iso")
    content.setAttribute(
      "transform",
      `translate(${width / 2} ${height / 2}) matrix(.9 .18 -.3 .75 0 0) translate(${-width / 2} ${-height / 2})`,
    );
  if (enabled("occupancy")) {
    // Exact current topdown cell classification, coalesced into paths, without inventing walls.
    const paths = ["", "", ""];
    for (let x = 0; x < nx; x++) {
      let y = 0;
      while (y < ny) {
        const type = mapping.topdown[x][y];
        let end = y + 1;
        while (end < ny && mapping.topdown[x][end] === type) end++;
        if (type === 1 || type === 2) {
          const px = ox + x * scale,
            py = height - oy - end * scale;
          paths[type] +=
            `M${px} ${py}h${scale + 0.05}v${(end - y) * scale + 0.05}h${-scale - 0.05}z`;
        }
        y = end;
      }
    }
    if (paths[1])
      content.append(el("path", { d: paths[1], class: "map-cells map-free" }));
    if (paths[2])
      content.append(
        el("path", { d: paths[2], class: "map-cells map-occupied" }),
      );
  }
  if (document.querySelector('[data-layer="grid"]')?.checked) {
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
        tabindex: "0",
        role: "button",
        "aria-label": `Select ${entity.entity_id}`,
      });
      group.append(el("title", {}, entity.entity_id));
      footprintBounds.push([x1 - 4, y2 - 4, x2 + 4, y1 + 4]);
      // 30px also retains 24px screen bounds in the isometric projection.
      const hitWidth = Math.max(30, x2 - x1),
        hitHeight = Math.max(30, y1 - y2);
      group.append(el("rect", {
        x: (x1 + x2 - hitWidth) / 2,
        y: (y1 + y2 - hitHeight) / 2,
        width: hitWidth,
        height: hitHeight,
        class: "entity-hit",
        fill: "transparent",
        stroke: "none",
        "pointer-events": "all",
      }));
      group.append(
        el("rect", {
          x: x1,
          y: y2,
          width: Math.max(3, x2 - x1),
          height: Math.max(3, y1 - y2),
          rx: 1,
          class: "entity-bound",
          fill: selected ? "#f0b44c14" : "#c8d0d508",
          stroke: selected ? "#f0b44c" : "#7d878d",
          "stroke-width": selected ? 1.5 : 1,
          "vector-effect": "non-scaling-stroke",
        }),
      );
      if (selected) {
        const l = 7;
        group.append(
          el("path", {
            d: `M${x1 - 3} ${y2 + l}v${-l - 3}h${l + 3}M${x2 - l} ${y2 - 3}h${l + 3}v${l + 3}M${x2 + 3} ${y1 - l}v${l + 3}h${-l - 3}M${x1 + l} ${y1 + 3}h${-l - 3}v${-l - 3}`,
            fill: "none",
            stroke: "#f0b44c",
            "stroke-width": 1.5,
            "vector-effect": "non-scaling-stroke",
          }),
        );
      }
      entityLabels.push({ group, entity, selected, x1, x2, y1, y2 });
      const select = () => {
        currentEntity = entity.entity_id;
        renderEntity(ws);
        mapSignature = "";
        if (!large) workspaceAction("select", { entity_id: entity.entity_id });
        else drawMap(svg, ws, true);
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
  const annotations = el("g");
  if (enabled("route") && ws.destination) {
    const p = ws.destination;
    annotations.append(
      el("path", {
        d: p.route_xy_m
          .map((xy, i) => (i ? "L" : "M") + point(...xy).join(" "))
          .join(" "),
        class: "route-path",
      }),
    );
    const [x, y] = point(p.x_m, p.y_m);
    annotations.append(
      el("circle", {
        cx: x,
        cy: y,
        r: 5,
        fill: "#06140f",
        stroke: "#3ecf9b",
        "stroke-width": 2,
      }),
    );
    const hx = x + 30 * Math.cos(p.yaw_rad),
      hy = y - 30 * Math.sin(p.yaw_rad);
    annotations.append(
      el("path", {
        d: `M${x} ${y}L${hx} ${hy}M${hx - 7 * Math.cos(p.yaw_rad - 0.5)} ${hy + 7 * Math.sin(p.yaw_rad - 0.5)}L${hx} ${hy}L${hx - 7 * Math.cos(p.yaw_rad + 0.5)} ${hy + 7 * Math.sin(p.yaw_rad + 0.5)}`,
        class: "map-heading",
        fill: "none",
      }),
    );
    labelBounds.push([x - 12, y - 12, x + 12, y + 12]);
    const destinationLabel = [
      [Math.min(width - 86, x + 56), Math.max(46, y - 32)],
      [x + 18, y + 34], [x - 96, y + 34],
      [x - 96, y - 24], [x + 18, y - 24],
    ].find(([lx, ly]) => {
      const box = [lx - 3, ly - 14, lx + 82, ly + 4];
      return box[0] >= 12 && box[2] <= width - 12 &&
        box[1] >= 12 && box[3] <= height - 28 &&
        ![...labelBounds, ...footprintBounds].some(b =>
          box[0] < b[2] + 5 && box[2] > b[0] - 5 &&
          box[1] < b[3] + 5 && box[3] > b[1] - 5);
    });
    if (destinationLabel) {
      const [lx, ly] = destinationLabel;
      annotations.append(el("path", {
        d: `M${x} ${y}L${Math.max(lx, Math.min(lx + 79, x))} ${ly - 5}`,
        class: "annotation-leader",
      }));
      annotations.append(el("text", {
        x: lx, y: ly, class: "map-label destination-label",
      }, "Destination"));
      reserveLabel(lx, ly, "Destination", 12);
    }
    if (ws.selected?.center_m) {
      const [tx, ty] = point(...ws.selected.center_m);
      content.append(
        el("line", { x1: x, y1: y, x2: tx, y2: ty, class: "target-ray" }),
      );
    }
  }
  if (enabled("robot") && ws.robot) {
    const r = ws.robot,
      [x, y] = point(r.x_m, r.y_m);
    const glyph = el("g", {
      transform: `translate(${x} ${y}) rotate(${(-r.yaw_rad * 180) / Math.PI})`,
    });
    glyph.append(
      el("circle", {
        r: 10,
        fill: "#15181a",
        stroke: "#5b6469",
        "stroke-width": 1,
      }),
    );
    glyph.append(el("path", { d: "M7 0L-5 -5L-3 0L-5 5Z", fill: "#e6e8e9" }));
    content.append(glyph);
    const close =
      ws.destination &&
      Math.hypot(
        ...point(ws.destination.x_m, ws.destination.y_m).map(
          (v, i) => v - [x, y][i],
        ),
      ) < 44;
    const preferredY = Math.min(height < 250 ? height - 52 : height - 28, close ? y + 29 : y - 14);
    labelBounds.push([x - 12, y - 12, x + 12, y + 12]);
    const labelPosition = [
      [close ? x - 48 : x + 17, preferredY],
      [x + 18, y + 30], [x - 48, y - 18], [x + 18, y - 18],
      [x - 74, y + 46], [x + 18, y + 48], [x - 74, y - 40],
    ].find(([lx, ly]) => {
      const box = [lx - 3, ly - 14, lx + 39, ly + 4];
      return box[0] >= 12 && box[2] <= width - 12 && box[1] >= 12 && box[3] <= height - 28 &&
        ![...labelBounds, ...footprintBounds].some(b => box[0] < b[2] + 5 && box[2] > b[0] - 5 && box[1] < b[3] + 5 && box[3] > b[1] - 5);
    });
    if (labelPosition) {
      const [lx, ly] = labelPosition;
      annotations.append(
        el("path", {
          d: `M${x} ${y}L${Math.max(lx, Math.min(lx + 36, x))} ${ly - 5}`,
          class: "annotation-leader",
        }),
      );
      annotations.append(el("text", { x: lx, y: ly, class: "map-label" }, "Robot"));
      reserveLabel(lx, ly, "Robot", 12);
    }
  }
  // Labels render above all footprints, with leaders to their own bounds.
  // Suppress secondary labels when the canvas cannot place them safely.
  const entityAnnotations = el("g", { "pointer-events": "none" });
  for (const item of entityLabels.sort((a, b) => Number(b.selected) - Number(a.selected))) {
    const { entity, selected, x1, x2, y1, y2 } = item;
    const size = selected ? 13 : 12;
    const textWidth = entity.entity_id.length * size * 0.6;
    const positions = [
      [x1, y2 - 16], [x2 + 16, y2 + 12],
      [x1, y1 + 26], [x1 - textWidth - 16, y2 + 12],
      [(x1 + x2 - textWidth) / 2, y2 - 22],
      [(x1 + x2 - textWidth) / 2, y1 + 28],
    ];
    let position = null;
    if (selected || (width >= 480 && height >= 300)) {
      for (const [x, y] of positions) {
        const box = [x - 3, y - size - 2, x + textWidth + 3, y + 4];
        if (box[0] < 12 || box[2] > width - 12 || box[1] < 12 || box[3] > height - 28) continue;
        if ([...labelBounds, ...footprintBounds].some((b) => box[0] < b[2] + 5 && box[2] > b[0] - 5 && box[1] < b[3] + 5 && box[3] > b[1] - 5)) continue;
        position = [x, y];
        reserveLabel(x, y, entity.entity_id, size);
        break;
      }
    }
    if (!position) continue;
    const [x, y] = position;
    const tx = Math.max(x, Math.min(x + textWidth, (x1 + x2) / 2));
    const ty = y - 5;
    const ax = Math.max(x1, Math.min(x2, tx));
    const ay = Math.max(y2, Math.min(y1, ty));
    entityAnnotations.append(el("path", {
      d: `M${ax} ${ay}L${tx} ${ty}`,
      class: "annotation-leader entity-leader",
    }));
    entityAnnotations.append(el("text", {
      x, y,
      class: "entity-label" + (selected ? " selected-label" : " context-label"),
    }, entity.entity_id));
  }
  content.append(annotations);
  content.append(entityAnnotations);
  svg.append(
    el("path", {
      d: `M24 ${height - 70}v5h${pixels}v-5`,
      stroke: "#8b9499",
      "stroke-width": 1.5,
      fill: "none",
    }),
  );
  svg.append(
    el(
      "text",
      { x: 24, y: height - 80, class: "map-origin" },
      `${length.toFixed(1)} m`,
    ),
  );
  $("map-scale").textContent = ws.world_frame;
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
    "\n\nPersisted entity history and change events are available in recorded memory sessions under History.";
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
    "AMBIGUOUS: select a candidate": `${ws.candidates?.length || "Several"} matches. Select one to inspect its evidence.`,
  };
  $("interaction-message").textContent = readable[ws.message] || ws.message;
  if (ws.available && !control.demo) {
    $("mode-pill").textContent =
      ws.mode === "robot_control"
        ? "Spot connected"
        : ws.mode === "dry_run"
          ? "Spot connected"
          : "Spot connected";
    $("connection-message").textContent = control.connected
      ? control.status
      : "Spot sensors connected · no command lease";
    $("connect-button").textContent = "Disconnect";
  }
  if (ws.available) {
    $("image-badge").textContent = control.demo ? "Demo data" : "Spot camera";
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
  $("target-label").textContent = c ? c.entity_id : "Find an object";
  $("target-explanation").textContent = c
    ? control.demo
      ? `${c.label} · world geometry available`
      : `${c.label} · ${c.track_state} · ${c.source.toLowerCase()} selection`
    : "Search above, use pointing, or select in World.";
  details(
    $("target-details"),
    c
      ? [
          ["World position", xyz(c.center_m)],
          ["Evidence age", `${fixed(c.age_s, 1)} s`],
          ["Evidence score", `${fixed(c.score)} · heuristic`],
          ["Camera source", c.camera || ws.visual_source],
        ]
      : [],
  );
  $("destination-target").textContent = ws.selected
    ? `Target / ${ws.selected.entity_id}`
    : "";
  $("target-raw").textContent = c
    ? JSON.stringify(c.evidence, null, 2)
    : "No candidate evidence";
  $("confirm-target").disabled = !ws.can_confirm;
  $("reject-target").disabled = !c;
  $("preview-button").disabled = !ws.confirmed;
  $("find-target").disabled = !ws.available;
  $("point-label").textContent = control.demo
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
    ? `${ws.world_frame} · ${(ws.mapping.config.voxel_m * 100).toFixed(0)} cm cells`
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
    ? "GO"
    : ws.mode === "dry_run"
      ? "GO · dry run"
      : "GO";
  $("go-button").disabled = !ws.can_go;
  $("go-note").textContent =
    ws.mode === "observe"
      ? "Observe has no movement authorization."
      : ws.mode === "robot_control"
        ? "GO for the real robot is not included in this version. Use Dry run."
        : control.demo
        ? "Virtual destination only."
        : "Exact SE2 pose is recorded. No motion.";
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
    ? `${ws.world_frame} · ${ws.mapping?.revision || 0} updates`
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
  if ($("diagnostic-drawer").open)
    $("telemetry").textContent = JSON.stringify(
      {
        control_mode: ws.mode,
        destination_preview: ws.destination,
        stages: ws.timings,
        modules: ws.modules,
        sources: ws.cameras,
        backend: ws.backend_mode,
        processing_pair: {
          visual: ws.visual_source,
          depth: ws.depth_source,
          alignment_verified: ws.alignment_verified,
        },
        robot: ws.robot,
        frame_versions: control.frame_versions,
        worker_status: control.status,
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
  updateWorkstation(ws, control);
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
  $("alignment-label").textContent = ws.simulated
    ? "Fixture alignment (synthetic)"
    : "Alignment physically verified for this pair";
  $("alignment-note").textContent = ws.simulated
    ? "Fixed RGB-D geometry supplied by the synthetic fixture."
    : "Measured calibration, timestamps and odom transforms are still checked. No world geometry or destination is authorized from an unverified pair.";
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
      ["Process", "process"],
      ["Display", "display"],
    ]) {
      const label = document.createElement("label");
      const toggle = document.createElement("input");
      toggle.type = "checkbox";
      toggle.checked =
        status[
          { acquire: "acquired", process: "processed", display: "displayed" }[
            key
          ]
        ];
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
    const rate = document.createElement("label");
    rate.className = "source-rate";
    rate.append(hz, document.createTextNode("Hz"));
    row.append(rate);
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
  for (const link of document.querySelectorAll(".primary-nav > a")) {
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
$("rerun-help").addEventListener("click", openAdvanced3D);
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
  for (const button of document.querySelectorAll("[data-map-view]"))
    button.setAttribute(
      "aria-pressed",
      String(button.dataset.mapView === $("map-view").value),
    );
  drawMap($("main-map"), workspaceState, true);
});
for (const button of document.querySelectorAll("[data-map-view]"))
  button.addEventListener("click", () => {
    $("map-view").value = button.dataset.mapView;
    $("map-view").dispatchEvent(new Event("change"));
    $("map-panel-meta").textContent = button.textContent.trim();
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

/* Canvas shell: presentation only. Actions still use the existing guarded adapters. */
function updateCanvasSource(ws, control) {
  if (!ws) return;
  const showReview = control.demo && reviewFixture && $("camera-select").value === "evidence";
  const camera = showReview ? "Lab preview" : names[$("camera-select").value] || ws.visual_source;
  $("canvas-source").textContent = `${camera} · ${ws.world_frame}`;
}
// Bring a Live panel into view; the dock keeps the rest of the layout.
function revealLivePanel(id) {
  liveDock.show(id);
  liveDock.flash(id);
  mapSignature = "";
  if (workspaceState && id === "world")
    requestAnimationFrame(() => drawMap($("route-map"), workspaceState));
}
function updateWorkstation(ws, control) {
  const phase = ws.confirmed
    ? "confirmed"
    : ws.selected
      ? "candidate"
      : "empty";
  document.querySelector(".target-column").dataset.phase = phase;
  if (phase !== inspectorPhase) {
    if (inspectorPhase !== null) selectInspectorTab("target");
    $("target-inputs").open = phase === "empty";
    inspectorPhase = phase;
    document.querySelector(".target-body").scrollTop = 0;
  }
  $("target-inputs").querySelector("summary").textContent = phase === "empty"
    ? "Choose target" : "Change target";
  $("inspector-heading").textContent =
    phase === "confirmed"
      ? "Destination"
      : phase === "candidate"
        ? "Target"
        : "Interaction";
  $("target-kind").textContent = ws.selected?.label || "Target";
  $("evidence-thumbnail").hidden = !ws.selected;
  if (ws.selected && evidenceVersion !== ws.version) {
    evidenceVersion = ws.version;
    loadImage("evidence", $("target-camera"));
  }
  if (ws.selected?.entity_id && priorSelected !== ws.selected.entity_id) {
    if (control.demo && !liveDock.visible("world")) revealLivePanel("world");
    liveDock.show("inspector");
  }
  priorSelected = ws.selected?.entity_id || null;
  $("fixture-toggle").hidden = !control.demo;
  const showReview =
    control.demo && reviewFixture && $("camera-select").value === "evidence";
  $("review-image").hidden = !showReview;
  if (showReview && !$("review-image").getAttribute("src"))
    $("review-image").src = "/fixtures/lab-review.png";
  $("fixture-toggle").textContent = reviewFixture
    ? "Sensor fixture"
    : "Lab preview";
  const evidenceOption = $("camera-select").querySelector(
    'option[value="evidence"]',
  );
  if (evidenceOption)
    evidenceOption.textContent = showReview ? "Lab preview" : "Evidence";
  if (showReview) {
    $("image-title").textContent = "Lab preview";
    $("image-meta").textContent = "Visual fixture";
  }
  updateCanvasSource(ws, control);
  $("canvas-selection").textContent = ws.selected
    ? `Target / ${ws.selected.entity_id}`
    : "No selection";
  $("footer-status").textContent = ws.navigation_active
    ? "Movement active"
    : ws.confirmed
      ? "Destination preview"
      : ws.available
        ? "Sensors available"
        : "No sensor connection";
  if (control.demo) $("mode-pill").textContent = "Offline demo";
  else if (ws.available) $("mode-pill").textContent = "Spot connected";
  const defaultMessage = [
    "SCOPE source updated.",
    "Select a target",
    "Target confirmed",
    "Target rejected",
    "Synthetic room; GO moves only the virtual robot",
    "Candidate highlighted; confirm or reject",
    "Destination previewed; GO is separate",
    "",
  ];
  if (defaultMessage.includes(ws.message))
    $("interaction-message").textContent = "";
  if (phase === "empty" && ws.message === "NOT_FOUND" && !$("target-query").value.trim())
    $("interaction-message").textContent = "";
  if (phase === "empty")
    $("target-stage").textContent = ws.candidates?.length
      ? "Choose result"
      : "Ready";
  const manualReady = ws.mode !== "robot_control" ||
    (control.connected && control.armed && !control.gesture_active);
  const destinationReady = !!ws.destination && ws.can_go && manualReady;
  if (phase === "confirmed") {
    $("target-stage").textContent = destinationReady
      ? "Confirmed"
      : ws.mode === "observe" ? "Observe"
      : ws.mode === "robot_control" ? "Dry run only" : "Refresh required";
    $("go-note").textContent = control.failed
      ? "Connection failed. Reconnect before GO."
      : ws.mode === "observe"
        ? "Observe has no movement authorization."
        : ws.mode === "robot_control"
          ? "GO for the real robot is not included in this version. Use Dry run."
        : !ws.destination
          ? "Authorization cleared. Refresh before GO."
          : !ws.can_go
            ? ws.selected?.age_s > 1.5
              ? "Evidence stale. Refresh before GO."
              : "Preview expired. Refresh before GO."
            : control.demo
              ? "Moves the virtual robot."
              : "Records the exact SE2 command. No motion.";
  }
  if (phase === "candidate") $("target-stage").textContent = "Candidate";
  document.querySelector(".destination .section-heading h2").textContent =
    destinationReady ? "Destination ready" : "Review destination";
  $("live-world-title").textContent = ws.destination
    ? "Destination / top view"
    : "World / top view";
  scheduleInspectorOverflow();
}
$("fixture-toggle").addEventListener("click", () => {
  reviewFixture = !reviewFixture;
  updateWorkstation(workspaceState, state);
});
$("camera-select").addEventListener("change", () => {
  if (workspaceState) updateWorkstation(workspaceState, state);
});
$("review-route").addEventListener("click", () => revealLivePanel("world"));
$("live-inspector-toggle").addEventListener("click", () =>
  liveDock.toggle("inspector"),
);
liveDock.onChange(() =>
  $("live-inspector-toggle").setAttribute(
    "aria-expanded",
    String(liveDock.visible("inspector")),
  ),
);
worldDock.onChange(() =>
  $("world-inspector-toggle").setAttribute(
    "aria-expanded",
    String(worldDock.visible("entity")),
  ),
);
function showDrawer(id) {
  clearMovement();
  if (id === "diagnostic-drawer") $(id).show();
  else $(id).showModal();
}
// Robot controls share the Live inspector; switching tabs clears held movement.
function selectInspectorTab(name) {
  const column = document.querySelector(".target-column");
  if (column.dataset.tab === name) return;
  clearMovement();
  column.dataset.tab = name;
  for (const tab of document.querySelectorAll("[data-inspector-tab]")) {
    const selected = tab.dataset.inspectorTab === name;
    tab.setAttribute("aria-selected", String(selected));
    tab.tabIndex = selected ? 0 : -1;
    $(tab.getAttribute("aria-controls")).hidden = !selected;
  }
  scheduleInspectorOverflow();
}
for (const tab of document.querySelectorAll("[data-inspector-tab]")) {
  tab.addEventListener("click", () => selectInspectorTab(tab.dataset.inspectorTab));
  tab.addEventListener("keydown", (event) => {
    if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
    event.preventDefault();
    const next = tab.dataset.inspectorTab === "target" ? "robot" : "target";
    selectInspectorTab(next);
    $(`tab-${next}`).focus();
  });
}
for (const id of [
  "footer-diagnostics",
  "world-diagnostics",
  "debug-diagnostics",
])
  $(id).addEventListener("click", () => {
    showDrawer("diagnostic-drawer");
    if (workspaceState) updateWorkspace(workspaceState, state);
  });
$("close-diagnostics").addEventListener("click", () =>
  $("diagnostic-drawer").close(),
);
$("diagnostic-drawer").addEventListener("cancel", clearMovement);
$("connection-tool").addEventListener("click", () =>
  showDrawer("connection-menu"),
);
$("connection-status").addEventListener("click", () =>
  showDrawer("connection-menu"),
);
document.addEventListener("keydown", (event) => {
  if (event.key !== "Escape") return;
  if ($("diagnostic-drawer").open) {
    clearMovement();
    $("diagnostic-drawer").close();
  }
});
$("close-connection-menu").addEventListener("click", () =>
  $("connection-menu").close(),
);
for (const id of ["demo-button", "connect-button"])
  $(id).addEventListener("click", () => $("connection-menu").close());
$("live-3d").addEventListener("click", openAdvanced3D);
$("live-fit").addEventListener("click", () => {
  mapSignature = "";
  if (workspaceState) drawMap($("route-map"), workspaceState);
});
for (const [id, delta] of [
  ["map-zoom-in", 0.25],
  ["map-zoom-out", -0.25],
  ["map-fit", 0],
])
  $(id).addEventListener("click", () => {
    worldZoom = delta ? Math.max(0.5, Math.min(3, worldZoom + delta)) : 1;
    mapSignature = "";
    if (workspaceState) drawMap($("main-map"), workspaceState, true);
  });
let resizeQueued = false;
let inspectorOverflowQueued = false;
function scheduleInspectorOverflow() {
  if (inspectorOverflowQueued) return;
  inspectorOverflowQueued = true;
  requestAnimationFrame(() => {
    inspectorOverflowQueued = false;
    const body = document.querySelector(".target-body");
    $("inspector-overflow").hidden = body.scrollHeight <= body.clientHeight + body.scrollTop + 8;
  });
}
const inspectorBody = document.querySelector(".target-body");
new ResizeObserver(scheduleInspectorOverflow).observe(inspectorBody);
inspectorBody.addEventListener("scroll", scheduleInspectorOverflow, { passive: true });
inspectorBody.addEventListener("toggle", scheduleInspectorOverflow, true);
const mapResize = new ResizeObserver(() => {
  if (resizeQueued) return;
  resizeQueued = true;
  requestAnimationFrame(() => {
    resizeQueued = false;
    if (workspaceState) {
      drawMap($("route-map"), workspaceState);
      drawMap($("main-map"), workspaceState, true);
    }
  });
});
mapResize.observe($("route-map"));
mapResize.observe($("main-map"));

$("world-inspector-toggle").addEventListener("click", () =>
  worldDock.toggle("entity"),
);

// Modal setup dialogs retain an immediately reachable Stop inside the top layer.
for (const dialog of document.querySelectorAll(
  "dialog:not(.manual-drawer):not(.diagnostic-drawer)",
)) {
  const stop = document.createElement("button");
  stop.type = "button";
  stop.className = "dialog-stop";
  stop.textContent = "STOP MOVEMENT";
  stop.addEventListener("click", () => $("stop-button").click());
  dialog.querySelector(".dialog-head").append(stop);
}
async function openAdvanced3D() {
  showDrawer("rerun-dialog");
  let list = $("recording-options");
  if (!list) {
    list = document.createElement("div");
    list.id = "recording-options";
    $("rerun-dialog").querySelector(".dialog-head").after(list);
  }
  const loading = document.createElement("p");
  loading.textContent = "Loading local recordings…";
  list.replaceChildren(loading);
  try {
    const entries = (await api("/api/runs")).artifacts
      .filter((item) => item.name.endsWith(".rrd"))
      .slice(0, 12);
    list.replaceChildren();
    if (!entries.length) {
      const note = document.createElement("p");
      note.textContent =
        "No recorded 3D found. Record or replay using the commands below.";
      list.append(note);
    }
    for (const item of entries) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "recording-option";
      const name = document.createElement("strong"),
        run = document.createElement("span"),
        arrow = document.createElement("span");
      name.textContent = item.name;
      run.textContent = item.run;
      run.className = "recording-run";
      arrow.textContent = "Open";
      arrow.className = "recording-arrow";
      button.append(name, run, arrow);
      button.title = "Open actual recording in the native Rerun viewer";
      button.addEventListener("click", async () => {
        try {
          toast((await api("/api/runs/open", { id: item.id })).message);
          $("rerun-dialog").close();
        } catch (error) {
          toast(error.message);
        }
      });
      list.append(button);
    }
  } catch (error) {
    list.replaceChildren();
    const note = document.createElement("p");
    note.textContent = error.message;
    list.append(note);
  }
}
