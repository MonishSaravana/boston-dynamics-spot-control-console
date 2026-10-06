const token = document.querySelector('meta[name="scope-token"]').content;
const $ = (id) => document.getElementById(id);
const names = {
  evidence: "Perception evidence",
  frontleft_fisheye_image: "Front left",
  frontright_fisheye_image: "Front right",
  left_fisheye_image: "Left",
  right_fisheye_image: "Right",
  back_fisheye_image: "Back",
  split: "All cameras",
  panorama: "Front panorama",
};
const controller = crypto.randomUUID();
let state = null;
let driveSequence = 0;
let held = new Set();
let previousVersions = {};
let previousModelVersion = null;
let previousView = null;
let polling = false;
let toastTimer;
const objectUrls = new Map();

async function api(path, data) {
  const response = await fetch(path, {
    method: data === undefined ? "GET" : "POST",
    headers: {
      "X-SCOPE-Token": token,
      ...(data === undefined ? {} : { "Content-Type": "application/json" }),
    },
    body: data === undefined ? undefined : JSON.stringify(data),
    cache: "no-store",
    keepalive: data?.name === "stop",
  });
  const result = await response.json();
  if (!response.ok)
    throw new Error(result.error || `Request failed (${response.status})`);
  return result;
}

function toast(message) {
  const box = $("toast");
  box.textContent = message;
  box.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    box.hidden = true;
  }, 5000);
}

async function command(name, extra = {}) {
  try {
    await api("/api/action", { name, epoch: state?.control_epoch, ...extra });
  } catch (error) {
    toast(error.message);
  }
  await refresh();
}

async function loadImage(name, element) {
  try {
    const response = await fetch(
      name === "model"
        ? "/api/model"
        : name === "evidence" ||
            (state?.workspace?.available && !state?.demo && name !== "panorama")
          ? `/api/workspace/image/${name}`
          : `/api/frame/${name}`,
      {
        headers: { "X-SCOPE-Token": token },
        cache: "no-store",
      },
    );
    if (!response.ok) {
      element.hidden = true;
      if (element === $("camera-image")) $("camera-empty").hidden = false;
      return;
    }
    const url = URL.createObjectURL(await response.blob());
    const old = objectUrls.get(element);
    element.src = url;
    element.hidden = false;
    if (element === $("camera-image")) $("camera-empty").hidden = true;
    objectUrls.set(element, url);
    if (old) URL.revokeObjectURL(old);
  } catch (_) {
    /* The status poll reports connection failures. */
  }
}

function setView() {
  const view = $("camera-select").value;
  const split = view === "split";
  $("camera-single").hidden = split;
  $("camera-split").hidden = !split;
  $("panorama-tools").hidden = view !== "panorama";
  $("image-title").textContent = names[view] || view;
  $("camera-image").alt =
    `${names[view] || view} ${state?.demo ? "simulated image" : "camera feed"}`;
  if (view !== previousView) {
    previousView = view;
    if (
      view !== "split" &&
      (view === "evidence"
        ? state?.workspace?.available
        : view === "panorama"
          ? state?.has_panorama
          : state?.frame_versions[view])
    )
      loadImage(view, $("camera-image"));
  }
}

function buildSplit(cameras) {
  const grid = $("camera-split");
  const signature = cameras.join(",");
  if (grid.dataset.cameras === signature) return;
  grid.dataset.cameras = signature;
  grid.replaceChildren();
  for (const camera of cameras) {
    const tile = document.createElement("div");
    tile.className = "split-tile";
    const image = document.createElement("img");
    image.alt = `${names[camera]} ${state?.demo ? "simulated image" : "camera feed"}`;
    image.hidden = true;
    image.dataset.camera = camera;
    const label = document.createElement("span");
    label.textContent = names[camera] || camera;
    tile.append(image, label);
    grid.append(tile);
    loadImage(camera, image);
  }
}

function clearImages() {
  for (const [element, url] of objectUrls) {
    URL.revokeObjectURL(url);
    element.removeAttribute("src");
    element.hidden = true;
  }
  objectUrls.clear();
  $("camera-empty").hidden = false;
  previousVersions = {};
  previousView = null;
}

function updateState(next) {
  if (
    state &&
    (state.demo !== next.demo ||
      (state.connected && !next.connected) ||
      (state.workspace?.available && !next.workspace?.available))
  )
    clearImages();
  state = next;
  const ws = next.workspace;
  const displayCameras =
    ws?.available && !next.demo
      ? Object.entries(ws.cameras || {})
          .filter(([_, camera]) => camera.displayed && camera.acquired)
          .map(([name]) => name)
      : next.cameras;
  const available = [...displayCameras];
  if (ws?.available) available.unshift("evidence");
  if (displayCameras.length > 1) available.push("split");
  if (
    next.cameras.includes("frontleft_fisheye_image") &&
    next.cameras.includes("frontright_fisheye_image")
  )
    available.push("panorama");
  const select = $("camera-select");
  for (const camera of displayCameras) {
    if (![...select.options].some((option) => option.value === camera)) {
      const option = document.createElement("option");
      option.value = camera;
      option.textContent = names[camera] || camera;
      select.append(option);
    }
  }
  for (const option of select.options)
    option.disabled = !available.includes(option.value);
  if (available.length && !available.includes(select.value))
    select.value = available[0];
  if (!available.length) select.value = "frontleft_fisheye_image";
  select.disabled = !available.length;
  buildSplit(displayCameras);
  setView();

  $("mode-pill").textContent = next.demo ? "Demo data" : "No robot";
  $("mode-pill").classList.toggle("demo", next.demo);
  $("state-dot").className =
    "state-dot" + (next.failed ? " error" : next.armed ? " ready" : "");
  $("state-title").textContent = next.demo
    ? "Simulated imagery · no robot commands"
    : next.failed
      ? "Connection problem · movement disabled"
      : next.armed
        ? "Standing confirmed · movement ready"
        : next.connected
          ? "Connected · movement locked"
          : ws?.available
            ? "Spot sensors connected · no manual commands"
            : "No robot connected";
  $("connection-message").textContent =
    ws?.available && !next.demo && !next.connected
      ? "Spot sensors connected · command authority absent"
      : next.status;
  $("connect-button").textContent = next.demo
    ? "Connect to Spot"
    : next.connected || (ws?.available && !next.demo)
      ? "Disconnect"
      : "Connect to Spot";
  $("connect-button").disabled = next.demo;
  $("demo-button").textContent = next.demo ? "Leave demo" : "Demo data";
  $("demo-button").disabled = next.connected || (ws?.available && !next.demo);
  $("drive-state").textContent = next.armed ? "READY" : "LOCKED";
  $("drive-state").classList.toggle("ready", next.armed);
  $("power-button").disabled =
    next.demo || !next.connected || next.powered || next.failed;
  $("stand-button").disabled =
    next.demo ||
    !next.connected ||
    !next.powered ||
    next.failed ||
    next.gesture_active;
  $("apply-button").disabled =
    next.demo ||
    !next.armed ||
    next.failed ||
    next.gesture_active ||
    held.size > 0;
  $("gesture-toggle").disabled =
    next.demo ||
    !next.armed ||
    next.failed ||
    (!next.gesture_active && !next.gesture_available);
  $("gesture-message").textContent = next.gesture_status;
  if (
    next.gesture_status.startsWith("OFF") ||
    next.gesture_status.includes("disabled")
  )
    $("gesture-toggle").checked = false;
  $("posture-message").textContent = next.posture_status;
  $("model-status").textContent = next.model_status;
  $("image-badge").textContent = next.demo
    ? "SIMULATED"
    : next.connected
      ? "BUILT-IN CAMERA"
      : "NO FEED";
  $("image-badge").classList.toggle("live", !next.demo);
  $("image-meta").textContent = next.demo
    ? "Original test scene · not Spot footage"
    : next.connected
      ? "Spot camera feed"
      : "Connect to see camera images";
  $("camera-empty").textContent = next.connected
    ? "Waiting for first camera image…"
    : next.demo
      ? "Loading simulated image…"
      : "Connect to see camera images";
  $("camera-caption").textContent =
    select.value === "panorama"
      ? next.demo
        ? "Simulated front composite; live stitching is not exercised."
        : next.panorama_status
      : next.demo
        ? "Original grayscale test artwork, not captured from Spot."
        : next.connected
          ? "Built-in fisheye cameras. Frames keep their source grayscale."
          : "Connect to Spot to receive built-in camera images.";
  $("auto-panorama").disabled = next.demo || !next.connected;
  $("stitch-button").disabled = next.demo || !next.connected;
  for (const button of document.querySelectorAll(".direction"))
    button.disabled = !next.armed || next.failed || next.gesture_active;

  const view = select.value;
  if (
    ws?.available &&
    (view === "evidence" || (!next.demo && displayCameras.includes(view)))
  ) {
    if (previousVersions.workspace !== ws.version)
      loadImage(view, $("camera-image"));
  } else if (view === "split") {
    for (const image of document.querySelectorAll(".split-tile img")) {
      const camera = image.dataset.camera;
      if (previousVersions[camera] !== next.frame_versions[camera])
        loadImage(camera, image);
    }
  } else if (view === "panorama") {
    if (
      next.has_panorama &&
      previousVersions.panorama !== next.panorama_version
    )
      loadImage("panorama", $("camera-image"));
  } else if (
    next.frame_versions[view] &&
    previousVersions[view] !== next.frame_versions[view]
  ) {
    loadImage(view, $("camera-image"));
  }
  previousVersions = {
    ...next.frame_versions,
    panorama: next.panorama_version,
    workspace: ws?.version,
  };
  if (previousModelVersion !== next.model_version) {
    previousModelVersion = next.model_version;
    loadImage("model", $("model-image"));
  }
  if ((!next.armed || next.failed) && held.size) clearMovement();
  if (typeof updateWorkspace === "function") updateWorkspace(ws, next);
}

async function refresh() {
  if (polling) return;
  polling = true;
  try {
    updateState(await api("/api/state"));
  } catch (error) {
    $("state-title").textContent = "Local service unavailable";
    clearMovement();
  } finally {
    polling = false;
  }
}

function keyName(event) {
  const key = event.key.toLowerCase();
  return (
    { arrowup: "w", arrowleft: "a", arrowdown: "s", arrowright: "d" }[key] ||
    ("wasd".includes(key) && key.length === 1 ? key : null)
  );
}

function canDrive() {
  return (
    state?.armed &&
    (!state.workspace?.available ||
      (state.workspace.mode === "robot_control" &&
        !state.workspace.navigation_active)) &&
    (!location.hash ||
      ["#operate", "#manual-controls"].includes(location.hash)) &&
    !state.failed &&
    !state.gesture_active &&
    !document.hidden &&
    !$("connect-dialog").open &&
    ![
      "settings-dialog",
      "diagnostic-drawer",
      "connection-menu",
      "rerun-dialog",
    ].some((id) => $(id).open) &&
    !["INPUT", "SELECT", "TEXTAREA"].includes(
      document.activeElement?.tagName,
    ) &&
    !document.activeElement?.closest(".inspector-resizer")
  );
}

function paintHeld() {
  for (const button of document.querySelectorAll(".direction"))
    button.classList.toggle("active", held.has(button.dataset.direction));
  $("apply-button").disabled =
    !state?.armed || state?.failed || state?.gesture_active || held.size > 0;
}

async function sendDrive(released = false) {
  if (!state?.armed) return;
  try {
    await api("/api/drive", {
      controller,
      sequence: ++driveSequence,
      keys: [...held],
      released,
      epoch: state.control_epoch,
    });
  } catch (error) {
    held.clear();
    paintHeld();
    command("stop");
    toast(error.message);
  }
}

function clearMovement() {
  if (
    !held.size &&
    !state?.gesture_active &&
    !state?.workspace?.navigation_active &&
    !$("gesture-toggle").checked
  )
    return;
  held.clear();
  paintHeld();
  command("stop");
}

document.addEventListener("keydown", (event) => {
  const key = keyName(event);
  if (!key || !canDrive()) return;
  event.preventDefault();
  if (!held.has(key)) {
    held.add(key);
    paintHeld();
    sendDrive();
  }
});
document.addEventListener("keyup", (event) => {
  const key = keyName(event);
  if (!key || !held.has(key)) return;
  event.preventDefault();
  held.delete(key);
  paintHeld();
  sendDrive(true);
});
window.addEventListener("blur", clearMovement);
document.addEventListener("visibilitychange", () => {
  if (document.hidden) clearMovement();
});
window.addEventListener("pagehide", clearMovement);
setInterval(() => {
  if (held.size && canDrive()) sendDrive();
}, 100);
setInterval(() => {
  if (
    (state?.gesture_active || $("gesture-toggle").checked) &&
    !document.hidden &&
    document.hasFocus()
  ) {
    api("/api/presence", { controller }).catch(() => {
      $("gesture-toggle").checked = false;
      command("stop");
    });
  }
}, 100);
for (const button of document.querySelectorAll(".direction")) {
  button.addEventListener("pointerdown", (event) => {
    if (!canDrive()) return;
    event.preventDefault();
    button.setPointerCapture(event.pointerId);
    held.add(button.dataset.direction);
    paintHeld();
    sendDrive();
  });
  const release = () => {
    if (!held.delete(button.dataset.direction)) return;
    paintHeld();
    sendDrive(true);
  };
  button.addEventListener("pointerup", release);
  button.addEventListener("pointercancel", release);
  button.addEventListener("lostpointercapture", release);
}

$("stop-button").addEventListener("click", () => {
  held.clear();
  paintHeld();
  command("stop");
});
$("power-button").addEventListener("click", () => command("power"));
$("stand-button").addEventListener("click", () => command("stand"));
$("stitch-button").addEventListener("click", () => command("stitch"));
$("auto-panorama").addEventListener("change", (event) =>
  command("auto_panorama", { enabled: event.target.checked }),
);
$("gesture-toggle").addEventListener("change", (event) =>
  command("gesture", { enabled: event.target.checked, controller }),
);
$("camera-select").addEventListener("change", () => {
  clearMovement();
  setView();
  refresh();
});

for (const id of ["speed", "height", "roll", "pitch"]) {
  $(id).addEventListener("input", () => {
    const n = Number($(id).value);
    $(id + "-value").textContent =
      id === "speed"
        ? `${(n / 100).toFixed(2)} m/s`
        : id === "height"
          ? `+${n} cm`
          : `${n > 0 ? "+" : ""}${n}°`;
  });
}
$("speed").addEventListener("change", () => {
  if (state?.connected)
    command("speed", { value: Number($("speed").value) / 100 });
});
$("apply-button").addEventListener("click", () =>
  command("posture", {
    height: Number($("height").value),
    roll: Number($("roll").value),
    pitch: Number($("pitch").value),
  }),
);

const dialog = $("connect-dialog");
$("demo-button").addEventListener("click", async () => {
  clearMovement();
  try {
    await api("/api/mode", { demo: !state?.demo });
    previousView = null;
    previousVersions = {};
    await refresh();
  } catch (error) {
    toast(error.message);
  }
});
$("connect-button").addEventListener("click", async () => {
  clearMovement();
  if (state?.connected || (state?.workspace?.available && !state?.demo)) {
    {
      try {
        await api("/api/disconnect", {});
        await refresh();
      } catch (error) {
        toast(error.message);
      }
    }
  } else {
    $("connect-mode").value = state?.workspace?.mode || "observe";
    $("authority-row").hidden = $("connect-mode").value !== "robot_control";
    dialog.showModal();
  }
});
$("close-dialog").addEventListener("click", () => dialog.close());
$("cancel-connect").addEventListener("click", () => dialog.close());
$("connect-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const errorLabel = $("connect-error");
  errorLabel.hidden = true;
  try {
    $("connect-submit").disabled = true;
    $("connect-submit").textContent = "Connecting…";
    await api("/api/workspace/connect", {
      hostname: $("robot-host").value.trim(),
      username: $("robot-user").value.trim(),
      password: $("robot-password").value,
      mode: $("connect-mode").value,
      human_pose: $("connect-humans").checked,
      command_authority: $("command-authority").checked,
    });
    $("robot-password").value = "";
    dialog.close();
    await refresh();
  } catch (error) {
    errorLabel.textContent = error.message;
    errorLabel.hidden = false;
  } finally {
    $("connect-submit").disabled = false;
    $("connect-submit").textContent = "Connect";
    $("robot-password").value = "";
  }
});

refresh();
setInterval(refresh, 500);
