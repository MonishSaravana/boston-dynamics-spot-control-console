/* Interactive Spot model built from the local SDK base URDF.
   Measured mode mirrors the joint angles the server reports from robot
   telemetry (or the labelled simulated zero pose in the demo). Pose preview is
   local only: nothing in this file sends a robot command. Copying a preview to
   the posture request only fills the existing sliders; Apply keeps its checks. */
(() => {
  const DEG = Math.PI / 180;
  const LEGS = ["fl", "fr", "hl", "hr"];
  const LEG_NAMES = { fl: "Front left", fr: "Front right", hl: "Hind left", hr: "Hind right" };
  const PARTS = ["hx", "hy", "kn"];
  const PART_NAMES = { hx: "hip X", hy: "hip Y", kn: "knee" };
  const JOINTS = LEGS.flatMap((leg) => PARTS.map((part) => `${leg}.${part}`));
  const LINK_JOINT = { hip: "hx", uleg: "hy", lleg: "kn" };
  // Approximate standing body height for the preview's Stand pose.
  const NOMINAL_HEIGHT = 0.52;
  const BODY = [
    { key: "height", label: "Height", unit: "cm", min: -25, max: 20, step: 1, scale: 0.01 },
    { key: "roll", label: "Roll", unit: "°", min: -20, max: 20, step: 0.5, scale: DEG },
    { key: "pitch", label: "Pitch", unit: "°", min: -20, max: 20, step: 0.5, scale: DEG },
    { key: "yaw", label: "Yaw", unit: "°", min: -30, max: 30, step: 0.5, scale: DEG },
  ];
  const COLORS = {
    yellow: [0.9, 0.7, 0.16],
    dark: [0.25, 0.27, 0.28],
    accent: [0.24, 0.81, 0.61],
    ghost: [0.75, 0.8, 0.82],
  };

  // ---- 4x4 matrices, column-major ----
  const M = {
    ident: () => new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
    mul(a, b) {
      const out = new Float32Array(16);
      for (let c = 0; c < 4; c++)
        for (let r = 0; r < 4; r++) {
          let sum = 0;
          for (let k = 0; k < 4; k++) sum += a[k * 4 + r] * b[c * 4 + k];
          out[c * 4 + r] = sum;
        }
      return out;
    },
    trans(x, y, z) {
      const m = M.ident();
      m[12] = x;
      m[13] = y;
      m[14] = z;
      return m;
    },
    fromRows(rows, t = [0, 0, 0]) {
      const m = M.ident();
      for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) m[c * 4 + r] = rows[r][c];
      m[12] = t[0];
      m[13] = t[1];
      m[14] = t[2];
      return m;
    },
    axis(axis, angle) {
      const n = Math.hypot(...axis) || 1;
      const [x, y, z] = axis.map((v) => v / n);
      const c = Math.cos(angle),
        s = Math.sin(angle),
        k = 1 - c;
      return M.fromRows([
        [c + x * x * k, x * y * k - z * s, x * z * k + y * s],
        [y * x * k + z * s, c + y * y * k, y * z * k - x * s],
        [z * x * k - y * s, z * y * k + x * s, c + z * z * k],
      ]);
    },
    // URDF fixed-axis roll, pitch, yaw: Rz(yaw) Ry(pitch) Rx(roll).
    rpy(roll, pitch, yaw, t) {
      const [cr, sr, cp, sp, cy, sy] = [
        Math.cos(roll), Math.sin(roll), Math.cos(pitch),
        Math.sin(pitch), Math.cos(yaw), Math.sin(yaw),
      ];
      return M.fromRows(
        [
          [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
          [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
          [-sp, cp * sr, cp * cr],
        ],
        t,
      );
    },
    invRigid(m) {
      const out = M.ident();
      for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) out[c * 4 + r] = m[r * 4 + c];
      for (let r = 0; r < 3; r++)
        out[12 + r] = -(out[r] * m[12] + out[4 + r] * m[13] + out[8 + r] * m[14]);
      return out;
    },
    point(m, [x, y, z]) {
      return [
        m[0] * x + m[4] * y + m[8] * z + m[12],
        m[1] * x + m[5] * y + m[9] * z + m[13],
        m[2] * x + m[6] * y + m[10] * z + m[14],
      ];
    },
    dir(m, [x, y, z]) {
      return [
        m[0] * x + m[4] * y + m[8] * z,
        m[1] * x + m[5] * y + m[9] * z,
        m[2] * x + m[6] * y + m[10] * z,
      ];
    },
    perspective(fovy, aspect, near, far) {
      const f = 1 / Math.tan(fovy / 2),
        m = new Float32Array(16);
      m[0] = f / aspect;
      m[5] = f;
      m[10] = (far + near) / (near - far);
      m[11] = -1;
      m[14] = (2 * far * near) / (near - far);
      return m;
    },
    lookAt(eye, target, up) {
      const sub = (a, b) => a.map((v, i) => v - b[i]);
      const norm = (v) => v.map((x) => x / (Math.hypot(...v) || 1));
      const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
      const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
      const z = norm(sub(eye, target)),
        x = norm(cross(up, z)),
        y = cross(z, x);
      const m = M.ident();
      for (let i = 0; i < 3; i++) {
        m[i * 4] = x[i];
        m[i * 4 + 1] = y[i];
        m[i * 4 + 2] = z[i];
      }
      m[12] = -dot(x, eye);
      m[13] = -dot(y, eye);
      m[14] = -dot(z, eye);
      return m;
    },
  };

  // ---- Kinematics from the URDF ----
  let geo = null; // { joints: {name: {...}}, links: {...}, foot: {leg: [x,y,z]} }
  function prepareGeometry(raw) {
    const joints = {};
    for (const joint of raw.joints)
      joints[joint.name] = { ...joint, origin: M.rpy(...joint.rpy, joint.xyz) };
    const foot = {};
    for (const leg of LEGS) {
      const v = raw.links[`${leg}.lleg`].vertices;
      let minZ = Infinity;
      for (let i = 2; i < v.length; i += 3) minZ = Math.min(minZ, v[i]);
      let sx = 0, sy = 0, n = 0;
      for (let i = 0; i < v.length; i += 3)
        if (v[i + 2] < minZ + 0.008) {
          sx += v[i];
          sy += v[i + 1];
          n++;
        }
      foot[leg] = [sx / n, sy / n, minZ];
    }
    return { joints, links: raw.links, foot };
  }
  function linkTransforms(q) {
    const out = { base: M.ident() };
    for (const name of JOINTS) {
      const joint = geo.joints[name];
      out[joint.child] = M.mul(
        M.mul(out[joint.parent], joint.origin),
        M.axis(joint.axis, q[name] || 0),
      );
    }
    return out;
  }
  function footInBase(leg, a, b, c) {
    const j = (part) => geo.joints[`${leg}.${part}`];
    let m = M.mul(j("hx").origin, M.axis(j("hx").axis, a));
    m = M.mul(M.mul(m, j("hy").origin), M.axis(j("hy").axis, b));
    m = M.mul(M.mul(m, j("kn").origin), M.axis(j("kn").axis, c));
    return M.point(m, geo.foot[leg]);
  }
  const clampJoint = (name, value) =>
    Math.max(geo.joints[name].lower, Math.min(geo.joints[name].upper, value));
  // Damped least squares, three joints per leg; returns the residual in metres.
  function solveLeg(leg, q, target) {
    const names = PARTS.map((part) => `${leg}.${part}`);
    let x = names.map((name) => q[name]);
    let residual = Infinity;
    for (let iteration = 0; iteration < 40; iteration++) {
      const f = footInBase(leg, ...x);
      const e = target.map((t, i) => t - f[i]);
      residual = Math.hypot(...e);
      if (residual < 1e-5) break;
      const h = 1e-4,
        J = [[], [], []];
      for (let k = 0; k < 3; k++) {
        const step = [...x];
        step[k] += h;
        const g = footInBase(leg, ...step);
        for (let r = 0; r < 3; r++) J[r][k] = (g[r] - f[r]) / h;
      }
      const lambda2 = 1e-4;
      const A = [0, 1, 2].map((r) =>
        [0, 1, 2].map((c) => J[r][0] * J[c][0] + J[r][1] * J[c][1] + J[r][2] * J[c][2] + (r === c ? lambda2 : 0)),
      );
      const det =
        A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
        A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
        A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
      if (Math.abs(det) < 1e-14) break;
      const inv = [
        [A[1][1] * A[2][2] - A[1][2] * A[2][1], A[0][2] * A[2][1] - A[0][1] * A[2][2], A[0][1] * A[1][2] - A[0][2] * A[1][1]],
        [A[1][2] * A[2][0] - A[1][0] * A[2][2], A[0][0] * A[2][2] - A[0][2] * A[2][0], A[0][2] * A[1][0] - A[0][0] * A[1][2]],
        [A[1][0] * A[2][1] - A[1][1] * A[2][0], A[0][1] * A[2][0] - A[0][0] * A[2][1], A[0][0] * A[1][1] - A[0][1] * A[1][0]],
      ].map((row) => row.map((v) => v / det));
      const y = inv.map((row) => row[0] * e[0] + row[1] * e[1] + row[2] * e[2]);
      x = x.map((value, k) =>
        clampJoint(names[k], value + J[0][k] * y[0] + J[1][k] * y[1] + J[2][k] * y[2]),
      );
    }
    names.forEach((name, k) => (q[name] = x[k]));
    return residual;
  }

  // ---- State ----
  const model = {
    mode: "measured",
    measured: null,
    demo: false,
    status: "Waiting for measured joint telemetry",
    shown: null,
    preview: null,
    selected: null,
    hover: null,
    unreachable: [],
    camera: { az: 0.75, el: 0.3, dist: 2.1, target: null },
  };
  const bodyMatrix = (body) =>
    M.rpy(body.roll, body.pitch, body.yaw, [0, 0, NOMINAL_HEIGHT + body.height]);
  function standPose() {
    const q = Object.fromEntries(
      JOINTS.map((name) => [name, name.endsWith("hy") ? 0.8 : name.endsWith("kn") ? -1.6 : 0]),
    );
    const feet = {};
    for (const leg of LEGS) {
      const hx = geo.joints[`${leg}.hx`].xyz,
        hy = geo.joints[`${leg}.hy`].xyz;
      solveLeg(leg, q, [hx[0], hx[1] + hy[1], -NOMINAL_HEIGHT]);
      feet[leg] = M.point(bodyMatrix({ height: 0, roll: 0, pitch: 0, yaw: 0 }), footInBase(leg, ...PARTS.map((p) => q[`${leg}.${p}`])));
    }
    return { q, body: { height: 0, roll: 0, pitch: 0, yaw: 0 }, feet };
  }
  // A pose resting on its lowest foot, level, for measured or URDF-zero angles.
  function groundedPose(angles) {
    const q = { ...angles };
    const lowest = Math.min(...LEGS.map((leg) => footInBase(leg, ...PARTS.map((p) => q[`${leg}.${p}`]))[2]));
    const body = { height: -lowest - NOMINAL_HEIGHT, roll: 0, pitch: 0, yaw: 0 };
    const B = bodyMatrix(body);
    const feet = Object.fromEntries(
      LEGS.map((leg) => [leg, M.point(B, footInBase(leg, ...PARTS.map((p) => q[`${leg}.${p}`])))]),
    );
    return { q, body, feet };
  }
  function solveBody() {
    const p = model.preview;
    const inverse = M.invRigid(bodyMatrix(p.body));
    model.unreachable = LEGS.filter((leg) => solveLeg(leg, p.q, M.point(inverse, p.feet[leg])) > 0.01);
  }
  function setJoint(name, value) {
    const p = model.preview;
    p.q[name] = clampJoint(name, value);
    const leg = name.split(".")[0];
    p.feet[leg] = M.point(bodyMatrix(p.body), footInBase(leg, ...PARTS.map((part) => p.q[`${leg}.${part}`])));
    model.unreachable = model.unreachable.filter((item) => item !== leg);
    changed();
  }
  function setBody(key, value) {
    const spec = BODY.find((item) => item.key === key);
    model.preview.body[key] = Math.max(spec.min * spec.scale, Math.min(spec.max * spec.scale, value));
    solveBody();
    changed();
  }

  // ---- WebGL ----
  const canvas = document.getElementById("model-canvas");
  const gl = canvas.getContext("webgl2", { antialias: true, alpha: false });
  let programs = null,
    meshes = {},
    gridBuffer = null,
    pick = null;
  const VS = `#version 300 es
  in vec3 aPos; in vec3 aNormal;
  uniform mat4 uModel, uViewProj;
  out vec3 vNormal; out vec3 vWorld;
  void main() { vec4 w = uModel * vec4(aPos, 1.0); vWorld = w.xyz; vNormal = mat3(uModel) * aNormal; gl_Position = uViewProj * w; }`;
  const FS = `#version 300 es
  precision highp float;
  in vec3 vNormal; in vec3 vWorld;
  uniform vec3 uColor, uTint, uEye; uniform float uTintMix, uAlpha;
  out vec4 outColor;
  void main() {
    vec3 n = normalize(vNormal), v = normalize(uEye - vWorld);
    if (dot(n, v) < 0.0) n = -n;
    float diffuse = max(dot(n, normalize(vec3(0.35, -0.45, 0.85))), 0.0);
    float fill = max(dot(n, normalize(vec3(-0.6, 0.5, 0.3))), 0.0);
    float rim = pow(1.0 - max(dot(n, v), 0.0), 3.0);
    vec3 base = mix(uColor, uTint, uTintMix);
    outColor = vec4(base * (0.26 + 0.2 * (0.5 + 0.5 * n.z) + 0.6 * diffuse + 0.14 * fill) + rim * 0.07, uAlpha);
  }`;
  const PICK_FS = `#version 300 es
  precision highp float;
  uniform float uId; out vec4 outColor;
  void main() { outColor = vec4(uId / 255.0, 0.0, 0.0, 1.0); }`;
  const LINE_VS = `#version 300 es
  in vec3 aPos; in vec4 aColor; uniform mat4 uViewProj; out vec4 vColor;
  void main() { vColor = aColor; gl_Position = uViewProj * vec4(aPos, 1.0); }`;
  const LINE_FS = `#version 300 es
  precision highp float; in vec4 vColor; out vec4 outColor;
  void main() { outColor = vColor; }`;
  function program(vs, fs) {
    const compile = (type, source) => {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
      return shader;
    };
    const p = gl.createProgram();
    gl.attachShader(p, compile(gl.VERTEX_SHADER, vs));
    gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fs));
    // Every vertex array uses location 0 for position and 1 for normal or colour.
    gl.bindAttribLocation(p, 0, "aPos");
    gl.bindAttribLocation(p, 1, vs.includes("aColor") ? "aColor" : "aNormal");
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    const u = {};
    for (let i = 0; i < gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS); i++) {
      const name = gl.getActiveUniform(p, i).name;
      u[name] = gl.getUniformLocation(p, name);
    }
    return { p, u };
  }
  function uploadMeshes() {
    let id = 1;
    for (const [name, link] of Object.entries(geo.links)) {
      const v = link.vertices,
        f = link.faces;
      const data = new Float32Array(f.length * 6);
      for (let t = 0; t < f.length; t += 3) {
        const a = f[t] * 3, b = f[t + 1] * 3, c = f[t + 2] * 3;
        const e1 = [v[b] - v[a], v[b + 1] - v[a + 1], v[b + 2] - v[a + 2]];
        const e2 = [v[c] - v[a], v[c + 1] - v[a + 1], v[c + 2] - v[a + 2]];
        const n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
        const len = Math.hypot(...n) || 1;
        [a, b, c].forEach((index, k) => {
          data.set([v[index], v[index + 1], v[index + 2], n[0] / len, n[1] / len, n[2] / len], (t + k) * 6);
        });
      }
      const vao = gl.createVertexArray();
      gl.bindVertexArray(vao);
      const buffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 24, 0);
      gl.enableVertexAttribArray(1);
      gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 24, 12);
      const yellow = link.color[0] > 0.5;
      meshes[name] = { vao, count: f.length, id: id++, color: yellow ? COLORS.yellow : COLORS.dark };
    }
    gl.bindVertexArray(null);
  }
  function lineBuffer(segments) {
    // segments: [[x,y,z,r,g,b,a, x,y,z,r,g,b,a], ...]
    const data = new Float32Array(segments.flat());
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 28, 0);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 4, gl.FLOAT, false, 28, 12);
    gl.bindVertexArray(null);
    return { vao, buffer, count: data.length / 7 };
  }
  function buildGrid() {
    const segments = [];
    for (let i = -15; i <= 15; i++) {
      const major = i % 5 === 0;
      const alpha = major ? 0.16 : 0.07;
      const c = [0.75, 0.8, 0.83, alpha];
      segments.push([i / 10, -1.5, 0, ...c, i / 10, 1.5, 0, ...c]);
      segments.push([-1.5, i / 10, 0, ...c, 1.5, i / 10, 0, ...c]);
    }
    return lineBuffer(segments);
  }

  // Until the user pans, the view follows the body so any pose stays framed.
  function cameraTarget() {
    if (model.camera.target) return model.camera.target;
    const pose = currentPose();
    return [0.02, 0, pose ? pose.B[14] * 0.62 : 0.32];
  }
  function viewProjection() {
    const { az, el, dist } = model.camera;
    const target = cameraTarget();
    const eye = [
      target[0] + dist * Math.cos(el) * Math.cos(az),
      target[1] + dist * Math.cos(el) * Math.sin(az),
      target[2] + dist * Math.sin(el),
    ];
    const aspect = canvas.width / Math.max(1, canvas.height);
    return { vp: M.mul(M.perspective(35 * DEG, aspect, 0.05, 30), M.lookAt(eye, target, [0, 0, 1])), eye };
  }
  function currentPose() {
    if (model.mode === "preview" && model.preview) return { q: model.preview.q, B: bodyMatrix(model.preview.body) };
    if (!model.shown) return null;
    const grounded = groundedPose(model.shown);
    return { q: grounded.q, B: bodyMatrix(grounded.body) };
  }
  function worldTransforms(pose) {
    const links = linkTransforms(pose.q);
    return Object.fromEntries(Object.entries(links).map(([name, m]) => [name, M.mul(pose.B, m)]));
  }

  let frameQueued = false;
  function requestFrame() {
    if (frameQueued || !programs) return;
    frameQueued = true;
    requestAnimationFrame(draw);
  }
  function draw() {
    frameQueued = false;
    if (!canvas.width || !canvas.height) return;
    // Measured angles ease toward the latest telemetry so 2 Hz updates read smoothly.
    let animating = false;
    if (model.measured && model.shown) {
      for (const name of JOINTS) {
        const delta = model.measured[name] - model.shown[name];
        if (Math.abs(delta) > 1e-4) {
          model.shown[name] += delta * 0.25;
          animating = true;
        } else model.shown[name] = model.measured[name];
      }
      if (animating && model.mode === "measured") syncTable();
    }
    const { vp, eye } = viewProjection();
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(0.027, 0.031, 0.031, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    // Ground grid and foot targets.
    const lines = programs.line;
    gl.useProgram(lines.p);
    gl.uniformMatrix4fv(lines.u.uViewProj, false, vp);
    gl.depthMask(false);
    gl.bindVertexArray(gridBuffer.vao);
    gl.drawArrays(gl.LINES, 0, gridBuffer.count);
    const extras = [];
    const axis = 0.18;
    extras.push([0, 0, 0.001, 0.9, 0.36, 0.33, 0.9, axis, 0, 0.001, 0.9, 0.36, 0.33, 0.9]);
    extras.push([0, 0, 0.001, 0.24, 0.81, 0.61, 0.9, 0, axis, 0.001, 0.24, 0.81, 0.61, 0.9]);
    extras.push([0, 0, 0.001, 0.4, 0.6, 0.95, 0.9, 0, 0, axis, 0.4, 0.6, 0.95, 0.9]);
    if (model.mode === "preview" && model.preview)
      for (const leg of LEGS) {
        const [fx, fy] = model.preview.feet[leg];
        const color = model.unreachable.includes(leg) ? [1, 0.56, 0.6, 0.95] : [0.24, 0.81, 0.61, 0.8];
        for (let i = 0; i < 16; i++) {
          const a = (i / 16) * Math.PI * 2, b = ((i + 1) / 16) * Math.PI * 2, r = 0.04;
          extras.push([fx + r * Math.cos(a), fy + r * Math.sin(a), 0.002, ...color, fx + r * Math.cos(b), fy + r * Math.sin(b), 0.002, ...color]);
        }
      }
    const extraBuffer = lineBuffer(extras);
    gl.bindVertexArray(extraBuffer.vao);
    gl.drawArrays(gl.LINES, 0, extraBuffer.count);
    gl.deleteBuffer(extraBuffer.buffer);
    gl.deleteVertexArray(extraBuffer.vao);
    gl.depthMask(true);

    const pose = currentPose();
    const shaded = programs.mesh;
    gl.useProgram(shaded.p);
    gl.uniformMatrix4fv(shaded.u.uViewProj, false, vp);
    gl.uniform3fv(shaded.u.uEye, eye);
    if (pose) {
      const world = worldTransforms(pose);
      const selectedLink = model.selected ? geo.joints[model.selected].child : null;
      for (const [name, mesh] of Object.entries(meshes)) {
        gl.uniformMatrix4fv(shaded.u.uModel, false, world[name]);
        gl.uniform3fv(shaded.u.uColor, mesh.color);
        gl.uniform3fv(shaded.u.uTint, COLORS.accent);
        gl.uniform1f(shaded.u.uTintMix, name === selectedLink ? 0.55 : name === model.hover ? 0.22 : 0);
        gl.uniform1f(shaded.u.uAlpha, 1);
        gl.bindVertexArray(mesh.vao);
        gl.drawArrays(gl.TRIANGLES, 0, mesh.count);
      }
    }
    // Measured pose as a ghost behind a pose preview, when they differ.
    if (model.mode === "preview" && model.shown && model.measured) {
      const differs = JOINTS.some((name) => Math.abs(model.measured[name] - model.preview.q[name]) > 0.5 * DEG);
      if (differs) {
        const grounded = groundedPose(model.shown);
        const world = worldTransforms({ q: grounded.q, B: bodyMatrix(grounded.body) });
        gl.depthMask(false);
        for (const [name, mesh] of Object.entries(meshes)) {
          gl.uniformMatrix4fv(shaded.u.uModel, false, world[name]);
          gl.uniform3fv(shaded.u.uColor, COLORS.ghost);
          gl.uniform1f(shaded.u.uTintMix, 0);
          gl.uniform1f(shaded.u.uAlpha, 0.14);
          gl.bindVertexArray(mesh.vao);
          gl.drawArrays(gl.TRIANGLES, 0, mesh.count);
        }
        gl.depthMask(true);
      }
    }
    gl.bindVertexArray(null);
    if (animating || drag) requestFrame();
  }

  function pickAt(x, y) {
    const pose = currentPose();
    if (!pose || !programs) return null;
    const w = canvas.width, h = canvas.height;
    if (!pick || pick.w !== w || pick.h !== h) {
      if (pick) {
        gl.deleteFramebuffer(pick.fb);
        gl.deleteTexture(pick.tex);
        gl.deleteRenderbuffer(pick.depth);
      }
      const fb = gl.createFramebuffer(), tex = gl.createTexture(), depth = gl.createRenderbuffer();
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
      gl.bindRenderbuffer(gl.RENDERBUFFER, depth);
      gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, w, h);
      gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
      gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, depth);
      pick = { fb, tex, depth, w, h };
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, pick.fb);
    gl.viewport(0, 0, w, h);
    gl.disable(gl.BLEND);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    const { vp } = viewProjection();
    const world = worldTransforms(pose);
    gl.useProgram(programs.pick.p);
    gl.uniformMatrix4fv(programs.pick.u.uViewProj, false, vp);
    for (const [name, mesh] of Object.entries(meshes)) {
      gl.uniformMatrix4fv(programs.pick.u.uModel, false, world[name]);
      gl.uniform1f(programs.pick.u.uId, mesh.id);
      gl.bindVertexArray(mesh.vao);
      gl.drawArrays(gl.TRIANGLES, 0, mesh.count);
    }
    const pixel = new Uint8Array(4);
    const scale = canvas.width / canvas.clientWidth;
    gl.readPixels(Math.round(x * scale), Math.round(h - y * scale), 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixel);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindVertexArray(null);
    gl.enable(gl.BLEND);
    return Object.entries(meshes).find(([, mesh]) => mesh.id === pixel[0])?.[0] || null;
  }
  function project(point, vp) {
    const [x, y, z] = point;
    const cx = vp[0] * x + vp[4] * y + vp[8] * z + vp[12];
    const cy = vp[1] * x + vp[5] * y + vp[9] * z + vp[13];
    const cw = vp[3] * x + vp[7] * y + vp[11] * z + vp[15];
    return [((cx / cw) * 0.5 + 0.5) * canvas.clientWidth, (1 - ((cy / cw) * 0.5 + 0.5)) * canvas.clientHeight];
  }
  // Screen-space motion of the foot for a small rotation of this joint.
  function jointTangent(name) {
    const pose = currentPose();
    const world = worldTransforms(pose);
    const joint = geo.joints[name];
    const leg = name.split(".")[0];
    const frame = world[joint.child];
    const pivot = M.point(frame, [0, 0, 0]);
    const axis = M.dir(frame, joint.axis);
    const foot = M.point(world[`${leg}.lleg`], geo.foot[leg]);
    const eps = 0.01;
    const rotate = M.axis(axis, eps);
    const moved = M.point(rotate, foot.map((v, i) => v - pivot[i])).map((v, i) => v + pivot[i]);
    const { vp } = viewProjection();
    const a = project(foot, vp), b = project(moved, vp);
    return [(b[0] - a[0]) / eps, (b[1] - a[1]) / eps];
  }

  // ---- Pointer interaction ----
  let drag = null;
  const jointForLink = (link) => {
    const [leg, part] = link.split(".");
    return part ? `${leg}.${LINK_JOINT[part]}` : null;
  };
  canvas.addEventListener("contextmenu", (event) => event.preventDefault());
  canvas.addEventListener("pointerdown", (event) => {
    if (!programs) return;
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left, y = event.clientY - rect.top;
    const hit = event.button === 0 && !event.shiftKey ? pickAt(x, y) : null;
    const editable = model.mode === "preview" && model.preview;
    if (hit && hit !== "base") {
      model.selected = jointForLink(hit);
      drag = editable ? { kind: "joint", name: model.selected, x: event.clientX, y: event.clientY, tangent: jointTangent(model.selected) } : { kind: "orbit", x: event.clientX, y: event.clientY };
    } else if (hit === "base" && editable) {
      drag = { kind: "body", x: event.clientX, y: event.clientY, roll: event.altKey };
    } else {
      drag = { kind: event.button === 2 || event.shiftKey ? "pan" : "orbit", x: event.clientX, y: event.clientY };
    }
    canvas.setPointerCapture(event.pointerId);
    canvas.classList.add("dragging");
    syncSelection();
    requestFrame();
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!drag) {
      if (model.mode === "preview" && event.pointerType === "mouse") {
        const rect = canvas.getBoundingClientRect();
        const hover = pickAt(event.clientX - rect.left, event.clientY - rect.top);
        if (hover !== model.hover) {
          model.hover = hover;
          canvas.style.cursor = hover ? "grab" : "";
          requestFrame();
        }
      }
      return;
    }
    const dx = event.clientX - drag.x, dy = event.clientY - drag.y;
    drag.x = event.clientX;
    drag.y = event.clientY;
    const cam = model.camera;
    if (drag.kind === "orbit") {
      cam.az -= dx * 0.008;
      cam.el = Math.max(-0.1, Math.min(1.45, cam.el + dy * 0.008));
    } else if (drag.kind === "pan") {
      const s = cam.dist * 0.0012;
      cam.target = cam.target || [...cameraTarget()];
      cam.target[0] += (dx * Math.sin(cam.az) - dy * Math.cos(cam.az) * Math.sin(cam.el)) * s;
      cam.target[1] += (-dx * Math.cos(cam.az) - dy * Math.sin(cam.az) * Math.sin(cam.el)) * s;
      cam.target[2] += dy * Math.cos(cam.el) * s;
    } else if (drag.kind === "joint") {
      const [tx, ty] = drag.tangent;
      const length2 = tx * tx + ty * ty;
      const delta = length2 > 4 ? (dx * tx + dy * ty) / length2 : -dy * 0.01;
      setJoint(drag.name, model.preview.q[drag.name] + Math.max(-0.2, Math.min(0.2, delta)));
      drag.tangent = jointTangent(drag.name);
    } else if (drag.kind === "body") {
      const body = model.preview.body;
      if (drag.roll || event.altKey) setBody("roll", body.roll + dx * 0.004);
      else {
        setBody("yaw", body.yaw - dx * 0.004);
        setBody("pitch", body.pitch + dy * 0.004);
      }
    }
    requestFrame();
  });
  const endDrag = () => {
    drag = null;
    canvas.classList.remove("dragging");
  };
  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);
  canvas.addEventListener("pointerleave", () => {
    if (!drag && model.hover) {
      model.hover = null;
      requestFrame();
    }
  });
  canvas.addEventListener(
    "wheel",
    (event) => {
      event.preventDefault();
      model.camera.dist = Math.max(0.7, Math.min(6, model.camera.dist * Math.exp(event.deltaY * 0.0012)));
      requestFrame();
    },
    { passive: false },
  );
  const resetView = () => {
    model.camera = { az: 0.75, el: 0.3, dist: 2.1, target: null };
    requestFrame();
  };
  canvas.addEventListener("dblclick", resetView);
  document.getElementById("model-reset-view").addEventListener("click", resetView);
  new ResizeObserver(() => {
    const ratio = Math.min(2, devicePixelRatio || 1);
    canvas.width = Math.round(canvas.clientWidth * ratio);
    canvas.height = Math.round(canvas.clientHeight * ratio);
    requestFrame();
  }).observe(canvas);

  // ---- Controls ----
  const bodyInputs = {};
  for (const spec of BODY) {
    const row = document.createElement("div");
    row.className = "model-row";
    const label = document.createElement("label");
    label.textContent = spec.label;
    label.htmlFor = `model-body-${spec.key}`;
    const range = document.createElement("input");
    range.type = "range";
    Object.assign(range, { min: spec.min, max: spec.max, step: spec.step });
    range.setAttribute("aria-label", `${spec.label} ${spec.unit}`);
    const number = document.createElement("input");
    number.type = "number";
    number.id = `model-body-${spec.key}`;
    Object.assign(number, { min: spec.min, max: spec.max, step: spec.step });
    const unit = document.createElement("span");
    unit.textContent = spec.unit;
    for (const input of [range, number])
      input.addEventListener("input", () => {
        if (input.value === "" || !model.preview) return;
        setBody(spec.key, Number(input.value) * spec.scale);
      });
    row.append(label, range, number, unit);
    document.getElementById("model-body-rows").append(row);
    bodyInputs[spec.key] = { range, number, spec };
  }
  const jointInputs = {};
  for (const leg of LEGS) {
    const row = document.createElement("tr");
    const head = document.createElement("th");
    head.scope = "row";
    head.textContent = leg.toUpperCase();
    head.title = LEG_NAMES[leg];
    row.append(head);
    for (const part of PARTS) {
      const name = `${leg}.${part}`;
      const cell = document.createElement("td");
      const input = document.createElement("input");
      input.type = "number";
      input.step = "1";
      input.setAttribute("aria-label", `${LEG_NAMES[leg]} ${PART_NAMES[part]} degrees`);
      input.addEventListener("focus", () => {
        model.selected = name;
        syncSelection();
        requestFrame();
      });
      input.addEventListener("input", () => {
        if (input.value === "" || !model.preview) return;
        setJoint(name, Number(input.value) * DEG);
      });
      cell.append(input);
      row.append(cell);
      jointInputs[name] = input;
    }
    document.getElementById("joint-rows").append(row);
  }
  const fmt = (value, digits = 1) => (Math.abs(value) < 0.05 ? 0 : value).toFixed(digits);
  function syncTable() {
    const q = model.mode === "preview" && model.preview ? model.preview.q : model.shown;
    for (const name of JOINTS) {
      const input = jointInputs[name];
      if (document.activeElement === input) continue;
      input.value = q ? fmt(q[name] / DEG) : "";
    }
    if (model.mode === "preview" && model.preview)
      for (const { range, number, spec } of Object.values(bodyInputs)) {
        const value = model.preview.body[spec.key] / spec.scale;
        if (document.activeElement !== range) range.value = value;
        if (document.activeElement !== number) number.value = fmt(value, spec.step < 1 ? 1 : 0);
      }
    syncSelection();
  }
  function syncSelection() {
    for (const [name, input] of Object.entries(jointInputs))
      input.closest("td").classList.toggle("selected", name === model.selected);
    const readout = document.getElementById("model-readout");
    const q = model.mode === "preview" && model.preview ? model.preview.q : model.shown;
    if (!model.selected || !q || !geo) {
      readout.hidden = true;
      return;
    }
    const [leg, part] = model.selected.split(".");
    const joint = geo.joints[model.selected];
    readout.textContent = `${model.selected} · ${LEG_NAMES[leg]} ${PART_NAMES[part]} ${fmt(q[model.selected] / DEG)}°  (${fmt(joint.lower / DEG, 0)} to ${fmt(joint.upper / DEG, 0)})`;
    readout.hidden = false;
  }
  function changed() {
    syncTable();
    const note = document.getElementById("model-note");
    note.textContent = model.unreachable.length
      ? `Out of reach for ${model.unreachable.map((leg) => LEG_NAMES[leg].toLowerCase()).join(", ")}; the leg stops at its joint limit.`
      : model.mode === "preview"
        ? "Local preview. Nothing here is sent to Spot."
        : "Measured joints follow robot telemetry. Pose preview is local and sends nothing to Spot.";
    note.classList.toggle("warning", model.unreachable.length > 0);
    requestFrame();
  }
  function setMode(mode) {
    model.mode = mode;
    if (mode === "preview" && !model.preview && geo)
      model.preview = model.measured ? groundedPose(model.measured) : standPose();
    for (const button of document.querySelectorAll("[data-model-mode]"))
      button.setAttribute("aria-pressed", String(button.dataset.modelMode === mode));
    const panel = document.querySelector(".model-panel");
    panel.dataset.mode = mode;
    for (const input of [...Object.values(jointInputs), ...Object.values(bodyInputs).flatMap((i) => [i.range, i.number])])
      input.disabled = mode !== "preview" || !geo;
    for (const id of ["model-stand", "model-zero", "model-copy-posture"])
      document.getElementById(id).disabled = mode !== "preview" || !geo;
    document.getElementById("model-from-measured").disabled = mode !== "preview" || !model.measured;
    model.hover = null;
    canvas.style.cursor = "";
    syncTag();
    changed();
  }
  function syncTag() {
    const tag = document.getElementById("model-tag");
    const status = document.getElementById("model-status");
    if (model.mode === "preview") {
      tag.textContent = "Preview · not sent";
      tag.dataset.state = "preview";
      status.textContent = model.measured
        ? "Pose preview. Ghost shows the measured pose."
        : "Pose preview. No measured pose to compare.";
    } else {
      tag.textContent = model.measured ? (model.demo ? "Simulated" : "Measured") : "No telemetry";
      tag.dataset.state = model.measured ? (model.demo ? "simulated" : "measured") : "none";
      status.textContent = model.status;
    }
  }
  for (const button of document.querySelectorAll("[data-model-mode]"))
    button.addEventListener("click", () => setMode(button.dataset.modelMode));
  document.getElementById("model-from-measured").addEventListener("click", () => {
    if (!model.measured) return;
    model.preview = groundedPose(model.measured);
    model.unreachable = [];
    changed();
  });
  document.getElementById("model-stand").addEventListener("click", () => {
    model.preview = standPose();
    model.unreachable = [];
    changed();
  });
  document.getElementById("model-zero").addEventListener("click", () => {
    model.preview = groundedPose(Object.fromEntries(JOINTS.map((name) => [name, 0])));
    model.unreachable = [];
    changed();
  });
  document.getElementById("model-copy-posture").addEventListener("click", () => {
    const body = model.preview.body;
    const requested = {
      height: Math.round(body.height * 100),
      roll: Math.round(body.roll / DEG),
      pitch: Math.round(body.pitch / DEG),
    };
    const limits = { height: [0, 10], roll: [-5, 5], pitch: [-5, 5] };
    const clamped = [];
    for (const [key, value] of Object.entries(requested)) {
      const [low, high] = limits[key];
      const next = Math.max(low, Math.min(high, value));
      if (next !== value) clamped.push(key);
      const input = document.getElementById(key);
      input.value = next;
      input.dispatchEvent(new Event("input"));
    }
    liveDock.show("inspector");
    selectInspectorTab("robot");
    toast(
      `${clamped.length ? `Clamped ${clamped.join(", ")} to app limits. ` : ""}Yaw stays preview-only. Review, then Apply after Stand is confirmed.`,
    );
  });

  // Called from app.js with every state poll.
  window.updateModelPanel = (next) => {
    const angles = next.model_angles;
    const valid = angles && JOINTS.every((name) => Number.isFinite(angles[name]));
    model.measured = valid ? { ...angles } : null;
    model.demo = next.demo;
    model.status = next.model_status;
    if (model.measured && !model.shown) model.shown = { ...model.measured };
    if (!model.measured) model.shown = null;
    document.getElementById("model-from-measured").disabled = model.mode !== "preview" || !model.measured;
    syncTag();
    if (model.mode === "measured") syncTable();
    requestFrame();
  };

  async function start() {
    try {
      if (!gl) throw new Error("WebGL2 unavailable");
      const response = await fetch("/api/model/geometry", {
        headers: { "X-SCOPE-Token": token },
        cache: "no-store",
      });
      if (!response.ok) throw new Error((await response.json()).error);
      geo = prepareGeometry(await response.json());
      programs = { mesh: program(VS, FS), pick: program(VS, PICK_FS), line: program(LINE_VS, LINE_FS) };
      uploadMeshes();
      gridBuffer = buildGrid();
      for (const name of JOINTS) {
        const joint = geo.joints[name];
        Object.assign(jointInputs[name], {
          min: (joint.lower / DEG).toFixed(1),
          max: (joint.upper / DEG).toFixed(1),
        });
      }
      setMode(model.mode);
      requestFrame();
    } catch (error) {
      window.modelFallback = true;
      const panel = document.querySelector(".model-panel");
      panel.classList.add("model-unavailable");
      document.getElementById("model-image").hidden = false;
      document.getElementById("model-note").textContent = `Interactive model unavailable: ${error.message}`;
      setMode("measured");
    }
  }
  start();
})();
