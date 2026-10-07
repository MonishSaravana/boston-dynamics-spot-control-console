/* Dockable panels: drag a panel bar onto another panel's edge to split it,
   onto its centre to swap, or onto the workspace edge to dock along it.
   Splitters resize neighbours. Layout is presentation only and is saved per
   browser; dragging or resizing first clears any held movement. */
const DOCK_MIN = 150;
const DOCK_STACK = matchMedia("(max-width: 650px)");

class Dock {
  constructor(root, { key, tree, order, edges = {} }) {
    this.root = root;
    this.key = key;
    this.defaultTree = tree;
    this.order = order;
    this.edges = edges;
    this.panels = new Map();
    for (const panel of root.querySelectorAll(":scope > [data-panel]"))
      this.panels.set(panel.dataset.panel, panel);
    this.hold = document.createElement("div");
    this.hold.hidden = true;
    this.listeners = new Set();
    this.tree = this.load() || structuredClone(tree);
    for (const panel of this.panels.values()) this.bindBar(panel);
    DOCK_STACK.addEventListener("change", () => this.render());
    this.render();
  }

  load() {
    try {
      const tree = JSON.parse(localStorage.getItem(this.key));
      return tree && this.valid(tree) ? tree : null;
    } catch (_) {
      return null;
    }
  }
  save() {
    try {
      localStorage.setItem(this.key, JSON.stringify(this.tree));
    } catch (_) {
      /* Layout persistence is a convenience only. */
    }
    for (const listener of this.listeners) listener();
  }
  valid(node, seen = new Set()) {
    if (typeof node?.panel === "string") {
      if (!this.panels.has(node.panel) || seen.has(node.panel)) return false;
      seen.add(node.panel);
      return true;
    }
    return (
      ["row", "column"].includes(node?.split) &&
      Array.isArray(node.children) &&
      node.children.length > 1 &&
      Array.isArray(node.sizes) &&
      node.sizes.length === node.children.length &&
      node.sizes.every((size) => Number.isFinite(size) && size > 0) &&
      node.children.every((child) => this.valid(child, seen))
    );
  }
  ids(node = this.tree, out = []) {
    if (!node) return out;
    if (node.panel) out.push(node.panel);
    else for (const child of node.children) this.ids(child, out);
    return out;
  }
  visible(id) {
    return this.ids().includes(id);
  }
  onChange(listener) {
    this.listeners.add(listener);
    listener();
  }

  // Tree edits always return a normalised tree: no single-child splits and no
  // same-direction nesting, so sizes stay proportional to what is on screen.
  normalise(node) {
    if (!node || node.panel) return node;
    const children = [],
      sizes = [];
    node.children.forEach((child, i) => {
      const next = this.normalise(child);
      if (!next) return;
      if (next.split === node.split) {
        const total = next.sizes.reduce((a, b) => a + b, 0);
        next.children.forEach((grand, j) => {
          children.push(grand);
          sizes.push((node.sizes[i] * next.sizes[j]) / total);
        });
      } else {
        children.push(next);
        sizes.push(node.sizes[i]);
      }
    });
    if (!children.length) return null;
    if (children.length === 1) return children[0];
    const total = sizes.reduce((a, b) => a + b, 0);
    return { split: node.split, children, sizes: sizes.map((s) => s / total) };
  }
  without(node, id) {
    if (!node) return null;
    if (node.panel) return node.panel === id ? null : node;
    const kept = [],
      sizes = [];
    node.children.forEach((child, i) => {
      const next = this.without(child, id);
      if (next) {
        kept.push(next);
        sizes.push(node.sizes[i]);
      }
    });
    return this.normalise({ split: node.split, children: kept, sizes });
  }
  beside(node, target, id, edge) {
    if (node.panel === target) {
      const split = edge === "left" || edge === "right" ? "row" : "column";
      const first = edge === "left" || edge === "top";
      return {
        split,
        children: first ? [{ panel: id }, node] : [node, { panel: id }],
        sizes: [0.5, 0.5],
      };
    }
    if (node.panel) return node;
    return {
      ...node,
      children: node.children.map((child) => this.beside(child, target, id, edge)),
    };
  }
  atRoot(id, edge, share = 0.3) {
    if (!this.tree) return { panel: id };
    const split = edge === "left" || edge === "right" ? "row" : "column";
    const first = edge === "left" || edge === "top";
    return this.normalise({
      split,
      children: first ? [{ panel: id }, this.tree] : [this.tree, { panel: id }],
      sizes: first ? [share, 1 - share] : [1 - share, share],
    });
  }
  swap(node, a, b) {
    if (node.panel) return { panel: node.panel === a ? b : node.panel === b ? a : node.panel };
    return { ...node, children: node.children.map((child) => this.swap(child, a, b)) };
  }

  show(id) {
    if (this.visible(id)) return;
    this.tree = this.atRoot(id, this.edges[id] || "right");
    this.render();
    this.save();
  }
  hide(id) {
    if (!this.visible(id)) return;
    this.tree = this.without(this.tree, id);
    this.render();
    this.save();
  }
  toggle(id, force) {
    if (force ?? !this.visible(id)) this.show(id);
    else this.hide(id);
  }
  reset() {
    this.tree = structuredClone(this.defaultTree);
    this.render();
    this.save();
  }
  flash(id) {
    const panel = this.panels.get(id);
    panel?.classList.remove("dock-flash");
    void panel?.offsetWidth;
    panel?.classList.add("dock-flash");
  }

  render() {
    const stacked = DOCK_STACK.matches;
    this.root.classList.toggle("dock-stacked", stacked);
    for (const panel of this.panels.values()) this.hold.append(panel);
    this.root.replaceChildren(this.hold);
    if (!this.tree) {
      const empty = document.createElement("p");
      empty.className = "dock-empty";
      empty.textContent = "All panels are hidden. Use Layout to show one.";
      this.root.append(empty);
      return;
    }
    if (stacked) {
      const shown = this.ids();
      for (const id of this.order.filter((name) => shown.includes(name)))
        this.root.append(this.leaf(id));
    } else this.root.append(this.build(this.tree));
  }
  leaf(id) {
    const leaf = document.createElement("div");
    leaf.className = "dock-leaf";
    leaf.dataset.leaf = id;
    leaf.append(this.panels.get(id));
    return leaf;
  }
  build(node) {
    if (node.panel) return this.leaf(node.panel);
    const element = document.createElement("div");
    element.className = `dock-split dock-${node.split}`;
    const cells = node.children.map((child, i) => {
      const cell = this.build(child);
      cell.style.flex = `${node.sizes[i]} 1 0px`;
      return cell;
    });
    cells.forEach((cell, i) => {
      element.append(cell);
      if (i < cells.length - 1)
        element.append(this.splitter(node, i, cells[i], cells[i + 1]));
    });
    return element;
  }

  splitter(node, index, before, after) {
    const handle = document.createElement("div");
    const horizontal = node.split === "row";
    handle.className = "dock-splitter";
    handle.tabIndex = 0;
    handle.setAttribute("role", "separator");
    handle.setAttribute("aria-orientation", horizontal ? "vertical" : "horizontal");
    handle.setAttribute("aria-label", "Resize panels");
    const axis = (rect) => (horizontal ? rect.width : rect.height);
    const apply = (delta) => {
      const a = axis(before.getBoundingClientRect()),
        b = axis(after.getBoundingClientRect());
      const total = a + b;
      const next = Math.max(DOCK_MIN, Math.min(total - DOCK_MIN, a + delta));
      const share = node.sizes[index] + node.sizes[index + 1];
      node.sizes[index] = (share * next) / total;
      node.sizes[index + 1] = share - node.sizes[index];
      before.style.flex = `${node.sizes[index]} 1 0px`;
      after.style.flex = `${node.sizes[index + 1]} 1 0px`;
    };
    let last = null;
    handle.addEventListener("pointerdown", (event) => {
      if (event.button !== 0) return;
      clearMovement();
      last = horizontal ? event.clientX : event.clientY;
      handle.setPointerCapture(event.pointerId);
      handle.classList.add("active");
      event.preventDefault();
    });
    handle.addEventListener("pointermove", (event) => {
      if (last === null) return;
      const position = horizontal ? event.clientX : event.clientY;
      apply(position - last);
      last = position;
    });
    const end = () => {
      if (last === null) return;
      last = null;
      handle.classList.remove("active");
      this.save();
    };
    handle.addEventListener("pointerup", end);
    handle.addEventListener("pointercancel", end);
    handle.addEventListener("lostpointercapture", end);
    handle.addEventListener("keydown", (event) => {
      const keys = horizontal ? ["ArrowLeft", "ArrowRight"] : ["ArrowUp", "ArrowDown"];
      if (!keys.includes(event.key)) return;
      event.preventDefault();
      clearMovement();
      apply(event.key === keys[0] ? -24 : 24);
      this.save();
    });
    return handle;
  }

  bindBar(panel) {
    const bar = panel.querySelector(":scope > .panel-bar");
    if (!bar) return;
    const id = panel.dataset.panel;
    let start = null,
      ghost = null,
      zone = null;
    const overlay = document.createElement("div");
    overlay.className = "dock-drop";
    const label = document.createElement("span");
    overlay.append(label);
    const target = (x, y) => {
      const rootRect = this.root.getBoundingClientRect();
      const edgeBand = 28;
      const edges = {
        left: x - rootRect.left,
        right: rootRect.right - x,
        top: y - rootRect.top,
        bottom: rootRect.bottom - y,
      };
      const nearest = Object.entries(edges).sort((a, b) => a[1] - b[1])[0];
      if (nearest[1] >= 0 && nearest[1] < edgeBand && this.ids().length > 1)
        return { root: true, edge: nearest[0], rect: rootRect };
      const leaf = document.elementFromPoint(x, y)?.closest(".dock-leaf");
      if (!leaf || !this.root.contains(leaf) || leaf.dataset.leaf === id) return null;
      const rect = leaf.getBoundingClientRect();
      const nx = (x - rect.left) / rect.width,
        ny = (y - rect.top) / rect.height;
      const sides = { left: nx, right: 1 - nx, top: ny, bottom: 1 - ny };
      const [edge, distance] = Object.entries(sides).sort((a, b) => a[1] - b[1])[0];
      return { leaf: leaf.dataset.leaf, edge: distance < 0.3 ? edge : "center", rect };
    };
    const showZone = (next) => {
      zone = next;
      if (!next) {
        overlay.hidden = true;
        return;
      }
      const { rect, edge } = next;
      const band = next.root ? 0.25 : 0.5;
      let { left, top, width, height } = rect;
      if (edge === "left") width *= band;
      if (edge === "right") (left += width * (1 - band)), (width *= band);
      if (edge === "top") height *= band;
      if (edge === "bottom") (top += height * (1 - band)), (height *= band);
      Object.assign(overlay.style, {
        left: `${left}px`,
        top: `${top}px`,
        width: `${width}px`,
        height: `${height}px`,
      });
      overlay.dataset.edge = edge;
      label.textContent = edge === "center"
        ? `Swap with ${this.panels.get(next.leaf)?.dataset.title || "panel"}`
        : `Dock ${edge}${next.root ? " of workspace" : ""}`;
      overlay.hidden = false;
    };
    const finish = (commit) => {
      if (ghost && commit && zone) {
        if (zone.root) {
          this.tree = this.without(this.tree, id);
          this.tree = this.atRoot(id, zone.edge);
        } else if (zone.edge === "center") {
          this.tree = this.swap(this.tree, id, zone.leaf);
        } else {
          this.tree = this.without(this.tree, id);
          this.tree = this.normalise(this.beside(this.tree, zone.leaf, id, zone.edge));
        }
        this.render();
        this.save();
      }
      ghost?.remove();
      overlay.remove();
      document.body.classList.remove("dock-dragging");
      ghost = null;
      start = null;
      zone = null;
    };
    bar.addEventListener("pointerdown", (event) => {
      if (
        event.button !== 0 ||
        DOCK_STACK.matches ||
        event.target.closest("button, input, select, a, label, output, [role='tab']")
      )
        return;
      start = { x: event.clientX, y: event.clientY };
      bar.setPointerCapture(event.pointerId);
    });
    bar.addEventListener("pointermove", (event) => {
      if (!start) return;
      if (!ghost) {
        if (Math.hypot(event.clientX - start.x, event.clientY - start.y) < 5) return;
        clearMovement();
        ghost = document.createElement("div");
        ghost.className = "dock-ghost";
        ghost.textContent = panel.dataset.title || id;
        overlay.hidden = true;
        document.body.append(overlay, ghost);
        document.body.classList.add("dock-dragging");
      }
      ghost.style.transform = `translate(${event.clientX + 12}px, ${event.clientY + 12}px)`;
      showZone(target(event.clientX, event.clientY));
    });
    bar.addEventListener("pointerup", () => finish(true));
    bar.addEventListener("pointercancel", () => finish(false));
    bar.addEventListener("lostpointercapture", () => ghost && finish(false));
    bar.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && ghost) finish(false);
    });
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && ghost) finish(false);
    });
  }
}

// Layout menus: one per dock, listing panels and a reset action.
function bindLayoutMenu(button, menu, dock, labels) {
  const list = document.createElement("div");
  list.className = "layout-options";
  for (const [id, label] of Object.entries(labels)) {
    const row = document.createElement("label");
    const box = document.createElement("input");
    box.type = "checkbox";
    box.dataset.panel = id;
    box.addEventListener("change", () => dock.toggle(id, box.checked));
    row.append(box, document.createTextNode(label));
    list.append(row);
  }
  const reset = document.createElement("button");
  reset.type = "button";
  reset.textContent = "Reset layout";
  reset.addEventListener("click", () => {
    dock.reset();
    menu.hidePopover();
  });
  const hint = document.createElement("p");
  hint.textContent = "Drag a panel bar onto an edge to dock it, or onto a panel to swap.";
  menu.append(list, reset, hint);
  dock.onChange(() => {
    for (const box of list.querySelectorAll("input"))
      box.checked = dock.visible(box.dataset.panel);
  });
  menu.addEventListener("toggle", (event) => {
    if (event.newState !== "open") return;
    const rect = button.getBoundingClientRect();
    menu.style.top = `${rect.bottom + 6}px`;
    menu.style.left = `${Math.max(8, rect.right - menu.offsetWidth)}px`;
  });
}

const liveDock = new Dock(document.getElementById("live-dock"), {
  key: "scope.layout.live.v1",
  order: ["camera", "inspector", "world", "model"],
  edges: { camera: "left", world: "bottom", model: "bottom", inspector: "right" },
  tree: {
    split: "row",
    sizes: [0.76, 0.24],
    children: [
      {
        split: "column",
        sizes: [0.56, 0.44],
        children: [
          { panel: "camera" },
          {
            split: "row",
            sizes: [0.42, 0.58],
            children: [{ panel: "world" }, { panel: "model" }],
          },
        ],
      },
      { panel: "inspector" },
    ],
  },
});
const worldDock = new Dock(document.getElementById("world-dock"), {
  key: "scope.layout.world.v1",
  order: ["map", "entity", "layers"],
  edges: { layers: "left", entity: "right", map: "left" },
  tree: {
    split: "row",
    sizes: [0.14, 0.62, 0.24],
    children: [{ panel: "layers" }, { panel: "map" }, { panel: "entity" }],
  },
});
bindLayoutMenu(
  document.getElementById("live-layout-button"),
  document.getElementById("live-layout-menu"),
  liveDock,
  { camera: "Camera", world: "World map", model: "Robot model", inspector: "Inspector" },
);
bindLayoutMenu(
  document.getElementById("world-layout-button"),
  document.getElementById("world-layout-menu"),
  worldDock,
  { layers: "Scene layers", map: "Map", entity: "Entity inspector" },
);
