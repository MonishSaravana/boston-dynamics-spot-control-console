# Interface review

The explicitly invoked web-designer skill guided three rendered directions,
real-code verification and two rounds with the same design critic. The user
brief takes precedence: a practical desktop robot console, retained technical
capability, no new architecture, no decorative runtime work and no live
robot UI testing. M5 remains READY FOR PHYSICAL VALIDATION.

## Renders

Local ignored artifacts in `output/ui/`:

- Before: `before/1280.png`, `before/1920.png`, using the original committed
  HTML/CSS/JS against the original offline demo server.
- Explorations: `explore/a`, `explore/b`, `explore/c`, at 390 and 1440 px.
- Round 1: `r1-fixed/` plus camera, confirmed, Maps, Runs and Evaluate captures.
- Round 2: `final/operate-normal.png` (1280×720), `operate-full.png`
  (1920×1080), `operate-full-page.png`, `maps.png`, `runs.png`,
  `evaluate.png`, `phone.png`; `film/strip.png`.
- Final mechanical check: `final-scan/`; platform results `final-matrix/`.

These are offline fixtures; no Spot camera images or SDK assets are tracked.
The supplied references guide hierarchy and density, not feature claims.
Direction A won: camera and world-frame evidence as a dark field instrument.
Direction B was a light inspection desk, C a camera-overlay station. Both
would distract from the requested control workflow. See `DIRECTION.md`.

## Two rounds

| Rubric | Round 1 | Round 2 |
| --- | ---: | ---: |
| Concept | 3 | 4 |
| Not category average | 4 | 4 |
| Not a slop face | 4 | 4 |
| First screen | 2 | 4 |
| Typography | 3 | 3 |
| Colour | 4 | 4 |
| Richness | 4 | 4 |
| Structure/rhythm | 3 | 4 |
| Craft | 2 | 4 |
| Responsive | 2 | 4 |
| Everyone's computer | 3 | 4 |
| Signature | 3 | 3 |
| Screenshot test | 3 | 3 |
| Fidelity (additional) | 3 | 4 |

Round 1 identified misleading post-GO wording, excessive completed-target
controls, weak normal-window hierarchy, small map annotations, an unbounded
Runs page and phone density/E-stop-note problems. Round 2 confirmed all
five asks landed. Exact destination, heading, GO and Stop fit at normal
size; the full route at that height requires a short scroll. The browser's
Maps link remains the larger route-inspection surface.

The critic judged Round 2 **Done by the stop rule**: no core score below 3
and 10 of 13 at 4. Scores of 3 are retained honestly. Distinctive motion
and portfolio spectacle were not pursued in a movement-authority interface.

Fixed after the last round, not rescored: moved small route-scale units to
ordinary 12 px UI text and added a nearby Open Maps to review route link to
the destination block, including phone. No third design round.

## Mechanical and runtime checks

Screenshot scans at 390, 768, 1280 and 1920 px have no failures. Twelve
matrix passes cover Windows 125%/150%, High Contrast, Linux font metrics,
Android, Firefox, WebKit desktop/iPhone, 320 px reflow, reduced motion and
expanded desktop/phone labels. Final full matrix: zero failures/warnings;
changed labels are checked again in the relevant partial matrix. Emulated
font metrics do not verify native Windows glyphs, ClearType or OS installs.

The screenshot tools require injected test styles. A temporary test-only
Playwright context uses `bypassCSP`; production retains its original strict
same-origin Content Security Policy, local token, Host and Origin checks.
JavaScript interaction checks use production CSP. One screenshot run reported
an aborted superseded local blob image during frame replacement; no page
exception occurred and the current frame remained visible. No external runtime
fonts, browser framework, image/model downloads or animations on control RPCs.

118 offline unittest cases: 115 passed, 3 optional external-dataset checks skipped.
Qt target and browser integration tests also pass after adapter extraction.
A four-frame existing synthetic recording checks local discovery and Rerun
access, including a successful native viewer inspection of its recorded
point cloud, skeletons, pointing scores, memory and timeline; no new benchmark dataset or research feature was introduced.
Live responsiveness, sensor geometry, command delivery and actual Stop
response remain physical checks, not conclusions of these screenshots.

## Scope decisions

Retained the native Rerun engine instead of constructing a replacement 3D
viewer. Retained old Qt entry points and every existing CLI under Evaluate's
advanced section. Kept the actual synthetic RGB-D fixture and measured/simulated
SDK model labels instead of replacing them with polished imaginary footage.

Only a local SVG favicon/title/description are needed for this loopback app.
Public social-card, app-store and marketing collateral are outside the user
brief. No generated/stock images or paid assets. No decorative animation or
public deployment. Tokens and extension points are documented in the unified
console guide. The look is recorded in the skill history as
`SCOPE|app|object|dark|grotesque|teal|data|topbar`.

# Workstation architecture revision

The review above records the earlier pass. The user rejected that layout;
its passing scores did not establish design approval. This revision replaces
the page architecture and is reviewed against the user's stricter workstation
brief. The same critic is retained and explicitly asked to grade actual pixels
more critically. README screenshots remain unchanged.

Three new directions were rendered at desktop and phone: optical workbench,
light survey desk, and editing suite. The optical workbench won; the light
direction gave text too much weight, and the editing direction failed narrow
reflow and risked confusing conceptual overlays with actual geometry. Evidence
is under ignored `output/workstation/`: before, explore, r1 and r2.

## Workstation round 1

| Rubric | Score |
| --- | ---: |
| Concept | 4 |
| Not category average | 3 |
| Not a slop face | 4 |
| First screen | 4 |
| Typography | 3 |
| Colour | 3 |
| Richness | 3 |
| Structure/rhythm | 4 |
| Craft | 2 |
| Responsive | 2 |
| Everyone's computer | 4 |
| Signature | 3 |
| Screenshot test | 3 |
| Fidelity (additional) | 4 |

The critic found the canvas shell a substantial architectural improvement,
but the pale occupancy, apparent footprint extrusion, crowded robot/destination
labels and buried phone GO remained below the references. Round 1 is **not
done**. Five core scores reached 4, with Craft and Responsive below 3.

Five changes made for round 2:

1. Occupancy and legend share neutral renderer tokens; cell classification and
   AABB extents are unchanged. Footprints use faint fills and crisp outlines.
2. The actual route is a solid mint path; the target-facing guide is a quiet
   dashed neutral line. Small world-coordinate anchors use screen-space label
   leaders, with robot and destination labels separated. Nothing lengthens a
   short route for appearance.
3. Candidate Confirm/Reject and destination GO/Reject share a fixed inspector
   action area. Evidence and metadata scroll above it. Destination heading is
   20 px, redundant target-category text is hidden.
4. Manual content scrolls below a fixed header and includes bottom padding.
   Synthetic alignment is labeled as fixture alignment; physical verification
   remains mandatory for a live pair. Source rates show Hz. Small maps fit the
   available canvas rather than assuming a 320×200 minimum drawing surface.
5. World offers Top view / 3D · Rerun. The recording handoff has a fixed header,
   bounded rows with filename first/run second and a compact footer. Actual
   local recordings open through the existing containment-checked endpoint.

Also: Live and World inspectors now resize with pointer or keyboard and
collapse. Resizing affects screen layout only; focused resize handles block
keyboard drive. The lab-preview selector names the mock preview accurately.

## Workstation verification

Final Python suite after source-policy wiring: 119 cases, 116 passed, three
existing optional external-dataset skips. The 39 focused adapter/web/safety
checks also passed. A meaningful added test verifies that disabling source
processing forwards the existing policy and clears target approval/destination.
No perception or navigation algorithm changed.

Chrome, Firefox and WebKit interaction checks under production CSP pass typed,
pointing and direct selection, Observe blocking, single-use virtual GO, Stop
from a modal, manual/diagnostic drawers, camera modes, memory inspection,
inspector resizing/collapse and retained CLI/Rerun access, with no page errors.
No robot was connected. The actual new 3D-recording button was clicked and its
successful native Rerun view inspected: point cloud, skeletons, pointing scores,
memory and timeline remain visible. This is a four-frame existing synthetic
recording, not new physical evidence.

Round 1 scan is clean at 390/768/1280/1920. Full platform matrix initially
found high-contrast shapes and expanded phone labels; partial reruns verify
both fixes. Two WebKit matrix warnings arise from an in-flight poll as the
test context closes; independent production-CSP WebKit interactions have no
exceptions. A superseded blob-image request abort during frame replacement
does not lose the visible current frame. Native Windows/ClearType, physical
devices, screen readers and live perception/control performance are unverified.

The mock lab photograph is generated, local and separate from sensor data;
only demo review displays it. The true demo map remains a simple rectangle.
Rerun remains a separate native window, not a browser 3D renderer. The history
inspector exposes bounded stored payloads rather than a bespoke event editor.
These limitations are reported rather than disguised with invented geometry.

## Six rendered design tests

- Swap: partial pass. A canvas editor is a familiar shell; the target evidence,
  SE2 destination, explicit units, source controls and separate Stop/E-stop
  semantics make this SCOPE. The shell alone does not establish uniqueness.
- Squint: the camera or world is the first visual field; Stop is the strongest
  persistent control, and the current Confirm/GO action is isolated in the
  inspector footer. Debug intentionally gives commands greater weight.
- Face: the rendered scan finds zero card grids. The dark ground is retained
  from the user's references, but permanent KPI panels, stacked form sections,
  generic status cards, decorative gradients and dashboard introductions are gone.
- Specificity: the inspector width supports evidence values without expanding
  into the work; the 66/72/32 px shell keeps Stop and workspaces stable; mint
  means a target action/route, amber selected geometry, red Stop. Copy names
  actual data, units and requests instead of claiming measured results.
- Screenshot: the camera composition is deliberate; the actual sparse map is
  less visually rich than the supplied point-cloud reference. It is not made
  richer by adding geometry. The real richer recording is visible in Rerun.
- Subtraction: default control, module and setup panels were removed from the
  canvas. Their capabilities remain in contextual inspectors and drawers;
  removing the source/evidence/authority information would weaken review.

Round 2 mechanical results: the scan at 390/768/1280/1920 is clean, and all
12 matrix passes have zero failures and zero warnings. Every first screen,
numbered full-page slice and platform PNG was inspected. Production-CSP
interaction checks cover Chrome, Firefox and WebKit independently of the
scanner's measurement-only CSP bypass. The filmstrip captures the fast
confirmation handoff and its identical complete reduced-motion state; the
short route fade is already settled by the captured frames, not evidence of
an expensive continuous animation. The main capture script waits after a
viewport resize so the compositor/map layout settles before taking evidence.


## Workstation round 2

The same critic opened every state capture, scan slice, matrix and filmstrip.
The canvas architecture resolves the user's card/form rejection, but the
information inside it still needs work. No core score is below 3; only seven
of thirteen reach 4. **Not done by the stop rule.**

| Rubric | Round 1 | Round 2 |
| --- | ---: | ---: |
| Concept | 4 | 4 |
| Not the category average | 3 | 3 |
| Not a slop face | 4 | 4 |
| First screen | 4 | 4 |
| Typography | 3 | 3 |
| Colour | 3 | 4 |
| Richness | 3 | 3 |
| Structure and rhythm | 4 | 4 |
| Craft | 2 | 3 |
| Responsive | 2 | 4 |
| Everyone's computer | 4 | 4 |
| Signature | 3 | 3 |
| Screenshot test | 3 | 3 |
| Fidelity | 4 | 4 |

All five first-round asks landed. The next asks are: synchronous canvas source
metadata, stronger destination number hierarchy, short-inspector context and
a real overflow cue, actual entity labels and legible legend swatches, and a
sequence starting before Confirm. The R2 filmstrip begins after the handoff
has settled; it proves the completed state and reduced-motion parity, not
the transition itself.

Round 3 changes:

1. Camera/World/Split immediately update the footer; Split names camera and
   world together. This changes presentation only, not frame acquisition.
2. Destination coordinates are 16px, heading 14px, supporting details 12px.
   Target identity stays above the destination so approval remains traceable.
3. Confirmed context comes before the optional target chooser; the chooser
   collapses once on phase change. Short phone windows allocate a little more
   height to the inspector, and More ↓ appears only while content remains below.
4. Actual entity IDs get muted collision-aware labels where there is space.
   The selected ID is amber. The three 12px legend samples use their actual
   occupancy fills; a specificity bug in the prior legend was corrected.
5. The filmstrip begins with the actual candidate before Confirm, then captures
   the existing destination handoff and reduced-motion completion. No new
   animation or geometry was added.

Round 3 mechanics: four-width scan at 390/768/1280/1920 has no failures
or rubric warnings. One superseded blob frame request aborted during image
replacement; the current processing frame remains visible in all relevant
captures. The full twelve-pass matrix has zero failures and warnings. Every
PNG and numbered scan slice was inspected. Production-CSP Chrome, Firefox
and WebKit checks still pass the full shared interaction flow; focused checks
add immediate Camera/World/Split metadata, visible destination/GO at 390 and
320 px, and an overflow cue that disappears at the bottom.

## Workstation round 3

All five R2 asks landed. The critic scores nine of thirteen core lines at 4,
with none below 3: **not done by the stop rule**. The completed 320px matrix
capture retained an approval older than the existing ten-second proposal limit.
GO correctly stayed disabled, but the heading and demo note claimed readiness.
That is a presentation defect, not a weakened movement guard.

| Rubric | Round 1 | Round 2 | Round 3 |
| --- | ---: | ---: | ---: |
| Concept | 4 | 4 | 4 |
| Not the category average | 3 | 3 | 3 |
| Not a slop face | 4 | 4 | 4 |
| First screen | 4 | 4 | 4 |
| Typography | 3 | 3 | 4 |
| Colour | 3 | 4 | 4 |
| Richness | 3 | 3 | 3 |
| Structure and rhythm | 4 | 4 | 4 |
| Craft | 2 | 3 | 3 |
| Responsive | 2 | 4 | 4 |
| Everyone's computer | 4 | 4 | 4 |
| Signature | 3 | 3 | 4 |
| Screenshot test | 3 | 3 | 3 |
| Fidelity | 4 | 4 | 4 |

The actual before/after film carries the same target ID through confirmation.
Later identical frames prove a settled state, not an animation duration.
The honest sparse fixture and the separate richer Rerun recording remain limits.

Five changes for the final fourth round:

1. Destination heading reflects current GO readiness; an expired preview reads
   Review destination / Refresh required. Target identity remains visible.
2. The fixed GO footer explains expired evidence/preview, Observe authority,
   cleared authorization and failed connection without changing existing guards.
3. Browser checks wait for the real ten-second expiry, capture fresh and expired
   320×640 and 390×844 states, then verify Refresh restores guarded readiness.
4. Entity labels sit outside their own and adjacent footprints, with short
   leaders to their actual bounds; secondary labels are suppressed if needed.
5. Labels render after footprints. Placement excludes canvas captions, the
   metric scale, other footprints, markers and previously placed labels.
   Hover titles and accessible entity names retain IDs when a label cannot fit.

The R4 scan caught a 17×13px backpack target after labels moved into a separate
layer. A 24px hit area was still fractionally below the threshold after browser
rounding. Invisible areas now use at least 30px, retaining 24px screen bounds in
the isometric projection without changing visible world bounds. Forced-colors
styling explicitly preserves their transparency.

Round 4 mechanics: the final scan at 390/768/1280/1920 has zero failures
and warnings, and the complete twelve-pass platform matrix has zero failures
and warnings. Every first screen, full-page slice, platform PNG and contact
sheet was inspected. The measurement harness waits for the loaded workspace
instead of network idle, which continuous polling cannot reliably reach;
its initial Firefox null-state race was corrected before the full rerun.
Production CSP is unchanged. One superseded frame request aborted during
image replacement; it was not a page exception or missing current frame.

Chrome, Firefox and WebKit pass both the shared interaction flow and actual
ten-second approval expiry/Refresh checks at 320×640 and 390×844. The narrow
checks measure content against the inspector's visible clipping area, not
just the browser bounds. Fresh and expired screenshots were inspected in
both sizes. All major-state captures and the eight confirmation filmstrip
images were inspected. No live robot was used. The unchanged Python result
remains 116 passes and three external-dataset skips out of 119 cases, plus
39 focused adapter/web/safety passes.

## Workstation round 4 — final

The same critic judged all five R3 asks substantially landed. Ten of thirteen
core lines reach 4, with none below 3: **Done by the rubric stop rule**.
That completes this review process; it does not establish user approval,
physical validation or the references' visual richness.

| Rubric | Round 1 | Round 2 | Round 3 | Round 4 |
| --- | ---: | ---: | ---: | ---: |
| Concept | 4 | 4 | 4 | 4 |
| Not the category average | 3 | 3 | 3 | 3 |
| Not a slop face | 4 | 4 | 4 | 4 |
| First screen | 4 | 4 | 4 | 4 |
| Typography | 3 | 3 | 4 | 4 |
| Colour | 3 | 4 | 4 | 4 |
| Richness | 3 | 3 | 3 | 3 |
| Structure and rhythm | 4 | 4 | 4 | 4 |
| Craft | 2 | 3 | 3 | 4 |
| Responsive | 2 | 4 | 4 | 4 |
| Everyone's computer | 4 | 4 | 4 | 4 |
| Signature | 3 | 3 | 4 | 4 |
| Screenshot test | 3 | 3 | 3 | 3 |
| Fidelity | 4 | 4 | 4 | 4 |

The final critique named two concrete defects. Fixed after the last round,
not rescored: short map fitting reserves the caption bands; Destination uses
footprint/caption/scale exclusions and omits its label if none of the valid
placements fits, retaining the actual marker and inspector coordinates.
Disabled buttons in forced colors use system GrayText and a dashed outline.
These are visual changes; GO readiness and actual map geometry are unchanged.

The remaining 3s are deliberate honest limits: the shell is conventional,
the sparse main map has little spatial texture, and it has less screenshot
appeal than the references. The richer real reconstruction stays in native
Rerun. The confirmation film's settled first post-click frame does not prove
animation duration. No fifth critique round is opened.

Verification after the last fix pass: `output/workstation/final/` contains
27 current major-state and fresh/expired approval captures; its four-width
scan and full twelve-pass platform matrix have zero failures and warnings.
Every capture, scan slice, matrix PNG and confirmation film frame was
inspected. At 320px the diagram is smaller to leave its captions clear;
the destination marker and inspector coordinates remain. In the emulated
dark forced-colors theme, system GrayText is green, but the disabled GO now
has the distinct dashed outline. The production-CSP Chrome expiry/Refresh
check passed again after the layout fix. JavaScript syntax and Git whitespace
checks are clean. The Python suite was not repeated after these visual-only
fixes. No live hardware, real Windows ClearType, physical mobile device,
screen reader or live perception/control performance was validated.
