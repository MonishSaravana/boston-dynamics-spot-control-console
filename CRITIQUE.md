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
