# Project preferences

- SCOPE is a working Spot control console and an independent project, not a Boston Dynamics product. Present it as a practical engineering repository, not a product pitch.
- Keep the README useful to someone trying to understand or run the current code: start with what works and its limits, give copyable setup and launch commands, and explain controls in concrete units. Separate current behavior, offline demo behavior, and future plans.
- Write human like, natural copy. Avoid generic introductions, marketing adjectives, filler, repeated summaries, personal backstory, and capabilities the code does not show. Do not claim the code was written entirely by hand or write for AI-detector avoidance.
- Be precise about evidence. Distinguish requested posture from measured robot state, illustrative camera imagery from Spot captures, maintainer-reported live use from checks performed in this workspace, and checked platforms from installation guidance for untested platforms.
- Keep the Spot SDK and third-party model bundles out of the project history unless their redistribution terms are resolved. Preserve the independent-project attribution and avoid language implying Boston Dynamics endorsement. Never put credentials, environment files, logs, or private camera captures in the repository.

## Commit history

- Use short, natural subjects that name the actual change, such as `Add offline camera demo` or `Clarify E-stop setup`. Avoid vague subjects, inflated claims, invented work, and dates in the subject.

## Application safety and visual review

- Keep control labels direct, show units and limits, and distinguish the GUI Stop request from the class E-stop. Avoid unnecessary panels.
- Preserve the robot safety behavior: fresh readiness checks, zero velocity on release, Stop, focus loss, failure, and shutdown, short movement expiry, and the separate class E-stop workflow.
- Verify visual changes in offline demo mode at full and normal window sizes. Do not use a live robot for UI verification.
