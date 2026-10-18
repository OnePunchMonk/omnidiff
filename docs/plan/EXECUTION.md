# How work is executed

Applies from 2026-10-18 to every workstream in [plan.md](../../plan.md). The rule is: use the cheapest agent that can do the task, and review with Codex instead of Opus.

## Models available

Claude subagents are started with the Agent tool, which accepts the aliases `haiku`, `sonnet`, `opus` and `fable` and an `effort` of `low` to `max`. Codex runs through the CLI at `~/.local/bin/codex` (v0.153.4). On this ChatGPT account only `gpt-5.6-luna` works; `default`, `5.2-codex` and `gpt-6-luna` are rejected. The user config (`~/.codex/config.toml`) already sets `model = "gpt-5.6-luna"`, `model_reasoning_effort = "low"` and `service_tier = "priority"`.

API list prices (per million tokens, input / output, 2026-10) show the relative cost of the Claude aliases. Inside Claude Code the spend comes out of plan usage, but the ratios still hold.

| alias | model | input / output | relative |
|---|---|---|---|
| `haiku` | Claude Haiku 5.5 | $0.10 / $0.50 | 1x |
| `sonnet` | Claude Sonnet 5.5 | $2 / $10 | 20x |
| `opus` | Claude Opus 5.5 | $4 / $20 | 40x |
| `fable` | Claude Fable 5.1 | $10 / $50 | 100x |
| Codex | `gpt-5.6-luna` | ChatGPT plan quota, no per-token bill | separate budget |

Codex draws on a separate quota, so it costs no Claude usage. Its limits are its rate limit and the fact that it works only through the CLI: it cannot be steered mid-task the way a SendMessage to a subagent can.

## Defaults per task

| task | default | why, and the trade-off |
|---|---|---|
| mechanical edits: config files, script flags, doc rewrites against a checklist, README tables, merges, running the test gate, collecting results | **Haiku, low** | cheapest by 20x. It misses subtle invariants, so give it an exact checklist and keep its scope to files that tests or a script can check |
| implementation inside an existing pattern (a new registry entry, a fix with a named test, R1f–R4f, L1, L4, L5) | **Sonnet, low** | follows the repo style and the registries well at low effort. Raise to `medium` only after one failed review cycle |
| bulk new code with a clear spec (a new package skeleton plus tests, L2, L3, L6, the rename) | **Codex `gpt-5.6-luna`, `workspace-write`** | no Claude cost and good at long, well-specified edits. It has no conversation context, so the prompt must carry the spec, file ownership and test command |
| design notes and scope docs (S0) | **Codex, low** | the R-design.md interface spec was written this way |
| review of every branch | **Codex, low, `read-only`** | see below |
| escalated review | **Codex, medium**, then **Sonnet, medium** if still unresolved | see the escalation rule |
| orchestration, results write-ups, conflict resolution | the orchestrating session | needs the whole context |
| Opus and Fable | **not used** for workstreams | 40x and 100x Haiku. Only when the user asks |

Effort: `low` unless a row says otherwise. An agent that fails the same review twice gets one effort level higher, not a bigger model.

## Codex invocations

Implementation (in the workstream's worktree):

```bash
~/.local/bin/codex exec -m gpt-5.6-luna -c model_reasoning_effort="low" \
  -s workspace-write -C <worktree> -o docs/notes/<id>-codex.md - < prompt-<id>.md
```

Review (read-only, against `main`):

```bash
~/.local/bin/codex exec -m gpt-5.6-luna -c model_reasoning_effort="low" \
  -s read-only -C <worktree> -o docs/notes/<id>-review.md - < review-prompt-<id>.md
```

`codex exec review --base main` also works for a quick pass, but the custom prompt below is the default, because it carries the plan and the repo principles.

### Review prompt template

```text
You are reviewing branch ws/<id> of the diffusion-omni repo (pure PyTorch + NumPy diffusion monorepo).
Read-only: do not edit files, do not run git commands that change state.

Scope of this workstream (from plan.md):
<paste the plan.md row: id, what it owns, done-when>

Interface or scope doc: <path, e.g. docs/notes/R-design.md#r1 or docs/scope/<layer>.md>

Do this:
1. Run `git diff main...HEAD --stat` and read the full diff.
2. Run `uv run pytest -q` and report the result line.
3. Check, in this order:
   a. Correctness: does each new method or component have a test of the property that makes it
      correct (repo principle 1)? Would the test fail if the code were wrong?
   b. The done-when criteria above, one by one: met / not met, with file:line.
   c. Security for anything that parses input, loads files or imports modules by name.
   d. Files touched outside the workstream's ownership.
   e. Core (packages/core) imports anything beyond torch, numpy, pyyaml and the standard library.
   f. Style: registries and context injection used; docstrings say why; no dead code.
4. Do not report style nits unless nothing else is wrong.

Output, as markdown:
- Verdict: APPROVE or CHANGES REQUESTED.
- Findings, ranked: [blocker] / [major] / [minor], each with file:line, what is wrong, and the smallest fix.
- Test result line.
Keep it under 60 lines.
```

### Escalation rule

A review moves up one level (Codex low → Codex medium → Sonnet medium, read-only) when any of these holds:

- the change parses network or file input, deserialises data, or imports code by name: R1 (HTTP service), R2 (miles adapter), `load_bundle`, the workflow graph loader (L4);
- it changes `packages/core` (L6, any core edit made for another layer);
- two review cycles on the same branch still end in CHANGES REQUESTED;
- the implementer disputes a [blocker] finding.

An escalated review gets the earlier review files as input. If Sonnet medium and the implementer still disagree, the orchestrator decides and records the reason in `docs/notes/<id>.md`.

## Worktrees and branches

- One workstream = one branch `ws/<id>` = one worktree under `/Users/aggar/diffusion-omni/.claude/worktrees/` (Agent tool `isolation: "worktree"`, or `git worktree add .claude/worktrees/codex-<id> -b ws/<id> main` for Codex).
- Branch from the current `main`. Rebase onto `main` before review and again before merge.
- An agent edits only the files its plan row owns. Shared files (`methods/__init__.py`, `pyproject.toml` entry points, `core/registry.py`) get one-line additions only.
- Notes and findings go in `docs/notes/<id>.md`. Agents do not edit `README.md`, `docs/results/` or `docs/scope/` unless that is the workstream (DOC, S0).
- Teachers and long runs stay in the main checkout under `/Users/aggar/diffusion-omni/runs/`. An agent keeps its own experiments under 15 CPU-minutes. CPU only; no GPU run without a written cost estimate and the user's sign-off.
- Remove a worktree after its branch is merged (`git worktree remove`), so stale branches do not pile up. Today there are 14 worktrees; the merged ones (D1–D7, C1, ARCH) can go once the user agrees.

## Test gate

- `uv run pytest -q` passes before every commit on a branch and on the merged tree before any merge. Baseline on `main` at `9fe1202`: 165 passed.
- A new method or component lands with a test of its defining property. A bug fix lands with a test that fails without the fix.
- A flaky test is a blocker. Timing assertions are replaced by deterministic ones (fake clocks, explicit flush calls).
- Configs: new configs are covered by the existing end-to-end config test via `tiny_config`.

## Commit and push policy

- Commits: `git -c user.name=OnePunchMonk -c user.email=aggarwal.avaya@yahoo.com commit ...`. No co-author trailers and no session links.
- Agents commit on their own `ws/<id>` branch only. **Agents never push.**
- The user's standing rule: nothing is pushed, nothing is committed to a PR branch, and no PR is opened without explicit approval. So the default integration step is **prepare the merge, then ask**: the integrator rebases, merges into a local `main`, runs the test gate, and reports the merge commit and test line to the user, who approves the push.
- Recorded exception, **to be re-confirmed by the user**: earlier in the 2026-10-17/18 session the user granted standing approval for the Haiku integrator to merge reviewed workstream branches into `main` of `OnePunchMonk/diffusion-omni` and push. Scope: this private repo only, `main` only, only branches with an APPROVE review and a green test gate. It does not cover force pushes, tags, releases, new PRs, issue comments, the rename (RN) or any other repo. Until the user re-confirms, the integrator uses "prepare merge, then ask".
- Issue comments on #2 and #3 are drafted in `docs/plan/` and posted only by the user or with their explicit approval.

## Integrator checklist (Haiku, low)

1. `git fetch`; check out `main`; confirm it equals `origin/main`.
2. For each branch in the plan's merge order: confirm `docs/notes/<id>-review.md` says APPROVE; `git merge --no-ff ws/<id>`; `uv run pytest -q`.
3. On a conflict or a test failure: `git merge --abort`, stop, and report the branch, the files and the failing tests. Do not resolve conflicts.
4. Report the merge commits and the final test line. Push only under the re-confirmed standing approval; otherwise ask.
