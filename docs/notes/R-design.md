# Wave 2 Reward/Eval Interface Design

This document fixes the contracts for parallel R1–R4 implementation. No workstream should edit another workstream’s owned files.

## Shared invariants

- Reward call semantics remain `reward(samples, prompts) -> array-like[B]`.
- `samples` are decoded NumPy arrays for image/video rewards or `list[str]` for text rewards.
- `prompts` has exactly one entry per sample; entries may be `null`.
- All public APIs preserve input order.
- Invalid requests return structured errors; no partial scores are returned.
- Existing reward specs use the registry format: `"brightness"` or `{"type": "brightness", ...}`.

## R1 — Batched reward service

### Ownership

- `packages/reward/src/diffusion_omni/reward/service.py`
- `packages/reward/src/diffusion_omni/reward/client.py`
- `packages/reward/src/diffusion_omni/reward/service_cli.py`
- `packages/reward/tests/test_service.py`
- One registration in the existing reward CLI integration, owned by R4.

Do not edit `registry.py`, `scorecard.py`, `eval.py`, or `pyproject.toml`.

### Server

Expose:

- `POST /score`
- `GET /health`

`serve_reward(spec, host="127.0.0.1", port=0, max_batch_size=32, max_wait_ms=10) -> ThreadingHTTPServer`

The transport must use Python standard-library HTTP primitives. NumPy remains an existing package dependency for `.npy` decoding and reward execution; no web framework or new hard dependency is allowed.

The server loads one configured reward spec at startup from `--spec`. A request may override it only when explicitly enabled by the server; default behavior is fixed-spec mode.

Request schema:

```json
{
  "request_id": "optional-string",
  "samples": {
    "encoding": "npy_base64",
    "data": "<base64 of .npy bytes>",
    "shape": [2, 3, 8, 8],
    "dtype": "float32"
  },
  "prompts": ["make red", "make red"],
  "reward": {"type": "channel_dominance", "channel": 0}
}
```

Text alternative:

```json
{
  "samples": {
    "encoding": "text",
    "data": ["a generated answer", "another answer"]
  },
  "prompts": ["question one", "question two"],
  "reward": "char_fraction"
}
```

Rules:

- `npy_base64` must contain a valid `.npy` payload; reject object arrays and arrays whose first dimension does not equal `len(prompts)`.
- `text.data` must be a JSON list of strings.
- `reward` is optional in fixed-spec mode and required in override mode.
- Maximum request size and maximum sample count are configurable and rejected with HTTP 413.
- Unsupported encoding, malformed JSON, shape mismatch, unknown reward, or reward exceptions return HTTP 400 with:

```json
{"error": {"type": "invalid_request", "message": "..."}}
```

Success response:

```json
{
  "schema_version": "diffusion_omni.reward/v1",
  "request_id": "optional-string",
  "scores": [0.25, 0.5],
  "batch_size": 2
}
```

Batching policy:

- Concurrent requests enter a FIFO queue.
- Requests are coalesced only when their canonical reward specs match.
- Flush when either `max_batch_size` samples are available or the oldest queued request reaches `max_wait_ms`.
- One reward invocation processes the combined samples and prompts.
- Results are split back to requests in original order.
- A failed invocation fails every request in that batch.
- The server must not mutate caller-owned arrays or prompt lists.

### Client

`RewardClient` uses `urllib`, `json`, `base64`, and `http.client`; it must not require `requests`.

```python
class RewardClient:
    def __init__(self, base_url: str, *, timeout: float = 30.0): ...

    def score(
        self,
        samples,
        prompts: Sequence[str | None],
        reward: str | dict,
        *,
        request_id: str | None = None,
    ) -> list[float]: ...

    def score_npy(
        self,
        npy_bytes: bytes,
        prompts: Sequence[str | None],
        reward: str | dict,
        *,
        shape: Sequence[int] | None = None,
        dtype: str | None = None,
        request_id: str | None = None,
    ) -> list[float]: ...

    def score_text(
        self,
        samples: Sequence[str],
        prompts: Sequence[str | None],
        reward: str | dict,
        *,
        request_id: str | None = None,
    ) -> list[float]: ...

    def health(self) -> dict: ...
```

`score()` accepts either text sequences or serialized `.npy` bytes and dispatches to the explicit methods. HTTP, timeout, and server errors raise `RewardClientError`, carrying status code and server message.

## R2 — miles-diffusion adapter

### Ownership

- `packages/reward/src/diffusion_omni/reward/miles.py`
- `packages/reward/tests/test_miles.py`
- `docs/notes/R2.md`

The upstream interface is confirmed from the current repository documentation:

- `--custom-rm-path` loads a dotted function path.
- The function signature is `async def custom_rm(args, samples: list[Sample], **kwargs) -> list[float]`.
- Samples expose `sample.prompt` and `sample.generated_output`.
- Remote scoring is explicitly demonstrated through `args.rm_url`.

See [`customization.md`](https://github.com/radixark/miles_diffusion/blob/main/docs/user-guide/customization.md#L272-L305) and [`rewards.md`](https://github.com/radixark/miles_diffusion/blob/main/docs/user-guide/rewards.md#L201-L209).

Required adapter:

```python
async def diffusion_omni_rm(
    args,
    samples: list,
    **kwargs,
) -> list[float]:
    ...
```

Behavior:

- Read the reward spec from `args.diffusion_omni_reward_spec` or `args.custom_rm_args`.
- Read the service URL from `args.rm_url`.
- Extract prompts from `sample.prompt`.
- Convert `sample.generated_output` to the service’s text or `.npy` representation.
- Call `RewardClient` through `asyncio.to_thread` unless an async transport is added later.
- Return exactly one finite `float` per input sample.
- Preserve sample order and raise on count mismatch.
- Do not import Miles at module import time; the stub test must run without Miles installed.

`docs/notes/R2.md` records the flag, signature, source links, and the generated-output conversion assumption.

## R3 — Reliability module

### Ownership

- `packages/reward/src/diffusion_omni/reward/reliability.py`
- `packages/reward/scripts/reward_reliability.py`
- `packages/reward/tests/test_reliability.py`
- `docs/notes/R3.md`

Public API:

```python
def rankdata(values: Sequence[float]) -> np.ndarray: ...

def spearman(
    reward_scores: Sequence[float],
    oracle_scores: Sequence[float],
) -> float: ...

def bootstrap_ci(
    values: Sequence[float],
    *,
    statistic: Callable = np.mean,
    n_resamples: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
) -> tuple[float, float]: ...

def agreement(
    reward_scores: Sequence[float],
    oracle_scores: Sequence[float],
    *,
    seed: int = 0,
) -> dict:
    # n, spearman, bootstrap_ci
    ...

def hacking_probe(
    reward_a: Sequence[float],
    reward_b: Sequence[float],
    oracle: Sequence[float],
    *,
    steps: Sequence[int] | None = None,
) -> dict:
    # correlations, regret/divergence, best-by-A vs best-by-oracle
    ...
```

The script accepts JSON/JSONL inputs, runs agreement and reward-hacking comparisons, and writes versioned JSON. Synthetic tests must cover perfect agreement, ties, inverse ranking, and shrinking bootstrap intervals as `n` increases.

## R4 — EVALS registry and CLI

### Ownership

- `packages/reward/src/diffusion_omni/reward/eval.py`
- `packages/reward/tests/test_eval.py`
- `packages/reward/src/diffusion_omni/reward/cli.py`
- `packages/core/src/diffusion_omni/core/registry.py`
- One-line import/registration for R1’s `service_cli.py`.

Add:

```python
EVALS = Registry("eval")
```

and include it in `ALL`.

Suite contract:

```python
class EvalSuite(Protocol):
    def __call__(self, *, ckpt: str, seed: int, n: int, device: str, **kwargs) -> dict:
        ...
```

`diffusion-omni eval` arguments:

```text
--suite NAME
--ckpt NAME=PATH       repeatable
--ref PATH              suite-specific optional reference
--seeds INT [INT ...]
-n INT
--device DEVICE
--out PATH
```

`scorecard` is registered as the initial suite and delegates to existing `run_scorecard`.

Output schema:

```json
{
  "schema_version": "diffusion_omni.eval/v1",
  "suite": "scorecard",
  "seeds": [0, 1, 2],
  "config": {"n": 256, "device": "cpu"},
  "runs": [
    {"seed": 0, "rows": [{"model": "student", "nfe": 2, "reward": 0.4}]}
  ],
  "summary": [
    {
      "group": {"model": "student", "nfe": 2},
      "metrics": {
        "reward": {
          "n": 3,
          "mean": 0.4,
          "std": 0.02,
          "ci95": [0.38, 0.42]
        }
      }
    }
  ]
}
```

Unknown suites, malformed checkpoint assignments, and empty seed lists fail with parser errors. Existing scorecard JSON/Markdown behavior remains unchanged.

## Ownership summary

| Workstream | Exclusive files | Shared edits |
|---|---|---|
| R1 | `service.py`, `client.py`, `service_cli.py`, service tests | one CLI registration |
| R2 | `miles.py`, Miles tests, `docs/notes/R2.md` | none |
| R3 | `reliability.py`, script, tests, notes | none |
| R4 | `eval.py`, eval tests, `cli.py`, `registry.py` | one R1 CLI registration |

No workstream edits `pyproject.toml`; the existing `diffusion_omni.commands` entry point remains the integration point.

