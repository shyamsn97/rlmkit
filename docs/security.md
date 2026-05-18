# Security

## Trust model

`LocalRuntime` runs agent Python in your process. Same permissions as
your interpreter: filesystem, network, env vars, subprocesses. **Use
it only for code you'd run yourself.**

For untrusted agents, or agents you haven't audited yet, use an
isolated runtime:

- `DockerRuntime` — a fresh container per session.
- `ModalRuntime` — a remote Modal container.
- Custom `Runtime` — SSH, `kubectl exec`, Firecracker, gVisor, anything.

## Docker isolation knobs

```python
DockerRuntime(
    image="rlmflow:local",
    network="none",           # no outbound traffic
    cpus=1.0,                 # CPU quota
    memory="512m",            # OOM cap
    user="1000:1000",         # non-root
    extra_args=[
        "--read-only",        # read-only rootfs
        "--security-opt", "no-new-privileges",
    ],
    mounts={"./workspace": "/workspace"},
)
```

Mount only what the agent needs. A hostile agent inside the container
can still fill its writable volumes, burn CPU up to the quota, and
call any tool you injected.

## Engine-level caps

Independent of the runtime:

- `max_depth` — recursion limit.
- `max_iterations` — LLM calls per agent.
- `max_budget` — total tokens across the subtree.
- `max_output_length` — truncate oversized stdout.
- `max_concurrency` — opt into threaded parallel children when set.

## Proxied tools

`runtime.inject(name, fn)` and `runtime.register_tool(fn)` route calls
from inside the REPL back to the host. The host runs the callable
with its CWD set to `self.workspace`.

Tools you register are part of the trust boundary. The container can
be sealed off, but any injected tool runs on the host with host
privileges. Keep that surface small and validate arguments.

## Overrides for approval gates

Override `_run_code(view, code)` to gate, classify, or rewrite code
before it touches the runtime. The hook returns
`(suspended: bool, raw: object)` — the same tuple the runtime
ordinarily yields — so you can short-circuit execution with a
rejection string and the engine will record it as the action's
observation:

```python
from rlmflow import RLMFlow

class ReviewingRLM(RLMFlow):
    def _run_code(self, view, code: str):
        if "rm -rf" in code and input(f"run? {code}\n> ") != "y":
            return False, "rejected by reviewer"
        return super()._run_code(view, code)
```

Wrap the runtime itself if you want approval at the transport layer.
Subclass `Runtime` and override `execute`, `start_code`, or
`resume_code` to call a classifier, diff tool, or manual approval
step before delegating to the underlying runtime.
