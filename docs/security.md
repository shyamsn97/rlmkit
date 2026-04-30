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

```python
class ReviewingRLM(RLM):
    def step_exec(self, state):
        code = state.event.code if state.event else ""
        if "rm -rf" in code and not input(f"run? {code}\n> ") == "y":
            return state.update(status=Status.FINISHED, result="rejected")
        return super().step_exec(state)
```

Or override `execute_code` to route code through a classifier, a diff
tool, or a manual approval step before it reaches the runtime.
