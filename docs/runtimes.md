# Runtimes

A `Runtime` executes agent Python. The actual execution and generator
suspension live in `rlmflow.runtime.repl.REPL`; a `Runtime` subclass
only decides how to talk to one.

## Protocol

Two abstract methods:

```python
class Runtime(ABC):
    @abstractmethod
    def send(self, msg: dict) -> None: ...

    @abstractmethod
    def recv(self) -> dict: ...
```

Everything else — `execute`, `start_code`, `resume_code`, `inject`,
tool proxy loop, workspace chdir — is in the base class. Messages are
JSON dicts; the wire format is stable.

Commands: `run`, `resume`, `inject`, `inject_proxy`, `inject_object_proxy`.
Responses: `output`, `suspended`, `proxy`, `value`, `error`.

## Shipped runtimes

| Runtime | What it does |
|---|---|
| `LocalRuntime` | In-process. `send`/`recv` dispatch straight to an in-process `REPL`. |
| `DockerRuntime(image, ...)` | Run `python -m rlmflow.runtime.repl` inside a fresh `docker run -i --rm` container; talk to it over stdio. |
| `sandbox.ModalRuntime` | Run the REPL inside a Modal container. |
| `sandbox.E2BRuntime` | Run the REPL inside an E2B Sandbox. |
| `sandbox.DaytonaRuntime` | Run the REPL inside a Daytona Sandbox. |

Need a non-Docker stdio transport (`ssh`, `kubectl exec`, a pre-existing
container)? Subclass `Runtime` and provide `send`/`recv` over your own
pipes — `DockerRuntime` is the reference implementation.

## Docker

```bash
docker build -t rlmflow:local .
```

```python
from rlmflow import RLMFlow
from rlmflow.runtime.docker import DockerRuntime

rt = DockerRuntime(
    image="rlmflow:local",
    mounts={"./data": "/workspace"},
    env={"OPENAI_API_KEY": os.environ["OPENAI_API_KEY"]},
    network="none",
    cpus=1.0,
    memory="512m",
)
agent = RLMFlow(llm_client=llm, runtime=rt, runtime_factory=rt.clone)
```

## Remote sandboxes

Install provider extras as needed:

```bash
pip install rlmflow[modal]
pip install rlmflow[e2b]
pip install rlmflow[daytona]
pip install rlmflow[sandbox]   # all three
```

Each provider runtime lives under `rlmflow.runtime.sandbox` and keeps the
same base `Runtime` protocol. Modal, E2B, and Daytona use
`RemoteFileRuntime`, a public base class that keeps one remote REPL process
alive and exchanges JSON messages through remote files. That keeps REPL
variables and suspended launcher state across turns even when
the provider exposes command execution as one-shot calls.

```python
from rlmflow import RLMFlow
from rlmflow.runtime.sandbox.e2b import E2BRuntime

rt = E2BRuntime(workspace=workspace)
agent = RLMFlow(llm_client=llm, runtime=rt, runtime_factory=rt.clone)
```

See [`examples/sandbox/`](../examples/sandbox/) for real-agent examples
on Modal, E2B, and Daytona.

## Writing your own

Implement `send`/`recv` over whatever transport you want. Minimal
example, given a process you've already started:

```python
class MyRuntime(Runtime):
    def __init__(self, stdin, stdout, workspace="."):
        super().__init__(workspace=workspace)
        self.stdin, self.stdout = stdin, stdout

    def send(self, msg):
        self.stdin.write((json.dumps(msg) + "\n").encode())
        self.stdin.flush()

    def recv(self):
        return json.loads(self.stdout.readline())
```

Override `inject` only if you can bind values directly, as
`LocalRuntime` does. Otherwise the base implementation handles
callables, literals, and method-proxy objects for you.

Override `clone` only if your `__init__` takes arguments beyond
`workspace`.
