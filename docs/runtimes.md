# Runtimes

A `Runtime` executes agent Python. The actual execution and generator
suspension live in `rlmkit.runtime.repl.REPL`; a `Runtime` subclass
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
| `SubprocessRuntime(argv)` | Spawn any `argv` that runs `python -m rlmkit.runtime.repl` and talk to it over stdio. |
| `DockerRuntime(image, ...)` | `SubprocessRuntime` with an ergonomic `docker run` argv builder. |
| `ModalRuntime` | Run the REPL inside a Modal container. |

## Subprocess examples

```python
SubprocessRuntime(["python", "-m", "rlmkit.runtime.repl"])

SubprocessRuntime(["docker", "exec", "-i", "ctr",
                   "python", "-m", "rlmkit.runtime.repl"])

SubprocessRuntime(["ssh", "box", "python", "-m", "rlmkit.runtime.repl"])

SubprocessRuntime(["kubectl", "exec", "-i", "pod", "--",
                   "python", "-m", "rlmkit.runtime.repl"])
```

## Docker

```bash
docker build -t rlmkit:local .
```

```python
from rlmkit.runtime.docker import DockerRuntime

rt = DockerRuntime(
    image="rlmkit:local",
    mounts={"./data": "/workspace"},
    env={"OPENAI_API_KEY": os.environ["OPENAI_API_KEY"]},
    network="none",
    cpus=1.0,
    memory="512m",
)
agent = RLM(llm_client=llm, runtime=rt, runtime_factory=rt.factory)
```

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
