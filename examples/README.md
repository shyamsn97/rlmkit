# Examples

Every compute example (`summarizer.py`, `needle_haystack.py`, `showcase.py`,
`coding-agent/agent.py`) takes the same flags:

| Flag | Default | Meaning |
|---|---|---|
| `--model MODEL` | `claude-opus-4-6` | Main LLM. Prefix decides client (`claude*` → Anthropic, else OpenAI). |
| `--fast-model MODEL` | unset | Optional cheap secondary model registered as `fast` for delegates. |
| `--docker-image IMAGE` | unset | If set, run agent code inside this Docker image. Must have `rlmflow` installed. Leaving this unset uses `LocalRuntime`. |
| `--max-depth N` | `3` | Max delegation depth. |
| `--max-iterations N` | `15` | Max LLM calls per agent. |
| `--no-viz` | off | Disable the live terminal visualization. |

## Running under Docker

The repo ships a `Dockerfile` at its root that builds an image with `rlmflow`
preinstalled. Build it once:

```bash
docker build -t rlmflow:local .
```

Then just pass `--docker-image rlmflow:local` to any example — presence of
the flag is what enables the Docker runtime:

```bash
python examples/summarizer.py      --docker-image rlmflow:local
python examples/needle_haystack.py --docker-image rlmflow:local
python examples/showcase.py        --docker-image rlmflow:local
python examples/coding-agent/agent.py --workspace ./proj --docker-image rlmflow:local
```

The host workspace is bind-mounted at `/workspace` inside the container, so
the standard `FILE_TOOLS` work identically in both modes.

For fully locked-down runs, `DockerRuntime` takes the usual Docker knobs
directly when built by hand:

```python
from rlmflow.runtime.docker import DockerRuntime

runtime = DockerRuntime(
    image="rlmflow:local",
    mounts={"./data": "/workspace"},
    env={"OPENAI_API_KEY": os.environ["OPENAI_API_KEY"]},
    network="none",       # air-gap the container
    cpus=1.0,
    memory="512m",
)
```

## Other examples

- `drop_in_llm.py` — shows that `RLMFlow` satisfies `LLMClient`, so you can nest
  agents or swap an agent in anywhere a plain LLM is accepted. No CLI flags.
- `view_demo.py` — builds a fake trace and opens the state viewer. No LLM or
  runtime needed.
