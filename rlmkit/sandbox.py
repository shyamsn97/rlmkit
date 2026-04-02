"""In-container driver for remote sandboxes.

Run this module as the entrypoint of a sandbox container::

    python -m rlmkit.sandbox

It reads JSON commands from stdin and writes JSON responses to stdout.
The protocol supports three commands:

- ``{"cmd": "inject", "name": "...", "value": "..."}``
  Evaluate *value* in the namespace and bind it to *name*.

- ``{"cmd": "run", "code": "..."}``
  Wrap *code* in a generator and drive it. Returns either
  ``{"suspended": true, "agent_ids": [...]}`` or
  ``{"suspended": false, "output": "..."}``.

- ``{"cmd": "resume", "value": <any>}``
  Send *value* into the suspended generator and continue driving.
  Same return format as ``run``.
"""

from __future__ import annotations

import io
import json
import sys
import textwrap

gen = None
buf = None
ns: dict = {"__builtins__": __builtins__}


def drive(send_value=None):
    global gen, buf
    try:
        req = gen.send(send_value)
        return {"suspended": True, "agent_ids": req.agent_ids}
    except StopIteration:
        return {"suspended": False, "output": buf.getvalue().strip()}
    except Exception as exc:
        return {"suspended": False, "output": buf.getvalue().strip() + f"\n{type(exc).__name__}: {exc}"}


def main():
    global gen, buf, ns
    for line in sys.stdin:
        msg = json.loads(line)
        cmd = msg["cmd"]
        if cmd == "run":
            buf = io.StringIO()
            ns["print"] = lambda *a, **kw: __builtins__["print"](*a, **{**kw, "file": buf})
            indented = textwrap.indent(msg["code"], "    ")
            exec(f"def __rlm_gen__():\n{indented}\n", ns)
            gen = ns["__rlm_gen__"]()
            resp = drive()
        elif cmd == "resume":
            resp = drive(send_value=msg.get("value"))
        elif cmd == "inject":
            ns[msg["name"]] = eval(msg["value"], ns)
            resp = {"ok": True}
        else:
            resp = {"error": f"unknown command: {cmd}"}
        print(json.dumps(resp), flush=True)


if __name__ == "__main__":
    main()
