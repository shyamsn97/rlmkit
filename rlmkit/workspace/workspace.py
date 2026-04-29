"""Workspace — branch-local working tree, session, context, trace handles."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from rlmkit.workspace.context import Context, FileContext
from rlmkit.workspace.session import FileSession, Session

if TYPE_CHECKING:
    from rlmkit.node import WorkspaceRef
    from rlmkit.runtime.runtime import Runtime


@dataclass
class Workspace:
    """Branch-local handle bundle."""

    root: Path
    session: Session
    context: Context
    branch_id: str = "main"

    def path(self, *parts: str) -> Path:
        """Return a path inside the workspace working tree."""
        return self.root.joinpath(*parts)

    @property
    def trace_dir(self) -> Path:
        return self.root / "trace"

    @property
    def checkpoint_path(self) -> Path:
        return self.root / "checkpoint.json"

    @classmethod
    def create(
        cls,
        dir: str | Path,
        *,
        branch_id: str = "main",
        session: Session | None = None,
        context: Context | None = None,
    ) -> Workspace:
        root = Path(dir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        (root / "trace").mkdir(parents=True, exist_ok=True)
        if session is None:
            session = FileSession(root / "session")
        if context is None:
            context = FileContext(root / "context")
        return cls(root=root, session=session, context=context, branch_id=branch_id)

    @classmethod
    def open(cls, ref: WorkspaceRef) -> Workspace:
        return cls.create(ref.root, branch_id=ref.branch_id)

    def ref(self) -> WorkspaceRef:
        from rlmkit.node import WorkspaceRef

        return WorkspaceRef(root=str(self.root), branch_id=self.branch_id)

    def materialize_runtime(self) -> Runtime:
        from rlmkit.runtime.local import LocalRuntime

        return LocalRuntime(workspace=self)

    def fork(
        self,
        *,
        new_branch_id: str,
        new_dir: str | Path,
    ) -> Workspace:
        new_root = Path(new_dir).resolve()
        if new_root.exists():
            shutil.rmtree(new_root)
        new_root.mkdir(parents=True, exist_ok=True)
        (new_root / "trace").mkdir(parents=True, exist_ok=True)

        reserved = {"session", "context", "trace", "checkpoint.json"}
        for item in self.root.iterdir():
            if item.name in reserved:
                continue
            dst = new_root / item.name
            if item.is_dir():
                shutil.copytree(item, dst)
            else:
                shutil.copy2(item, dst)

        new_session = self.session.fork(new_root / "session")
        new_context = self.context.fork(new_root / "context")
        return Workspace(
            root=new_root,
            session=new_session,
            context=new_context,
            branch_id=new_branch_id,
        )


__all__ = ["Workspace"]
