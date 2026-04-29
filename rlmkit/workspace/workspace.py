"""Workspace — branch-local files, context, trace, and checkpoint handles."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from rlmkit.workspace.context import ContextStore, FileContext

if TYPE_CHECKING:
    from rlmkit.node import WorkspaceRef
    from rlmkit.runtime.runtime import Runtime


@dataclass
class Workspace:
    """Branch-local handle bundle."""

    root: Path
    context: ContextStore
    branch_id: str = "main"

    @property
    def files(self) -> Path:
        return self.root / "files"

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
        context: ContextStore | None = None,
    ) -> Workspace:
        root = Path(dir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        (root / "files").mkdir(parents=True, exist_ok=True)
        (root / "trace").mkdir(parents=True, exist_ok=True)
        if context is None:
            context = FileContext(root / "context")
        return cls(root=root, context=context, branch_id=branch_id)

    @classmethod
    def open(cls, ref: WorkspaceRef) -> Workspace:
        return cls.create(ref.root, branch_id=ref.branch_id)

    def ref(self) -> WorkspaceRef:
        from rlmkit.node import WorkspaceRef

        return WorkspaceRef(root=str(self.root), branch_id=self.branch_id)

    def materialize_runtime(self) -> Runtime:
        from rlmkit.runtime.local import LocalRuntime

        return LocalRuntime(workspace=self.files)

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

        if self.files.exists():
            shutil.copytree(self.files, new_root / "files")
        else:
            (new_root / "files").mkdir(parents=True, exist_ok=True)

        new_context = self.context.fork(new_root / "context")
        return Workspace(root=new_root, context=new_context, branch_id=new_branch_id)


__all__ = ["Workspace"]
