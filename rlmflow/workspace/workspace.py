"""Workspace — branch-local working tree, session, and context handles."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from rlmflow.graph import Graph, WorkspaceRef, retrace_steps
from rlmflow.workspace.context import Context, FileContext
from rlmflow.workspace.session import FileSession, Session
from rlmflow.workspace.store import FileStore


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
        store = FileStore(root)
        if session is None:
            session = FileSession(store)
        if context is None:
            context = FileContext(store)
        return cls(root=root, session=session, context=context, branch_id=branch_id)

    @classmethod
    def open(cls, ref: WorkspaceRef) -> Workspace:
        return cls.create(ref.root, branch_id=ref.branch_id)

    @classmethod
    def open_path(
        cls,
        dir: str | Path,
        *,
        branch_id: str = "main",
    ) -> Workspace:
        """Open an existing workspace directory by path.

        This is intentionally the same materialization path as ``create``:
        workspace storage is append-only, so constructing the handle should not
        mutate run state beyond ensuring the root directory exists.
        """
        return cls.create(dir, branch_id=branch_id)

    @staticmethod
    def check_path(path: str | Path) -> bool:
        """Return True if ``path`` looks like a persisted RLMFlow workspace."""
        root = Path(path)
        return (
            root.is_dir()
            and (root / "graph.json").is_file()
            and (root / "session").is_dir()
        )

    def ref(self) -> WorkspaceRef:
        return WorkspaceRef(root=str(self.root), branch_id=self.branch_id)

    def load_graph(self) -> Graph:
        """Load the current graph snapshot from this workspace's session."""
        return self.session.load_graph()

    def load_steps(self) -> list[Graph]:
        """Load the run as a list of snapshots, one per state-append.

        Retraces the persisted graph as it would have looked after each
        successive state was written, ordered the way an ``RLMFlow``
        with unbounded ``max_concurrency`` would have produced them
        (children spawned by the same supervising step are
        round-robined, not drained one-at-a-time).
        """
        return retrace_steps(self.load_graph())

    def open_viewer(self, **kwargs):
        """Open the interactive viewer for this workspace."""
        from rlmflow.utils.viewer import open_viewer

        return open_viewer(self, **kwargs)

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

        reserved = {
            "session",
            "context",
            "graph.json",
            "trace",
            "checkpoint.json",
        }
        for item in self.root.iterdir():
            if item.name in reserved:
                continue
            dst = new_root / item.name
            if item.is_dir():
                shutil.copytree(item, dst)
            else:
                shutil.copy2(item, dst)

        new_session = self.session.fork(new_root)
        new_context = self.context.fork(new_root)
        return Workspace(
            root=new_root,
            session=new_session,
            context=new_context,
            branch_id=new_branch_id,
        )


__all__ = ["Workspace"]
