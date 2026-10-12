"""Polar runtime for the one fixed HTTP harness; never a candidate-code executor.

This is a transport adapter, not a sandbox. A SHA256 allowlist freezes the
trusted harness command before any model response exists. Candidate text is
serialized to the remote Docker verifier and is never imported or executed here.
"""

import asyncio
import hashlib
import os
import signal
from pathlib import Path

from polar.runtime.base import BaseRuntime
from polar.runtime.models import ExecResult


class TrustedHTTPRuntime(BaseRuntime):
    @property
    def runtime_id(self):
        return "trusted-http:" + self.session_id

    def resolve_host_path(self, runtime_path):
        root = self.session_dir.resolve()
        value = Path(runtime_path)
        if value.is_relative_to("/polar/session"):
            value = root / value.relative_to("/polar/session")
        else:
            value = root / value
        value = value.resolve()
        if not value.is_relative_to(root):
            raise ValueError("trusted harness path escapes its session")
        return value

    async def start(self):
        if self.spec.prepare:
            raise ValueError("trusted HTTP runtime accepts no shell prepare actions")
        digest = self.spec.kwargs.get("trusted_command_sha256", "")
        if len(digest) != 64:
            raise ValueError("fixed HTTP harness SHA256 is required")
        for part in ("workspace", "artifacts", "logs/agent", "logs/eval"):
            (self.session_dir / part).mkdir(parents=True, exist_ok=True)

    async def stop(self):
        if self._active_process and self._active_process.returncode is None:
            os.killpg(self._active_process.pid, signal.SIGKILL)
            await self._active_process.wait()
        self._destroyed = True

    async def exec(self, command, *, cwd=None, env=None, timeout_sec=None):
        # Polar's existing verifier_result_file evaluator requests precisely
        # this artifact read after the harness returns. Read it directly;
        # accepting it does not grant another executable shell command.
        if command == "cat '/polar/session/workspace/verifier_result.json'":
            path = self.resolve_host_path("/polar/session/workspace/verifier_result.json")
            try:
                return ExecResult(return_code=0, stdout=path.read_text())
            except OSError as error:
                return ExecResult(return_code=1, stderr=str(error))
        digest = hashlib.sha256(command.encode()).hexdigest()
        if digest != self.spec.kwargs["trusted_command_sha256"]:
            raise ValueError("refused command outside frozen HTTP harness")
        # The fixed command contains literal container session paths. Translate
        # only the command and path-valued variables, never JSON task content.
        command = command.replace("/polar/session", str(self.session_dir.resolve()))
        process_env = {"PATH": os.environ["PATH"], "LANG": "C.UTF-8", "PYTHONUNBUFFERED": "1"}
        for key, value in {**self.spec.env, **(env or {})}.items():
            if key.startswith(("POLAR_", "OPENAI_")) or key in (
                "ARTIFACTS_DIR", "LOGS_DIR", "WORKSPACE_DIR", "AGENT_LOG_DIR"
            ):
                if key.endswith("_DIR") and value.startswith("/polar/session"):
                    value = value.replace("/polar/session", str(self.session_dir.resolve()), 1)
                process_env[key] = value
        process = await asyncio.create_subprocess_exec(
            "bash", "-c", command,
            cwd=self.resolve_host_path(cwd or self.spec.workdir or "/polar/session/workspace"),
            env=process_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        self._active_process = process
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_sec)
            return ExecResult(return_code=process.returncode,
                              stdout=stdout.decode(errors="replace"),
                              stderr=stderr.decode(errors="replace"))
        except TimeoutError:
            os.killpg(process.pid, signal.SIGKILL)
            await process.wait()
            return ExecResult(return_code=-1, stderr="trusted HTTP harness timeout")
        finally:
            self._active_process = None

    async def upload_file(self, local_path, remote_path):
        raise ValueError("fixed HTTP harness accepts no uploaded executable files")

    async def upload_dir(self, local_path, remote_path):
        raise ValueError("fixed HTTP harness accepts no uploaded executable directories")

    async def download_file(self, remote_path, local_path):
        if not self._copy_from_bind_mount(remote_path, Path(local_path)):
            raise FileNotFoundError(remote_path)

    async def download_dir(self, remote_path, local_path):
        if not self._copy_from_bind_mount(remote_path, Path(local_path)):
            raise FileNotFoundError(remote_path)
