"""Opt-in whole-response-group refit barrier for the copied Polar collector.

Apply after ``patch_transport.apply``. The original behavior remains selected
unless the actor inherits ``NEMO_POLAR_GROUP_REFIT_BARRIER=1``. The optional
``NEMO_POLAR_GROUP_REFIT_DRAIN_TIMEOUT`` defaults to its existing request timeout.
This patch changes admission/refit synchronization, not replay targets or age.
"""

import ast
import hashlib
from pathlib import Path


def _replace_once(source, before, after):
    if source.count(before) != 1:
        raise RuntimeError(f"group-refit patch anchor count {source.count(before)}: {before[:80]}")
    return source.replace(before, after)


INITIALIZE = '''        self._group_refit_enabled = env_flag("NEMO_POLAR_GROUP_REFIT_BARRIER")
        # A Condition's reentrant lock also permits an already-finished Future
        # to invoke its done callback inline during add_done_callback().
        self._group_refit_condition = threading.Condition()
        self._group_refit_active = {}
        self._group_refit_next_id = 0
        self._group_refit_error = None
        self._group_refit_draining = False
        self._group_refit_drained = []
        if self._group_refit_enabled:
            self._group_refit_timeout = float(os.environ.get(
                "NEMO_POLAR_GROUP_REFIT_DRAIN_TIMEOUT", str(self._request_timeout)))
            if not 0 < self._group_refit_timeout < float("inf"):
                raise ValueError("whole-group refit drain timeout must be positive and finite")
            emit_rlvp_receipt("group_refit_barrier_config", enabled=True,
                drain_timeout_seconds=self._group_refit_timeout,
                concurrent_groups=self._concurrent_groups,
                response_workers_per_group=self._group_collection_workers)
'''


METHODS = '''    def _wait_for_group_refit(self) -> None:
        if not self._group_refit_enabled:
            self._refit_pause.wait()
            return
        with self._group_refit_condition:
            while self.running and not self._refit_pause.is_set() and not self._group_refit_error:
                self._group_refit_condition.wait()
            if self._group_refit_error:
                raise RuntimeError(self._group_refit_error)
            if not self.running:
                raise RuntimeError("collector stopped before whole-group admission")

    def _admit_group_for_refit(self, target: int) -> tuple[int, int]:
        # Caller holds the same condition used to close admission before refit.
        self._wait_for_group_refit()
        admission_id = self._group_refit_next_id
        self._group_refit_next_id += 1
        generation_version = self.current_weight_version
        record = {"admission_id": admission_id,
                  "generation_weight_version": generation_version,
                  "target_weight_version": target, "group_id": None}
        self._group_refit_active[admission_id] = record
        emit_rlvp_receipt("group_refit_admitted", **record,
                         active_groups=len(self._group_refit_active))
        return admission_id, generation_version

    def _fail_group_refit(self, error: Exception, phase: str, admission_id=None) -> None:
        with self._group_refit_condition:
            message = f"{type(error).__name__}: {error}"
            self._group_refit_error = self._group_refit_error or message
            record = self._group_refit_active.get(admission_id)
            if record is not None:
                record["error"] = message
            self.running = False
            self._refit_pause.clear()
            self._group_refit_condition.notify_all()
            emit_rlvp_receipt("group_refit_failure", phase=phase, error=message,
                sequence=self._refit_sequence, admission_id=admission_id,
                active_groups=[dict(v) for v in self._group_refit_active.values()])

    def _finish_group_for_refit(self, admission_id: int, future=None) -> None:
        with self._group_refit_condition:
            if future is not None:
                try:
                    error = future.exception()
                except Exception as exc:
                    error = exc
                if error is not None:
                    self._fail_group_refit(error, "future", admission_id)
            record = self._group_refit_active.pop(admission_id)
            if record["generation_weight_version"] != self.current_weight_version:
                error = RuntimeError("weight version changed before the whole group completed")
                record["error"] = str(error)
                self._fail_group_refit(error, "completion", admission_id)
            record["success"] = "error" not in record
            if self._group_refit_draining:
                self._group_refit_drained.append(dict(record))
            emit_rlvp_receipt("group_refit_completed", **record,
                             active_groups=len(self._group_refit_active))
            self._group_refit_condition.notify_all()

    def _collect_group_at_version(self, target: int) -> None:
        with self._group_refit_condition:
            admission_id, generation_version = self._admit_group_for_refit(target)
        try:
            self._collect_group(generation_weight_version=generation_version,
                target_weight_version=target, refit_admission_id=admission_id)
        except Exception as exc:
            self._fail_group_refit(exc, "collection", admission_id)
            raise
        finally:
            self._finish_group_for_refit(admission_id)

    def _submit_group_at_version(self, pool, run_group, target: int) -> None:
        with self._group_refit_condition:
            admission_id, generation_version = self._admit_group_for_refit(target)
            try:
                future = pool.submit(run_group, generation_version, target, admission_id)
            except Exception as exc:
                self._fail_group_refit(exc, "submission", admission_id)
                self._finish_group_for_refit(admission_id)
                raise
            # Registered before submit; retired only once the entire Future is
            # finished, including queued sessions, verification and replay add.
            future.add_done_callback(
                lambda completed: self._finish_group_for_refit(admission_id, completed))

    def _prepare_whole_group_refit(self) -> None:
        started = time.monotonic()
        deadline = started + self._group_refit_timeout
        try:
            with self._group_refit_condition:
                self._refit_sequence += 1
                self._refit_pause.clear()
                self._group_refit_draining = True
                self._group_refit_drained = []
                admitted = [dict(v) for v in self._group_refit_active.values()]
                emit_rlvp_receipt("refit_start", sequence=self._refit_sequence,
                    weight_version=self.current_weight_version, whole_group_barrier=True)
                emit_rlvp_receipt("group_refit_drain_start", sequence=self._refit_sequence,
                    weight_version=self.current_weight_version, admitted_groups=admitted,
                    gateway_pause_requested=False, timeout_seconds=self._group_refit_timeout)
                while self._group_refit_active:
                    if self._group_refit_error:
                        raise RuntimeError(self._group_refit_error)
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("whole response groups did not drain before refit")
                    self._group_refit_condition.wait(timeout=remaining)
                if self._group_refit_error:
                    raise RuntimeError(self._group_refit_error)
                self._group_refit_draining = False
                emit_rlvp_receipt("group_refit_drain", sequence=self._refit_sequence,
                    weight_version=self.current_weight_version, success=True,
                    admitted_groups=admitted, completed_groups=self._group_refit_drained,
                    active_groups=0, gateway_pause_requested=False,
                    elapsed_seconds=time.monotonic() - started)
            # Admission remains closed. Only now block gateway requests: pausing
            # earlier would strand response sessions that have not reached HTTP.
            if not self._gateway_url:
                raise RuntimeError("whole-group refit requires the configured gateway")
            pause = pause_gateway_generation(self._gateway_url,
                drain_timeout_seconds=self._gateway_drain_timeout)
            status = pause.get("status") or {}
            emit_rlvp_receipt("gateway_pause", sequence=self._refit_sequence,
                weight_version=self.current_weight_version, drained=pause.get("drained"),
                drain_timeout_seconds=self._gateway_drain_timeout,
                paused=status.get("paused"), inflight=status.get("inflight"),
                whole_group_barrier=True)
            if (pause.get("drained") is not True or status.get("paused") is not True
                    or status.get("inflight") != 0):
                raise RuntimeError("gateway failed the required whole-group refit drain: "
                                   + json.dumps(pause, sort_keys=True, default=str))
        except Exception as exc:
            self._fail_group_refit(exc, "prepare")
            raise

    def _resume_whole_group_admission(self) -> None:
        with self._group_refit_condition:
            if self._group_refit_error:
                raise RuntimeError(self._group_refit_error)
            self._refit_pause.set()
            self._group_refit_condition.notify_all()
            emit_rlvp_receipt("group_refit_admission_resumed", sequence=self._refit_sequence,
                             weight_version=self.current_weight_version)

'''


def patch_collector(source):
    """Return checked source; leave historical sources and disabled runs unchanged."""
    if "max_target = max_num_steps - 1" not in source or "from modal_refit_probe" not in source:
        raise RuntimeError("apply the existing transport patches before the group-refit patch")
    source = _replace_once(source, '        print("🌐 Using Polar external rollout collector")\n',
                           INITIALIZE + '        print("🌐 Using Polar external rollout collector")\n')
    source = _replace_once(source, "    def prepare_for_refit(self) -> None:\n",
                           METHODS + "    def prepare_for_refit(self) -> None:\n"
                           "        if self._group_refit_enabled:\n"
                           "            self._prepare_whole_group_refit()\n"
                           "            return\n")
    source = _replace_once(source, "    def resume_after_refit(self) -> None:\n",
                           "    def resume_after_refit(self) -> None:\n"
                           "        if self._group_refit_enabled and self._group_refit_error:\n"
                           "            raise RuntimeError(self._group_refit_error)\n")
    source = _replace_once(source, "        self._refit_pause.set()\n\n    def _run_post_refit_probe",
                           "        if self._group_refit_enabled:\n"
                           "            self._resume_whole_group_admission()\n"
                           "        else:\n"
                           "            self._refit_pause.set()\n\n    def _run_post_refit_probe")
    # Exactly four existing loop waits; the helper's own legacy wait stays intact.
    for indent in ("                ", "                    ", "                        "):
        before = indent + "self._refit_pause.wait()\n"
        source = source.replace("\n" + before, "\n" + indent + "self._wait_for_group_refit()\n")
    source = _replace_once(source,
        "                    self._collect_group(\n"
        "                        generation_weight_version=self.current_weight_version,\n"
        "                        target_weight_version=target,\n"
        "                    )\n",
        "                    if self._group_refit_enabled:\n"
        "                        self._collect_group_at_version(target)\n"
        "                    else:\n"
        "                        self._collect_group(\n"
        "                            generation_weight_version=self.current_weight_version,\n"
        "                            target_weight_version=target,\n"
        "                        )\n")
    source = _replace_once(source,
        "        def run_group(generation_weight_version: int, target: int) -> None:\n",
        "        def run_group(generation_weight_version: int, target: int, refit_admission_id=None) -> None:\n")
    source = _replace_once(source,
        "                    target_weight_version=target,\n"
        "                )\n"
        "            except Exception:  # noqa: BLE001 - mirror the sequential loop's fail-stop\n",
        "                    target_weight_version=target,\n"
        "                    refit_admission_id=refit_admission_id,\n"
        "                )\n"
        "            except Exception as exc:  # noqa: BLE001 - mirror the sequential loop's fail-stop\n"
        "                if self._group_refit_enabled:\n"
        "                    self._fail_group_refit(exc, \"collection\", refit_admission_id)\n")
    source = _replace_once(source,
        "                        pool.submit(run_group, self.current_weight_version, target)\n",
        "                        if self._group_refit_enabled:\n"
        "                            try:\n"
        "                                self._submit_group_at_version(pool, run_group, target)\n"
        "                            except Exception:\n"
        "                                slots.release()\n"
        "                                raise\n"
        "                        else:\n"
        "                            pool.submit(run_group, self.current_weight_version, target)\n")
    source = _replace_once(source,
        "        target_weight_version: int,\n"
        "    ) -> None:\n"
        "        import ray\n\n"
        "        with self._lock:\n"
        "            group_id = self._submitted_groups\n"
        "            self._submitted_groups += 1\n",
        "        target_weight_version: int,\n"
        "        refit_admission_id: int | None = None,\n"
        "    ) -> None:\n"
        "        import ray\n\n"
        "        with self._lock:\n"
        "            group_id = self._submitted_groups\n"
        "            self._submitted_groups += 1\n"
        "        if self._group_refit_enabled:\n"
        "            with self._group_refit_condition:\n"
        "                record = self._group_refit_active.get(refit_admission_id)\n"
        "                if (record is None or generation_weight_version != self.current_weight_version\n"
        "                        or record[\"generation_weight_version\"] != generation_weight_version\n"
        "                        or record[\"target_weight_version\"] != target_weight_version):\n"
        "                    raise RuntimeError(\"response group has no matching version admission\")\n"
        "                record[\"group_id\"] = group_id\n"
        "                emit_rlvp_receipt(\"group_refit_started\", **record)\n")
    ast.parse(source)
    return source


def apply(bridge_root=Path("/opt/rlvp-bridge")):
    """Patch only the runtime collector copy and return its before/after receipt."""
    path = Path(bridge_root) / "src/nemo_polar_bridge/collector.py"
    before = path.read_text()
    after = patch_collector(before)
    receipt = {"path": str(path), "before_sha256": hashlib.sha256(before.encode()).hexdigest(),
               "after_sha256": hashlib.sha256(after.encode()).hexdigest(),
               "opt_in_env": "NEMO_POLAR_GROUP_REFIT_BARRIER"}
    path.write_text(after)
    return [receipt]
