"""Bounded platform/probe fixes on the copied NeMo and Polar source only."""

import ast
import hashlib
from pathlib import Path


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise RuntimeError(f"transport patch anchor count {source.count(before)}: {before[:80]}")
    return source.replace(before, after)


def patch_collector_resume_cursor(source):
    """Preserve one runtime-validated complete prefetched target across resume."""
    source = replace_once(source,
        '''        if self._group_offset_override is not None:
            self._submitted_groups = self._group_offset_override
        else:
            self._submitted_groups = self.initial_weight_version * prompts_per_target(
                self.master_config
            )
''',
        '''        consumed_offset = self.initial_weight_version * prompts_per_target(self.master_config)
        restored_complete_targets = []
        if self._group_offset_override is not None:
            if self._group_offset_override not in (
                consumed_offset, consumed_offset + prompts_per_target(self.master_config)
            ):
                raise ValueError("collector resume offset must cover zero or one complete saved batch")
            self._submitted_groups = self._group_offset_override
            if self._submitted_groups > consumed_offset:
                # Runtime verified this complete batch against original receipts.
                # Remember its target even after the trainer consumes its rows.
                self._generating_targets.add(self.initial_weight_version)
                restored_complete_targets = [self.initial_weight_version]
        else:
            self._submitted_groups = consumed_offset
''')
    return replace_once(source,
        '            group_offset=self._submitted_groups,\n',
        '            group_offset=self._submitted_groups,\n'
        '            restored_complete_targets=restored_complete_targets,\n')


PROMPT_LOGPROBS_ROUTE = '''        @app.post("/rlvp/prompt-logprobs")
        async def rlvp_prompt_logprobs(payload: dict):
            # Fixed trusted diagnostic only: no candidate code, no optimizer,
            # no replay. Temperature one makes prompt likelihood unambiguous.
            import json
            import os
            from pathlib import Path
            fixed = json.loads(Path(os.environ["NRL_FIXED_LOGPROB_PROBE_PATH"]).read_text())
            token_ids = payload.get("token_ids")
            if token_ids != fixed["token_ids"] or not 2 <= len(token_ids) <= 256:
                raise ValueError("only the frozen fixed-prefix diagnostic is accepted")
            sampling = self.SamplingParams(temperature=1.0, top_p=1.0, top_k=-1,
                max_tokens=1, prompt_logprobs=0, logprobs=0, seed=17)
            final = None
            async for output in self.llm.generate(
                prompt={"prompt_token_ids": token_ids}, sampling_params=sampling,
                request_id="rlvp-fixed-prefix-" + str(uuid.uuid4())):
                final = output
            if final is None or final.prompt_logprobs is None:
                raise RuntimeError("vLLM did not return fixed-prefix likelihoods")
            values = [float(final.prompt_logprobs[index][token_ids[index]].logprob)
                      for index in range(1, len(token_ids))]
            return {"token_ids": token_ids, "next_token_logprobs": values,
                    "diagnostic_temperature": 1.0, "discarded_generated_tokens": 1}

'''


def apply(nemo_root=Path("/opt/nemo-rl"), bridge_root=Path("/opt/rlvp-bridge")):
    worker = nemo_root / "nemo_rl/models/generation/vllm/vllm_worker_async.py"
    backend = nemo_root / "nemo_rl/models/generation/vllm/vllm_backend.py"
    collector = bridge_root / "src/nemo_polar_bridge/collector.py"
    grpo = nemo_root / "nemo_rl/algorithms/grpo.py"
    sources = {path: path.read_text() for path in (worker, backend, collector, grpo)}
    updates = {}
    updates[backend] = replace_once(sources[backend],
        '            self._maybe_process_fp8_kv_cache()\n'
        '            self._emit_refit_receipt(True)\n',
        '            self._maybe_process_fp8_kv_cache()\n'
        '            # Collective packing/quantization locals are dead after return.\n'
        '            # Mirror IPC cleanup before allowing any generation to resume.\n'
        '            torch.cuda.synchronize()\n'
        '            before = {"allocated_bytes": torch.cuda.memory_allocated(),\n'
        '                      "reserved_bytes": torch.cuda.memory_reserved()}\n'
        '            gc.collect()\n'
        '            torch.cuda.empty_cache()\n'
        '            after = {"allocated_bytes": torch.cuda.memory_allocated(),\n'
        '                     "reserved_bytes": torch.cuda.memory_reserved()}\n'
        '            print("[rlvp_receipt] " + json.dumps({\n'
        '                "event": "refit_allocator_cleanup",\n'
        '                "sequence": getattr(self, "_rlvp_refit_sequence", 0),\n'
        '                "before": before, "after": after,\n'
        '                "released_reserved_bytes": before["reserved_bytes"] - after["reserved_bytes"],\n'
        '            }, sort_keys=True), flush=True)\n'
        '            self._emit_refit_receipt(True)\n')
    updates[worker] = replace_once(sources[worker],
        '        @app.post("/v1/chat/completions")\n',
        PROMPT_LOGPROBS_ROUTE + '        @app.post("/v1/chat/completions")\n')
    code = patch_collector_resume_cursor(sources[collector])
    code = replace_once(code,
        '    max_target = initial_weight_version + max_num_steps - 1\n',
        '    max_target = max_num_steps - 1  # Native GRPO uses an absolute step horizon.\n')
    code = replace_once(code,
        '    PolarAsyncTrajectoryCollector = ray.remote(**_remote_options)(_PolarAsyncTrajectoryCollector)\n'
        'except Exception:\n',
        '    PolarAsyncTrajectoryCollector = (\n'
        '        ray.remote(**_remote_options)(_PolarAsyncTrajectoryCollector)\n'
        '        if _remote_options else ray.remote(_PolarAsyncTrajectoryCollector)\n'
        '    )\n'
        'except ModuleNotFoundError as exc:\n'
        '    if exc.name != "ray":\n'
        '        raise\n')
    code = replace_once(code, '                    "temperature": 0.0,\n',
        '                    "temperature": float(self.master_config.policy["generation"]["temperature"]),\n')
    code = replace_once(code,
        '            model_id = str(getattr(self.master_config.policy, "model_name", "") or "")\n'
        '            if not model_id:\n'
        '                models = _get_json(f"{base_url}/v1/models", timeout=30.0)\n'
        '                model_id = str(models["data"][0]["id"])\n',
        '            model_id = str(self.master_config.policy.get("model_name") or "")\n'
        '            if not model_id:\n'
        '                raise RuntimeError("configured served model name is missing")\n')
    code = replace_once(code,
        '        This is a diagnostic receipt, so it must never abort training: every\n'
        '        failure is recorded as ``success: false`` with the error class. NeMo\'s\n'
        '        vLLM HTTP adapter has no ``/v1/models`` route, so the served model name\n'
        '        comes from the run configuration first and the route is only a fallback.\n',
        '        Use the configured model and sampling temperature: the native\n'
        '        adapter has no model-list route. Failures abort qualification\n'
        '        after the driver preserves the completed optimizer checkpoint.\n')
    code = replace_once(code,
        '            print(f"⚠️ post-refit probe failed (diagnostic only): {exc}\\n", end="", flush=True)\n',
        '            raise RuntimeError("post-refit generation proof failed") from exc\n')
    code = replace_once(code, '    def start_collection(self, dataloader: Any) -> None:\n',
        '    def start_collection(self, dataloader: Any) -> None:\n'
        '        from modal_refit_probe import record_fp8_probe\n'
        '        record_fp8_probe(self._openai_server_base_url(), "initial", self.current_weight_version)\n')
    code = replace_once(code,
        '        print("🔄 Polar collector resuming gateway generation after NeMo refit")\n',
        '        from modal_refit_probe import record_fp8_probe\n'
        '        record_fp8_probe(self._openai_server_base_url(), "post-refit", self.current_weight_version)\n'
        '        print("🔄 Polar collector resuming gateway generation after NeMo refit")\n')
    updates[collector] = code
    code = sources[grpo]
    start = '''    # Start trajectory collection in background
    collection_task = trajectory_collector.start_collection.remote(dataloader)

    # Ensure collector knows initial weight version
    trajectory_collector.set_weight_version.remote(weight_version)

    print("📦 Started continuous background trajectory collection")
'''
    code = replace_once(code, start, '')
    code = replace_once(code,
        '    print("✅ Policy generation setup complete, proceeding to validation...")\n',
        '    # Polar and the fixed-prefix proof must see the first completed refit.\n'
        '    ray.get(trajectory_collector.set_weight_version.remote(weight_version))\n'
        '    ray.get(trajectory_collector.start_collection.remote(dataloader))\n'
        '    print("📦 Started continuous background trajectory collection")\n\n'
        '    print("✅ Policy generation setup complete, proceeding to validation...")\n')
    code = replace_once(code,
        '                        trajectory_collector.resume_after_refit.remote()\n',
        '                        try:\n'
        '                            ray.get(trajectory_collector.resume_after_refit.remote())\n'
        '                        except Exception as exc:\n'
        '                            # Preserve the real optimizer before failing diagnostics.\n'
        '                            post_refit_error = exc\n')
    code = replace_once(code,
        '                generation_logger_metrics = None\n',
        '                generation_logger_metrics = None\n'
        '                post_refit_error = None\n')
    code = replace_once(code,
        '                        checkpointer.finalize_checkpoint(checkpoint_path)\n\n'
        '            # Logging\n'
        '            # Log training data (match sync GRPO logging payload for parity)\n',
        '                        checkpointer.finalize_checkpoint(checkpoint_path)\n\n'
        '                if post_refit_error is not None:\n'
        '                    raise RuntimeError("post-refit proof failed; native checkpoint preserved") from post_refit_error\n\n'
        '            # Logging\n'
        '            # Log training data (match sync GRPO logging payload for parity)\n')
    updates[grpo] = code
    receipts = []
    for path, content in updates.items():
        ast.parse(content)
        receipts.append({"path": str(path),
                         "before_sha256": hashlib.sha256(sources[path].encode()).hexdigest(),
                         "after_sha256": hashlib.sha256(content.encode()).hexdigest()})
    for path, content in updates.items():
        path.write_text(content)
    return receipts
