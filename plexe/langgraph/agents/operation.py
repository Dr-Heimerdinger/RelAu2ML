"""
Operation Agent — Hybrid Deterministic + Self-Debugging.

Architecture (inspired by Self-Debugging [ICLR 2024] & AutoML-Agent [ICML 2025]):
  Phase 1: Deterministic execution (0 LLM tokens) — run the training script as-is.
  Phase 2: On failure, invoke LLM to analyze the traceback and patch the script,
           then re-execute.  Up to MAX_DEBUG_ITERATIONS rounds (default 3).

This gives zero token cost on the happy path while still recovering from
runtime errors that require code-level reasoning.
"""

import json
import logging
import os
import re
from datetime import datetime
from typing import Optional, Dict, Any

from plexe.langgraph.state import PipelineState, PipelinePhase
from plexe.langgraph.tools.gnn_specialist import execute_training_script
from plexe.langgraph.utils.emitters import set_current_emitter, BaseEmitter

logger = logging.getLogger(__name__)


class OperationAgent:
    """Hybrid agent: deterministic execution + LLM-powered self-debugging on failure."""

    MAX_DEBUG_ITERATIONS = 3

    def __init__(
        self,
        emitter: Optional[BaseEmitter] = None,
        config=None,
        token_tracker=None,
    ):
        self.emitter = emitter
        self.token_tracker = token_tracker
        # Lazy-import to avoid circular deps; store config for LLM creation
        if config is None:
            from plexe.langgraph.config import AgentConfig
            config = AgentConfig.from_env()
        self.config = config

    @property
    def name(self) -> str:
        return "OperationAgent"

    def set_emitter(self, emitter: BaseEmitter):
        self.emitter = emitter

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def invoke(self, state: PipelineState) -> Dict[str, Any]:
        """Execute training and finalize the pipeline."""
        if self.emitter:
            self.emitter.emit_agent_start(self.name, "deterministic")
            set_current_emitter(self.emitter)

        working_dir = state.get("working_dir", ".")
        script_path = state.get(
            "training_script_path", os.path.join(working_dir, "train_script.py")
        )

        # Already completed — just finalize
        if state.get("training_result"):
            return self._finalize_success_from_state(state)

        # Script missing — hard error
        if not os.path.exists(script_path):
            return self._finalize_failure(
                f"Training script not found at {script_path}", state
            )

        # --- Phase 1: deterministic execution (0 tokens) ---
        exec_result = self._execute(script_path)
        if exec_result.get("status") == "success":
            return self._finalize_success(exec_result, state)

        # --- Phase 2: self-debug loop (LLM on failure) ---
        error_msg = self._extract_error(exec_result, exec_result.get("status", "error"))
        previous_errors = []

        # Cache the original script BEFORE entering the debug loop.
        # Each debug iteration fixes the ORIGINAL, not the previous broken patch,
        # to prevent error compounding (e.g. LLM introduces db.get_column() which
        # doesn't exist, and all subsequent iterations inherit that bug).
        try:
            with open(script_path) as f:
                original_script = f.read()
        except Exception as e:
            return self._finalize_failure(f"Cannot read script: {e}", state)

        for attempt in range(1, self.MAX_DEBUG_ITERATIONS + 1):
            logger.info(f"Self-debug attempt {attempt}/{self.MAX_DEBUG_ITERATIONS}")
            if self.emitter:
                self.emitter.emit_thought(
                    self.name,
                    f"Debug attempt {attempt}/{self.MAX_DEBUG_ITERATIONS}: analyzing error and patching script",
                )

            # Always debug from the original script to prevent error compounding
            current_script = original_script

            # Ask LLM to fix
            fixed_script = self._llm_debug(
                current_script, error_msg, state, attempt, previous_errors
            )

            if fixed_script is None:
                logger.warning("LLM could not produce a fix, stopping debug loop")
                break

            # Write patched script
            try:
                with open(script_path, "w") as f:
                    f.write(fixed_script)
                logger.info(f"Patched script written to {script_path}")
            except Exception as e:
                return self._finalize_failure(f"Cannot write patched script: {e}", state)

            # Re-execute
            exec_result = self._execute(script_path)
            if exec_result.get("status") == "success":
                if self.emitter:
                    self.emitter.emit_thought(
                        self.name,
                        f"Training succeeded after debug attempt {attempt}!",
                    )
                return self._finalize_success(exec_result, state)

            # Update error for next iteration
            previous_errors.append(error_msg)
            error_msg = self._extract_error(
                exec_result, exec_result.get("status", "error")
            )

        # All attempts exhausted — restore original script on disk so that
        # orchestrator-level retry (via error_handler → gnn_training) starts clean.
        try:
            with open(script_path, "w") as f:
                f.write(original_script)
        except Exception:
            pass  # best-effort restore

        all_errors = previous_errors + [error_msg]
        exhaustion_msg = (
            f"Training failed after {self.MAX_DEBUG_ITERATIONS} debug attempts.\n\n"
            f"Last error:\n{error_msg[:1500]}"
        )
        return self._finalize_failure(exhaustion_msg, state)

    # ------------------------------------------------------------------
    # Execution helper
    # ------------------------------------------------------------------
    def _execute(self, script_path: str) -> dict:
        """Run the training script and return the result dict."""
        if self.emitter:
            self.emitter.emit_tool_call(
                self.name,
                "execute_training_script",
                {"script_path": script_path, "timeout": 43200},
            )

        result = execute_training_script.invoke(
            {"script_path": script_path, "timeout": 43200}
        )

        status = result.get("status", "error")
        if self.emitter:
            stdout = result.get("stdout", "")
            stderr = result.get("stderr", "")

            # Build output: always include stdout; include stderr on failure
            # (Python tracebacks appear in stderr and are critical for debugging)
            parts = [status]
            if stdout:
                parts.append(stdout)
            if stderr and status != "success":
                parts.append(f"--- stderr ---\n{stderr}")

            self.emitter.emit_tool_result(
                self.name, "execute_training_script",
                "\n\n".join(parts)
            )

        return result

    # ------------------------------------------------------------------
    # LLM Self-Debug (only called on failure)
    # ------------------------------------------------------------------
    def _llm_debug(
        self,
        script: str,
        error: str,
        state: PipelineState,
        attempt: int,
        previous_errors: list,
    ) -> Optional[str]:
        """Invoke LLM to analyze error and produce a fixed script.

        Follows the Self-Debugging approach (Chen et al., ICLR 2024):
        feed traceback + code → LLM explains error → outputs complete fixed script.
        """
        from langchain_core.messages import SystemMessage, HumanMessage
        from plexe.langgraph.config import get_llm_from_model_id
        from plexe.langgraph.prompts.operation import (
            SELF_DEBUG_SYSTEM_PROMPT,
            SELF_DEBUG_PROMPT,
        )

        model_id = self.config.get_model_for_agent("operation")
        llm = get_llm_from_model_id(model_id, temperature=0.1)

        # Build previous-errors context
        prev_ctx = ""
        if previous_errors:
            prev_ctx = "\n\nPrevious failed attempts:\n"
            for i, err in enumerate(previous_errors, 1):
                prev_ctx += f"\n--- Attempt {i} error ---\n{err[:500]}\n"

        working_dir = state.get("working_dir", ".")
        script_path = state.get(
            "training_script_path", os.path.join(working_dir, "train_script.py")
        )
        task_info = state.get("task_info") or {}
        task_type = task_info.get("task_type", "unknown") if isinstance(task_info, dict) else "unknown"

        prompt = SELF_DEBUG_PROMPT.format(
            script=script,
            error=error,
            script_path=script_path,
            task_type=task_type,
            attempt=attempt,
            max_attempts=self.MAX_DEBUG_ITERATIONS,
            previous_errors=prev_ctx,
        )

        try:
            response = llm.invoke([
                SystemMessage(content=SELF_DEBUG_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ])

            # Track token usage
            if self.token_tracker and hasattr(response, "usage_metadata"):
                usage = response.usage_metadata or {}
                self.token_tracker.record(
                    "OperationAgent-debug",
                    usage.get("input_tokens", 0),
                    usage.get("output_tokens", 0),
                )
                if self.emitter:
                    self.emitter.emit_token_update(
                        "OperationAgent-debug",
                        self.token_tracker.summary(),
                    )

            return self._extract_code_block(response.content)

        except Exception as e:
            logger.error(f"LLM debug call failed: {e}")
            return None

    @staticmethod
    def _extract_code_block(text: str) -> Optional[str]:
        """Extract Python code from markdown code block in LLM response."""
        # Try ```python ... ``` first
        match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Fallback: ``` ... ```
        match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    # ------------------------------------------------------------------
    # Error extraction
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_error(exec_result: dict, status: str) -> str:
        """Extract the most useful error message from execution results.

        Python tracebacks appear at the *end* of stderr, but stderr often
        starts with thousands of characters of harmless warnings.  This method
        searches for the traceback first, then falls back to the tail.
        """
        # 1. Explicit "error" key (set on exceptions/timeouts)
        if exec_result.get("error"):
            return exec_result["error"]

        stderr = exec_result.get("stderr", "")
        stdout = exec_result.get("stdout", "")

        # 2. Traceback in stderr
        tb = re.search(r"(Traceback \(most recent call last\):.*)", stderr, re.DOTALL)
        if tb:
            return tb.group(1).strip()

        # 3. Traceback in stdout
        tb = re.search(r"(Traceback \(most recent call last\):.*)", stdout, re.DOTALL)
        if tb:
            return tb.group(1).strip()

        # 4. Tail of stderr
        if stderr:
            return f"(stderr tail)\n{stderr[-1500:]}"

        # 5. Tail of stdout
        if stdout:
            return f"(stdout tail)\n{stdout[-1500:]}"

        return f"Training failed with status: {status}"

    # ------------------------------------------------------------------
    # Finalization helpers
    # ------------------------------------------------------------------
    def _finalize_success(self, exec_result: dict, state: PipelineState) -> Dict[str, Any]:
        """Build the success return dict from execution results."""
        working_dir = state.get("working_dir", ".")
        script_path = state.get(
            "training_script_path", os.path.join(working_dir, "train_script.py")
        )

        # Read training results
        results_path = os.path.join(working_dir, "training_results.json")
        training_results = {}
        if os.path.exists(results_path):
            try:
                with open(results_path) as f:
                    training_results = json.load(f)
                logger.info(f"Training results loaded: {training_results}")
            except Exception as e:
                logger.warning(f"Could not read training results: {e}")
        else:
            training_results = exec_result.get("training_results", {})

        summary = self._build_summary(training_results, working_dir)

        if self.emitter:
            self.emitter.emit_agent_end(self.name, summary)

        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": summary,
                    "timestamp": datetime.now().isoformat(),
                }
            ],
            "training_result": {
                "metrics": training_results,
                "model_path": training_results.get(
                    "model_path", os.path.join(working_dir, "best_model.pt")
                ),
                "script_path": script_path,
            },
            "current_phase": PipelinePhase.COMPLETED.value,
        }

    def _finalize_success_from_state(self, state: PipelineState) -> Dict[str, Any]:
        """Finalize when training_result already exists in state."""
        working_dir = state.get("working_dir", ".")
        summary = self._build_summary(
            state["training_result"].get("metrics", {}), working_dir
        )
        if self.emitter:
            self.emitter.emit_agent_end(self.name, summary)
        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": summary,
                    "timestamp": datetime.now().isoformat(),
                }
            ],
            "current_phase": PipelinePhase.COMPLETED.value,
        }

    def _finalize_failure(self, error_msg: str, state: PipelineState) -> Dict[str, Any]:
        """Build the failure return dict."""
        logger.error(f"Training failed: {error_msg[:500]}")
        if self.emitter:
            self.emitter.emit_agent_end(self.name, f"Training failed: {error_msg[:500]}")
        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": f"Training failed.\n\n{error_msg[:1500]}",
                    "timestamp": datetime.now().isoformat(),
                }
            ],
            "active_errors": [f"Training failed: {error_msg[:500]}"],
            "current_phase": PipelinePhase.OPERATION.value,
        }

    # ------------------------------------------------------------------
    # Summary builder (pure Python, no LLM)
    # ------------------------------------------------------------------
    def _build_summary(self, training_results: dict, working_dir: str) -> str:
        model_path = training_results.get(
            "model_path", os.path.join(working_dir, "best_model.pt")
        )
        tune_metric = training_results.get("tune_metric", "")
        epochs = training_results.get("epochs_trained", "N/A")

        lines = [
            "## Training Complete",
            "",
            f"**Model**: `{model_path}`",
            f"**Epochs trained**: {epochs}",
        ]

        if tune_metric:
            best_val = training_results.get(f"best_val_{tune_metric}", "N/A")
            lines.append(f"**Tuning metric**: {tune_metric} = {best_val}")

        val_metrics = training_results.get("val_metrics", {})
        if val_metrics:
            lines.append("")
            lines.append("### Validation Metrics")
            for k, v in val_metrics.items():
                lines.append(f"  - {k}: {v}")

        test_metrics = training_results.get("test_metrics", {})
        if test_metrics:
            lines.append("")
            lines.append("### Test Metrics")
            for k, v in test_metrics.items():
                lines.append(f"  - {k}: {v}")

        lines.append("")
        lines.append("### Generated Artifacts")
        for fname in [
            "dataset.py",
            "task.py",
            "train_script.py",
            "best_model.pt",
            "training_results.json",
        ]:
            fpath = os.path.join(working_dir, fname)
            exists = os.path.exists(fpath)
            marker = "+" if exists else "-"
            lines.append(f"  {marker} `{fpath}`")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Inference code generation (unchanged)
    # ------------------------------------------------------------------
    def generate_inference_code(self, state: PipelineState) -> str:
        """Generate inference code for the trained model."""
        working_dir = state.get("working_dir", ".")
        task_info = state.get("task_info")
        task_type = "regression"
        if task_info and isinstance(task_info, dict):
            task_type = task_info.get("task_type", "regression")

        if task_type == "link_prediction":
            return self._generate_link_inference_code(working_dir)

        inference_code = f'''"""
Auto-generated inference code for the trained GNN model.
"""

import torch
import sys
import os

sys.path.insert(0, "{working_dir}")

from dataset import GenDataset
from task import GenTask

def load_model(model_path: str):
    """Load the trained model."""
    from plexe.relbench.modeling.nn import HeteroEncoder, HeteroTemporalEncoder, HeteroGraphSAGE
    from plexe.relbench.modeling.graph import make_pkey_fkey_graph
    from plexe.relbench.modeling.utils import get_stype_proposal

    csv_dir = "{working_dir}/csv_files"
    dataset = GenDataset(csv_dir=csv_dir)
    task = GenTask(dataset)
    db = dataset.get_db()

    col_to_stype_dict = get_stype_proposal(db)
    data, col_stats_dict = make_pkey_fkey_graph(
        db,
        col_to_stype_dict=col_to_stype_dict,
        text_embedder_cfg=None,
        cache_dir="{working_dir}/cache/",
    )

    class GNNModel(torch.nn.Module):
        def __init__(self, data, col_stats_dict, hidden_channels=128, out_channels=1):
            super().__init__()
            self.encoder = HeteroEncoder(
                channels=hidden_channels,
                node_to_col_names={{
                    node_type: list(col_stats_dict[node_type].keys())
                    for node_type in data.node_types
                    if node_type in col_stats_dict
                }},
                node_to_col_stats=col_stats_dict,
            )
            self.temporal_encoder = HeteroTemporalEncoder(
                node_types=data.node_types,
                channels=hidden_channels,
            )
            self.gnn = HeteroGraphSAGE(
                node_types=data.node_types,
                edge_types=data.edge_types,
                channels=hidden_channels,
                num_layers=2,
            )
            self.head = torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(hidden_channels, out_channels),
            )

        def forward(self, batch, entity_table):
            x_dict = self.encoder(batch.tf_dict)
            rel_time_dict = self.temporal_encoder(
                batch.seed_time, batch.time_dict, batch.batch_dict
            )
            for node_type in x_dict:
                x_dict[node_type] = x_dict[node_type] + rel_time_dict[node_type]
            x_dict = self.gnn(x_dict, batch.edge_index_dict)
            return self.head(x_dict[entity_table])

    model = GNNModel(data, col_stats_dict)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model, data, task


if __name__ == "__main__":
    model_path = "{working_dir}/best_model.pt"
    model, data, task = load_model(model_path)
    print("Model loaded successfully!")
'''

        return inference_code

    def _generate_link_inference_code(self, working_dir: str) -> str:
        """Generate inference code for link prediction models."""
        return f'''"""
Auto-generated inference code for a trained GNN link prediction model.
"""

import torch
import numpy as np
import sys
import os
from tqdm import tqdm

sys.path.insert(0, "{working_dir}")

from dataset import GenDataset
from task import GenTask

def load_model_and_predict(model_path: str, split: str = "test"):
    """Load the trained link prediction model and generate top-k predictions."""
    from plexe.relbench.modeling.nn import HeteroEncoder, HeteroTemporalEncoder, HeteroGraphSAGE
    from plexe.relbench.modeling.graph import make_pkey_fkey_graph
    from plexe.relbench.modeling.utils import get_stype_proposal, to_unix_time
    from torch_geometric.loader import NeighborLoader
    from torch_geometric.nn import MLP

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    csv_dir = "{working_dir}/csv_files"
    dataset = GenDataset(csv_dir=csv_dir)
    task = GenTask(dataset)
    db = dataset.get_db()

    col_to_stype_dict = get_stype_proposal(db)
    data, col_stats_dict = make_pkey_fkey_graph(
        db,
        col_to_stype_dict=col_to_stype_dict,
        text_embedder_cfg=None,
        cache_dir="{working_dir}/cache/",
    )

    channels = 128
    num_layers = 2
    num_neighbors = [128] * num_layers

    class Model(torch.nn.Module):
        def __init__(self, data, col_stats_dict, num_layers, channels, out_channels):
            super().__init__()
            self.encoder = HeteroEncoder(
                channels=channels,
                node_to_col_names_dict={{
                    node_type: data[node_type].tf.col_names_dict
                    for node_type in data.node_types
                }},
                node_to_col_stats=col_stats_dict,
            )
            self.temporal_encoder = HeteroTemporalEncoder(
                node_types=[
                    nt for nt in data.node_types if "time" in data[nt]
                ],
                channels=channels,
            )
            self.gnn = HeteroGraphSAGE(
                node_types=data.node_types,
                edge_types=data.edge_types,
                channels=channels,
                aggr="sum",
                num_layers=num_layers,
            )
            self.head = MLP(channels, out_channels=out_channels, norm="batch_norm", num_layers=1)
            self.reset_parameters()

        def reset_parameters(self):
            self.encoder.reset_parameters()
            self.temporal_encoder.reset_parameters()
            self.gnn.reset_parameters()
            self.head.reset_parameters()

        def forward(self, batch, entity_table):
            seed_time = batch[entity_table].seed_time
            x_dict = self.encoder(batch.tf_dict)
            rel_time_dict = self.temporal_encoder(seed_time, batch.time_dict, batch.batch_dict)
            for node_type, rel_time in rel_time_dict.items():
                x_dict[node_type] = x_dict[node_type] + rel_time
            x_dict = self.gnn(
                x_dict, batch.edge_index_dict,
                batch.num_sampled_nodes_dict, batch.num_sampled_edges_dict,
            )
            return self.head(x_dict[entity_table][:seed_time.size(0)])

    model = Model(data, col_stats_dict, num_layers, channels, channels).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    split_table = task.get_table(split)
    eval_k = task.eval_k
    src_entity_table = task.src_entity_table
    dst_entity_table = task.dst_entity_table
    num_dst = task.num_dst_nodes

    eval_time_np = to_unix_time(split_table.df[split_table.time_col].iloc[:1])
    eval_seed_time = int(eval_time_np[0])

    dst_loader = NeighborLoader(
        data, num_neighbors=num_neighbors, time_attr="time",
        input_nodes=(dst_entity_table, torch.arange(num_dst)),
        input_time=torch.full((num_dst,), eval_seed_time),
        batch_size=512, shuffle=False, num_workers=0,
    )
    dst_embs = []
    with torch.no_grad():
        for batch in tqdm(dst_loader, desc="Dst embeddings"):
            batch = batch.to(device)
            dst_embs.append(model(batch, dst_entity_table).cpu())
    all_dst_emb = torch.cat(dst_embs, dim=0)

    src_nodes = torch.from_numpy(split_table.df[task.src_entity_col].astype(int).values)
    src_time = torch.from_numpy(to_unix_time(split_table.df[split_table.time_col]))
    src_loader = NeighborLoader(
        data, num_neighbors=num_neighbors, time_attr="time",
        input_nodes=(src_entity_table, src_nodes), input_time=src_time,
        batch_size=512, shuffle=False, num_workers=0,
    )
    src_embs = []
    with torch.no_grad():
        for batch in tqdm(src_loader, desc="Src embeddings"):
            batch = batch.to(device)
            src_embs.append(model(batch, src_entity_table).cpu())
    all_src_emb = torch.cat(src_embs, dim=0)

    pred_indices = []
    for i in range(0, all_src_emb.size(0), 128):
        chunk = all_src_emb[i:i + 128]
        scores = chunk @ all_dst_emb.T
        topk = scores.topk(min(eval_k, scores.size(1)), dim=1).indices
        pred_indices.append(topk)
    pred = torch.cat(pred_indices, dim=0).numpy()

    metrics = task.evaluate(pred, split_table)
    return pred, metrics


if __name__ == "__main__":
    model_path = "{working_dir}/best_model.pt"
    pred, metrics = load_model_and_predict(model_path, split="test")
    print(f"Test metrics: {{metrics}}")
    print(f"Predictions shape: {{pred.shape}}")
'''
