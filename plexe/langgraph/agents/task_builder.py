"""
Task Builder Agent.

This agent builds RelBench Task objects for prediction tasks,
generating SQL queries and complete Python Task classes.
"""

import logging
import os
from typing import Optional, List, Dict, Any

from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage

from plexe.langgraph.agents.base import BaseAgent
from plexe.langgraph.config import AgentConfig
from plexe.langgraph.state import PipelineState, PipelinePhase
from plexe.langgraph.tools.dataset_builder import get_csv_files_info
from plexe.langgraph.tools.task_builder import test_sql_query, register_task_code, validate_dataset_timestamps, fix_dataset_timestamps, analyze_task_structure, determine_lookback_window
from plexe.langgraph.prompts.task_builder import TASK_BUILDER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class TaskBuilderAgent(BaseAgent):
    """Agent for building RelBench Task classes."""

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        additional_tools: Optional[List[BaseTool]] = None,
        token_tracker=None,
    ):
        tools = [
            get_csv_files_info,
            analyze_task_structure,
            determine_lookback_window,
            validate_dataset_timestamps,
            fix_dataset_timestamps,
            test_sql_query,
            register_task_code,
        ]

        if additional_tools:
            tools.extend(additional_tools)

        super().__init__(
            agent_type="task_builder",
            config=config,
            tools=tools,
            token_tracker=token_tracker,
        )
    
    @property
    def system_prompt(self) -> str:
        return TASK_BUILDER_SYSTEM_PROMPT
    
    def invoke(self, state: PipelineState) -> Dict[str, Any]:
        """
        Invoke the agent with retry mechanism for incomplete tasks.
        
        The agent will be re-invoked if it fails to create the task.py file,
        with a stronger prompt to complete the task.
        """
        working_dir = state.get("working_dir", "")
        task_path = os.path.join(working_dir, "task.py") if working_dir else ""
        max_retries = self.config.retry_config.get("task_building", 2)

        for attempt in range(max_retries + 1):
            if attempt > 0:
                logger.warning(f"TaskBuilderAgent retry attempt {attempt}/{max_retries}")
                if self.emitter:
                    self.emitter.emit_thought(
                        self.name, 
                        f"Retry attempt {attempt}: Previous attempt did not create task.py. Retrying with stronger instructions..."
                    )
            
            # Call parent invoke
            result = super().invoke(state)
            
            # Check if task.py was created
            if task_path and os.path.exists(task_path):
                logger.info(f"TaskBuilderAgent successfully created {task_path}")
                return result
            
            if attempt < max_retries:
                # Collect tool results from the failed attempt for retry reuse
                tool_context = ""
                if hasattr(self, '_last_tool_results') and self._last_tool_results:
                    tool_context = "\n\nPREVIOUS TOOL RESULTS (do NOT re-call these tools):\n"
                    for tr in self._last_tool_results:
                        tool_context += f"\n--- {tr['tool_name']} ---\n{tr['content']}\n"

                # Extract expected type and metric for retry message
                intent = state.get("user_intent", {})
                expected_type = "binary_classification"
                metric = ""
                if isinstance(intent, dict):
                    metric = intent.get("evaluation_metric", "")
                    # Reuse internal logic instead of duplicating
                    if metric:
                        m_low = metric.lower().replace("-", "_").replace(" ", "_")
                        if m_low in {"mae", "rmse", "r2", "mse"}: expected_type = "regression"
                        elif m_low in {"auroc", "auc", "roc_auc", "f1", "accuracy", "ap", "average_precision"}: expected_type = "binary_classification"
                        elif m_low in {"map", "precision_at_k", "recall_at_k", "map@k", "precision@k", "recall@k"}: expected_type = "link_prediction"
                        else: expected_type = intent.get("task_type", "binary_classification")
                    else:
                        expected_type = intent.get("task_type", "binary_classification")

                retry_message = self._build_retry_message(working_dir, tool_context, expected_type, metric)

                # Update messages in state for retry
                messages = state.get("messages", [])
                messages.append({
                    "role": "user",
                    "content": retry_message,
                })
                state = {**state, "messages": messages}

        # All retries exhausted, return the last result (which has error)
        return result

    def _build_retry_message(self, working_dir: str, tool_context: str = "",
                             expected_type: str = "", metric: str = "") -> str:
        """Build a retry message to force agent to complete the task."""
        # Determine requirements based on expected_type
        base_class = "RecommendationTask" if expected_type == "link_prediction" else "EntityTask"

        task_type_const = {
            "binary_classification": "TaskType.BINARY_CLASSIFICATION",
            "regression": "TaskType.REGRESSION",
            "link_prediction": "TaskType.LINK_PREDICTION"
        }.get(expected_type, "TaskType.BINARY_CLASSIFICATION")

        metrics_required = {
            "binary_classification": "[roc_auc, average_precision, f1, accuracy]",
            "regression": "[mae, rmse, r2]",
            "link_prediction": "[link_prediction_map, link_prediction_precision, link_prediction_recall]"
        }.get(expected_type, "[]")

        return f"""
CRITICAL: YOU DID NOT COMPLETE YOUR TASK!

The file {working_dir}/task.py does NOT exist or has the WRONG task type.

MANDATORY REQUIREMENTS:
1. Task Type: task_type = {task_type_const}
2. Base Class: class GenTask({base_class}):
3. Metrics: metrics = {metrics_required}
4. User requested metric: {metric or 'Not specified'}

DO NOT re-call tools you already used — their results are below.
DO NOT change the task type — use EXACTLY {expected_type}.

Just execute these steps NOW:
1. Generate the complete GenTask class inheriting from {base_class}
2. Set task_type = {task_type_const}
3. Include metrics = {metrics_required}
4. Call register_task_code(code, "GenTask", "{working_dir}/task.py", "{expected_type}")

{tool_context}
"""
    
    def _build_context(self, state: PipelineState) -> str:
        """Build context with task-specific information."""
        context_parts = []
        
        working_dir = state.get("working_dir", "")
        csv_dir = state.get("csv_dir", "")
        
        if working_dir:
            context_parts.append(f"Working directory: {working_dir}")
        
        if csv_dir:
            context_parts.append(f"CSV directory: {csv_dir}")

        # Inject cached CSV files info so the agent doesn't re-call get_csv_files_info
        if state.get("csv_files_info"):
            csv_info = state["csv_files_info"]
            context_parts.append("\n## CSV Files Info (pre-cached from DatasetBuilder):")
            for f in csv_info.get("files", []):
                context_parts.append(
                    f"  - {f.get('name')}: {f.get('row_count', '?')} rows, "
                    f"columns: {f.get('columns', [])}"
                )
            context_parts.append(
                "NOTE: CSV info is already available above. "
                "Only call get_csv_files_info if you need to re-verify column names."
            )

        # Pre-check: Verify dataset.py exists
        dataset_path = os.path.join(working_dir, "dataset.py") if working_dir else ""
        if dataset_path and not os.path.exists(dataset_path):
            context_parts.append(f"""
## CRITICAL ERROR - PREREQUISITE NOT MET
Dataset file does not exist at: {dataset_path}
The DatasetBuilderAgent must complete successfully before TaskBuilderAgent can run.
You CANNOT proceed without dataset.py. Report this error.
""")
        
        # User intent analysis
        if state.get("user_intent"):
            intent = state["user_intent"]
            pred_target = intent.get('prediction_target', '')
            eval_metric = intent.get('evaluation_metric', '')

            # CRITICAL: Determine final task_type based on metric FIRST
            task_type = intent.get('task_type', 'binary_classification')

            if eval_metric:
                metric_lower = eval_metric.lower().replace("-", "_").replace(" ", "_")

                # Metric ALWAYS overrides keyword-based task_type
                if metric_lower in {"mae", "rmse", "r2", "mse"}:
                    task_type = "regression"
                    logger.info(f"Metric {eval_metric} -> forcing task_type=regression")
                elif metric_lower in {"auroc", "auc", "roc_auc", "f1", "accuracy", "ap", "average_precision"}:
                    task_type = "binary_classification"
                    logger.info(f"Metric {eval_metric} -> forcing task_type=binary_classification")
                elif metric_lower in {"map", "precision_at_k", "recall_at_k", "map@k", "precision@k", "recall@k"}:
                    task_type = "link_prediction"
                    logger.info(f"Metric {eval_metric} -> forcing task_type=link_prediction")

            # Build CONSISTENT context with ONLY the resolved task_type
            context_parts.append(f"\nUser Intent (MANDATORY TO FOLLOW):")
            context_parts.append(f"  - Prediction target: {pred_target}")
            context_parts.append(f"  - REQUIRED task type: {task_type}")
            context_parts.append(f"  - REQUIRED evaluation metric: {eval_metric}")
            context_parts.append(f"  - REQUIRED base class: {'EntityTask' if task_type != 'link_prediction' else 'RecommendationTask'}")

            # Only suggest metrics that match the FINAL task_type
            if task_type == "binary_classification":
                context_parts.append(f"  - You MUST use these metrics: [roc_auc, average_precision, f1, accuracy]")
            elif task_type == "regression":
                context_parts.append(f"  - You MUST use these metrics: [mae, rmse, r2]")
            elif task_type == "link_prediction":
                context_parts.append(f"  - You MUST use these metrics: [link_prediction_map, link_prediction_precision, link_prediction_recall]")
        
        # Schema information
        if state.get("schema_info"):
            schema = state["schema_info"]
            context_parts.append("\n## Schema Information:")
            tables = list(schema.get("tables", {}).keys())
            context_parts.append(f"Available tables: {', '.join(tables)}")

            # Ground task generation to known canonical RelBench tasks when the
            # schema strongly matches a built-in dataset. This reduces semantic
            # drift in target definitions.
            table_set = set(tables)
            if {"users", "events", "event_attendees", "event_interest", "user_friends"}.issubset(table_set):
                context_parts.append(
                    "Canonical reference detected: Event dataset. If user intent matches, "
                    "mirror logic from plexe/relbench/tasks/event.py (UserAttendanceTask, "
                    "UserRepeatTask, UserIgnoreTask). For UserIgnoreTask semantics, use "
                    "event_attendees.status='invited' with threshold (>2), not event_interest.not_interested "
                    "unless the user explicitly asks to predict declines/not_interested."
                )
            elif {"users", "posts", "votes", "comments", "badges"}.issubset(table_set):
                context_parts.append(
                    "Canonical reference detected: Stack dataset. Prefer grounding to "
                    "plexe/relbench/tasks/stack.py for equivalent user intent."
                )
            elif {"customer", "review", "product", "relations"}.intersection(table_set):
                context_parts.append(
                    "Potential canonical reference: Amazon-like schema. Check "
                    "plexe/relbench/tasks/amazon.py before finalizing SQL semantics."
                )
            elif {"ad", "searchstream", "visitstream"}.intersection(table_set):
                context_parts.append(
                    "Potential canonical reference: Avito-like schema. Check "
                    "plexe/relbench/tasks/avito.py for robust sparse-event SQL patterns."
                )
            elif {"transactions", "articles", "customers"}.intersection(table_set):
                context_parts.append(
                    "Potential canonical reference: HM-like schema. Check "
                    "plexe/relbench/tasks/hm.py for task framing and temporal windows."
                )
            
            context_parts.append("\nTable Details:")
            for table_name, table_info in schema.get("tables", {}).items():
                columns = table_info.get("columns", [])
                pk = table_info.get("primary_key", [])
                context_parts.append(f"  - {table_name}:")
                context_parts.append(f"    * Columns: {', '.join([c['name'] for c in columns[:20]])}")
                if pk:
                    context_parts.append(f"    * Primary Key: {pk}")
            
            # Foreign key relationships
            if schema.get("relationships"):
                context_parts.append("\nForeign Key Relationships:")
                for rel in schema["relationships"]:
                    context_parts.append(
                        f"  - {rel['source_table']}.{rel['source_column']} -> {rel['target_table']}.{rel['target_column']}"
                    )
        
        # Dataset information
        if state.get("dataset_info"):
            ds = state["dataset_info"]
            context_parts.append("\n## Dataset Information:")
            context_parts.append(f"  - Class: {ds.get('class_name')}")
            if ds.get("val_timestamp"):
                context_parts.append(f"  - Validation timestamp: {ds.get('val_timestamp')}")
            if ds.get("test_timestamp"):
                context_parts.append(f"  - Test timestamp: {ds.get('test_timestamp')}")
        
        # EDA insights
        if state.get("eda_info"):
            eda = state["eda_info"]
            context_parts.append("\n## EDA Analysis:")
            
            if eda.get("statistics"):
                context_parts.append("Table Statistics:")
                for table, stats in eda["statistics"].items():
                    if isinstance(stats, dict):
                        row_count = stats.get("row_count", "unknown")
                        context_parts.append(f"  - {table}: {row_count} rows")
            
            if eda.get("temporal_analysis"):
                context_parts.append("\nTemporal Analysis:")
                for table, analysis in eda["temporal_analysis"].items():
                    cols = analysis.get("temporal_columns") or analysis.get("time_columns")
                    if cols:
                        context_parts.append(f"  - {table} time columns: {cols}")
                        for col_name, col_info in cols.items():
                            if isinstance(col_info, dict):
                                min_date = col_info.get('min')
                                max_date = col_info.get('max')
                                if min_date and max_date:
                                    context_parts.append(f"    * {col_name}: {min_date} to {max_date}")
            
            # Suggest timedelta based on temporal data
            if eda.get("suggested_timedelta"):
                context_parts.append(f"\nSuggested prediction window: {eda.get('suggested_timedelta')}")
        
        # Compute the val/test gap from dataset_info so the agent knows
        # the maximum timedelta the dataset can support.
        # Fall back to EDA suggested_splits when dataset_info has no timestamps.
        ds = state.get("dataset_info") or {}
        eda = state.get("eda_info") or {}
        splits = eda.get("suggested_splits") or {}

        val_ts_str = ds.get("val_timestamp") or splits.get("train_end", "")
        test_ts_str = ds.get("test_timestamp") or splits.get("val_end", "")
        timestamp_gap_days = None
        if val_ts_str and test_ts_str:
            try:
                import pandas as _pd
                _gap = (_pd.Timestamp(test_ts_str) - _pd.Timestamp(val_ts_str)).days
                timestamp_gap_days = int(_gap)
            except Exception:
                pass

        # Task generation instructions
        working_dir = state.get('working_dir', '')
        csv_dir = state.get('csv_dir', '')
        dataset_file = f"{working_dir}/dataset.py"

        context_parts.append(f"""
## Execute the Mandatory Workflow (system prompt Part 3):
- Dataset file: {dataset_file}
- CSV directory: {csv_dir}
- Task output: {working_dir}/task.py
{f'- Val/test gap: {timestamp_gap_days} days. timedelta MUST be <= {timestamp_gap_days}.' if timestamp_gap_days else ''}
- validate_dataset_timestamps("{dataset_file}", "{csv_dir}", timedelta_days)
- analyze_task_structure("{csv_dir}", event_table, entity_col, time_col, timedelta_days, task_description, entity_table)
- test_sql_query("{csv_dir}", query)
- register_task_code(code, "GenTask", "{working_dir}/task.py", task_type)

Reminders: CTR and in-window **counts/rates** are REGRESSION. Churn-style Pattern A can be hybridized with SUM/COUNT in the forward window. Match `entity_col` to the SQL output alias. `RecommendationTask`: `num_eval_timestamps` must stay 1 only.
""")
        
        return "\n".join(context_parts)
    
    @staticmethod
    def _read_task_type_from_file(task_path: str) -> Optional[str]:
        """Read the TaskType value from a generated task.py file."""
        try:
            with open(task_path) as f:
                for line in f:
                    stripped = line.strip()
                    if stripped.startswith("task_type"):
                        low = stripped.lower()
                        if "regression" in low:
                            return "regression"
                        if "binary" in low or "classification" in low:
                            return "binary_classification"
                        if "link" in low:
                            return "link_prediction"
        except Exception:
            pass
        return None

    @staticmethod
    def _read_base_class_from_file(task_path: str) -> Optional[str]:
        """Return the base class name from the class GenTask(...) declaration."""
        try:
            with open(task_path) as f:
                for line in f:
                    stripped = line.strip()
                    if stripped.startswith("class GenTask"):
                        if "RecommendationTask" in stripped:
                            return "RecommendationTask"
                        if "EntityTask" in stripped:
                            return "EntityTask"
        except Exception:
            pass
        return None

    def _process_result(self, result: Dict[str, Any], state: PipelineState) -> Dict[str, Any]:
        """Process result and extract task information."""
        base_result = super()._process_result(result, state)

        task_info = {}
        generated_code = state.get("generated_code", {})
        working_dir = state.get("working_dir", "")

        task_path = os.path.join(working_dir, "task.py")
        if os.path.exists(task_path):
            task_info["class_name"] = "GenTask"
            task_info["file_path"] = task_path
            logger.info(f"Task file created at: {task_path}")

            intent = state.get("user_intent", {})
            expected_type = (
                intent.get("task_type", "binary_classification")
                if isinstance(intent, dict)
                else "binary_classification"
            )
            # Robust override: infer expected task type from the user's metric if present.
            # Conversational intent extraction can be noisy; metric is usually unambiguous.
            if isinstance(intent, dict):
                metric = (intent.get("evaluation_metric") or "").strip().lower()
                if metric:
                    metric = metric.replace("-", "_").replace(" ", "_")
                    regression_metrics = {"mae", "mse", "rmse", "r2"}
                    binary_metrics = {
                        "roc_auc",
                        "auroc",
                        "auc",
                        "average_precision",
                        "ap",
                        "accuracy",
                        "f1",
                    }
                    link_metrics = {
                        "link_prediction_map",
                        "link_prediction_precision",
                        "link_prediction_recall",
                        "map",
                        "precision_at_k",
                        "recall_at_k",
                    }
                    if metric in regression_metrics:
                        expected_type = "regression"
                    elif metric in binary_metrics:
                        expected_type = "binary_classification"
                    elif metric in link_metrics:
                        expected_type = "link_prediction"
            
            logger.info(f"""
Task Type Decision Trail:
1. User intent task_type (from keywords): {intent.get('task_type')}
2. User evaluation_metric: {intent.get('evaluation_metric')}
3. Metric-inferred type: {expected_type}
4. Final decision: {expected_type}
""")
            # Validate: the generated file's task_type must match the
            # user's intent.  If the LLM wrote the wrong type, flag an
            # error so the retry loop can fix it.
            file_type = self._read_task_type_from_file(task_path)
            task_info["task_type"] = file_type or expected_type
            if file_type and file_type != expected_type:
                base_class = self._read_base_class_from_file(task_path)
                # Never trust link prediction output when the user/metric asked for an
                # entity-level task (binary/regression/multiclass), and vice versa.
                entity_like_expected = expected_type in {
                    "binary_classification",
                    "regression",
                    "multiclass_classification",
                    "multilabel_classification",
                }
                hard_mismatch = (
                    (entity_like_expected and file_type == "link_prediction")
                    or (expected_type == "link_prediction" and file_type != "link_prediction")
                )
                if hard_mismatch:
                    mismatch_msg = (
                        f"Task shape mismatch: generated task is '{file_type}' "
                        f"(base class '{base_class}') but user intent/metric implies "
                        f"'{expected_type}'. Example: AUROC requires EntityTask binary "
                        f"output [timestamp, entity, target], not RecommendationTask. "
                        f"Deleting task.py to force a retry."
                    )
                    logger.error(mismatch_msg)
                    os.remove(task_path)
                    base_result["active_errors"] = [mismatch_msg]
                    base_result["error_history"] = [mismatch_msg]
                    base_result["current_phase"] = PipelinePhase.TASK_BUILDING.value
                    return base_result

                file_consistent = (
                    (file_type == "link_prediction" and base_class == "RecommendationTask")
                    or (file_type != "link_prediction" and base_class == "EntityTask")
                )
                if file_consistent:
                    logger.warning(
                        f"Task type: file says '{file_type}' but user_intent says "
                        f"'{expected_type}'. Trusting the generated file (base class "
                        f"'{base_class}' is consistent)."
                    )
                    task_info["task_type"] = file_type
                else:
                    mismatch_msg = (
                        f"Task type mismatch: file says '{file_type}' "
                        f"(base class '{base_class}') but expected '{expected_type}'. "
                        f"Deleting task.py to force a retry."
                    )
                    logger.error(mismatch_msg)
                    os.remove(task_path)
                    base_result["active_errors"] = [mismatch_msg]
                    base_result["error_history"] = [mismatch_msg]
                    base_result["current_phase"] = PipelinePhase.TASK_BUILDING.value
                    return base_result

            base_result["task_info"] = task_info
            base_result["current_phase"] = PipelinePhase.GNN_TRAINING.value
        else:
            error_msg = f"CRITICAL ERROR: Task file not found at {task_path}. TaskBuilderAgent did not complete its task. The agent must call register_task_code() to generate task.py."
            logger.error(error_msg)
            base_result["active_errors"] = [error_msg]
            base_result["error_history"] = [error_msg]
            # Set task_info with error flag for debugging
            task_info["class_name"] = "GenTask"
            task_info["file_path"] = task_path
            task_info["error"] = error_msg
            task_info["task_type"] = "binary_classification"
            base_result["task_info"] = task_info
            # Do NOT advance phase - stay in task building
            base_result["current_phase"] = PipelinePhase.TASK_BUILDING.value
        
        if generated_code:
            base_result["generated_code"] = generated_code
        
        return base_result
