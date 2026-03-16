from typing import Dict, Any, List
from langchain_core.tools import tool as langchain_tool


def _round_to_clean_interval(raw_days: int) -> str:
    """Round raw lookback days to a human-readable interval string."""
    if raw_days >= 548:
        return "2 years"
    elif raw_days >= 274:
        return "1 year"
    elif raw_days >= 137:
        return "6 months"
    elif raw_days >= 60:
        return "3 months"
    else:
        return f"{raw_days} days"


def _find_entity_pk(entity_df, entity_col: str) -> str:
    """Find the primary key column in entity table matching entity_col."""
    # Direct match
    if entity_col in entity_df.columns:
        return entity_col
    # Try 'id' or 'Id' (common in stack, trial datasets)
    for candidate in ['id', 'Id', 'ID']:
        if candidate in entity_df.columns:
            return candidate
    # Try case-insensitive match
    for col in entity_df.columns:
        if col.lower() == entity_col.lower():
            return col
    return None


@langchain_tool
def analyze_task_structure(
    csv_dir: str,
    event_table: str,
    entity_col: str,
    time_col: str,
    timedelta_days: int,
    task_description: str,
    entity_table: str = "",
) -> Dict[str, Any]:
    """
    Analyze data structure to recommend SQL patterns for the training table.
    MUST be called before designing the SQL query (Step 4 of the workflow).

    Returns a multi-faceted analysis with:
    - Entity source analysis (dimension table vs event-derived)
    - Temporal gap analysis (key for Pattern A vs B decision)
    - Semantic signals (churn, link prediction, regression)
    - Schema hints (intermediate tables, potential JOINs)
    - Ranked pattern candidates with confidence and reasoning
    - Building block suggestions (CTE, nested JOIN, etc.)

    The LLM should use ALL sections to select the best pattern. The tool provides
    evidence and ranked candidates, but the LLM makes the final decision.

    Args:
        csv_dir: Directory containing CSV files
        event_table: Name of the event/fact table CSV (without .csv extension)
        entity_col: Column name identifying the entity (e.g., 'customer_id')
        time_col: Column name for the temporal column in the event table (e.g., 't_dat')
        timedelta_days: The prediction window in days (e.g., 7 for weekly churn)
        task_description: Brief description of the task (e.g., 'predict customer churn')
        entity_table: Name of the entity/dimension table CSV (without .csv extension).
            If provided, checks for creation/start date columns and entity coverage.

    Returns:
        Structured analysis with entity_source, temporal, semantic, schema_hints,
        pattern_candidates (ranked), and building_blocks suggestions
    """
    import os
    import numpy as np
    import pandas as pd

    try:
        csv_dir = os.path.abspath(csv_dir)
        file_path = os.path.join(csv_dir, f"{event_table}.csv")

        if not os.path.exists(file_path):
            return {"status": "error", "error": f"Event table CSV not found: {file_path}"}

        df = pd.read_csv(file_path)

        if entity_col not in df.columns:
            return {"status": "error", "error": f"Entity column '{entity_col}' not found in {event_table}.csv"}
        if time_col not in df.columns:
            return {"status": "error", "error": f"Time column '{time_col}' not found in {event_table}.csv"}

        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        df = df.dropna(subset=[time_col])

        task_lower = task_description.lower()
        timedelta = pd.Timedelta(days=timedelta_days)
        date_range = df[time_col].max() - df[time_col].min()
        date_range_days = max(int(date_range.days), 1)

        # ================================================================
        # Section 1: Entity Source Analysis
        # ================================================================
        entity_source = {
            "has_dedicated_entity_table": False,
            "entity_table_row_count": 0,
            "entity_table_has_creation_date": False,
            "creation_date_column": None,
            "entities_in_event_table": int(df[entity_col].nunique()),
            "entity_coverage_ratio": 1.0,  # default: all entities come from events
        }

        entity_creation_col = None
        entity_table_cols = []

        if entity_table:
            entity_file = os.path.join(csv_dir, f"{entity_table}.csv")
            if os.path.exists(entity_file):
                try:
                    entity_df = pd.read_csv(entity_file)
                    entity_table_cols = list(entity_df.columns)
                    entity_pk = _find_entity_pk(entity_df, entity_col)

                    entity_source["has_dedicated_entity_table"] = True
                    entity_source["entity_table_row_count"] = len(entity_df)

                    # Entity coverage: what fraction of entity table entities appear in events
                    if entity_pk:
                        entity_ids_in_table = entity_df[entity_pk].nunique()
                        entity_ids_in_events = df[entity_col].nunique()
                        entity_source["entity_coverage_ratio"] = round(
                            entity_ids_in_events / max(entity_ids_in_table, 1), 3
                        )

                    # Check for creation/start date columns
                    creation_signals = [
                        'creation', 'created', 'start_date', 'publish',
                        'registered', 'signup', 'join_date', 'enrollment',
                        # finance
                        'opened', 'open_date', 'inception', 'originated', 'activated', 'account_open',
                        # healthcare
                        'admission', 'admitted', 'intake', 'first_seen', 'onset',
                        # media/retail
                        'release', 'released', 'launched', 'listed', 'listed_date', 'added_date',
                        # general
                        'available', 'established', 'founded', 'first_active',
                    ]
                    sample_df = entity_df.head(100)
                    for col in sample_df.columns:
                        col_lower = col.lower()
                        if col_lower.endswith('id') or col_lower == 'id':
                            continue
                        if any(sig in col_lower for sig in creation_signals):
                            try:
                                parsed = pd.to_datetime(sample_df[col], errors='coerce')
                                if parsed.notna().sum() > len(sample_df) * 0.3:
                                    entity_creation_col = col
                                    entity_source["entity_table_has_creation_date"] = True
                                    entity_source["creation_date_column"] = col
                                    break
                            except Exception:
                                pass
                except Exception:
                    pass

        # ================================================================
        # Section 2: Temporal Analysis
        # ================================================================
        entity_event_counts = df.groupby(entity_col).size()
        median_events_per_entity = float(entity_event_counts.median())
        num_windows = max(date_range / timedelta, 1)
        total_events = len(df)
        unique_entities = int(df[entity_col].nunique())
        events_per_day = total_events / date_range_days

        # Compute inter-event gap statistics (the critical fix)
        inter_event_gap_median = 0.0
        inter_event_gap_p90 = 0.0
        inter_event_gap_max = 0.0
        try:
            df_sorted = df[[entity_col, time_col]].sort_values([entity_col, time_col])
            df_sorted = df_sorted.assign(
                prev_time=df_sorted.groupby(entity_col)[time_col].shift(1)
            )
            gap_series = (df_sorted[time_col] - df_sorted["prev_time"]).dt.days.dropna()
            if len(gap_series) > 0:
                inter_event_gap_median = float(gap_series.median())
                inter_event_gap_p90 = float(gap_series.quantile(0.9))
                inter_event_gap_max = float(gap_series.max())
        except Exception:
            inter_event_gap_median = float(date_range_days)
            inter_event_gap_p90 = float(date_range_days)
            inter_event_gap_max = float(date_range_days)

        # max_gap_exceeds_timedelta is true when either:
        # - P90 gap > 2x timedelta (many entities have gaps much larger than the window), OR
        # - Max gap > 3x timedelta (structural gaps exist, e.g., off-seasons, that
        #   self.timedelta can't span — even if typical gaps are small)
        # The 2x multiplier on p90 avoids false positives when gaps are just slightly
        # above timedelta (common in high-frequency data like daily retail transactions).
        max_gap_exceeds_timedelta = (
            inter_event_gap_p90 > timedelta_days * 2
            or inter_event_gap_max > timedelta_days * 3
        )

        # Always compute a data-driven lookback (useful regardless of pattern)
        raw_lookback_days = max(
            int(max(inter_event_gap_p90, inter_event_gap_max / 2) * 3),
            timedelta_days * 4,
            30,
        )
        raw_lookback_days = min(raw_lookback_days, date_range_days)
        suggested_lookback_interval = _round_to_clean_interval(raw_lookback_days)

        temporal = {
            "data_range_days": date_range_days,
            "total_events": int(total_events),
            "unique_entities": unique_entities,
            "events_per_entity_median": round(median_events_per_entity, 2),
            "events_per_entity_p25": round(float(entity_event_counts.quantile(0.25)), 2),
            "events_per_entity_p75": round(float(entity_event_counts.quantile(0.75)), 2),
            "events_per_window_median": round(float(median_events_per_entity / num_windows), 4),
            "inter_event_gap_median_days": round(inter_event_gap_median, 1),
            "inter_event_gap_p90_days": round(inter_event_gap_p90, 1),
            "inter_event_gap_max_days": round(inter_event_gap_max, 1),
            "max_gap_exceeds_timedelta": max_gap_exceeds_timedelta,
            "suggested_lookback_days": raw_lookback_days,
            "suggested_lookback_interval": suggested_lookback_interval,
            "timedelta_days": timedelta_days,
        }

        # ================================================================
        # Section 3: Semantic Signals
        # ================================================================
        churn_keywords = [
            'churn', 'no activity', 'no transaction', 'inactive',
            'absence', 'will not', "won't", 'stop', 'leave',
            'retain', 'retention', 'lapse', 'dormant',
        ]
        link_keywords = [
            'list of', 'recommend', 'which items', 'purchase list',
            'link prediction', 'map@', 'precision@', 'recall@',
            # social/graph/healthcare
            'who will', 'which users', 'which entities', 'will connect', 'will interact',
            'co-occur', 'relation', 'graph link', 'drug interaction', 'co-authorship',
        ]
        catalog_entity_signals = [
            'article', 'product', 'item', 'post', 'listing',
            'study', 'trial', 'facility', 'site', 'ad',
            # media/music
            'track', 'song', 'album', 'artist', 'video', 'movie', 'episode', 'show', 'playlist',
            # finance
            'account', 'security', 'fund', 'instrument', 'loan', 'invoice', 'order',
            # healthcare
            'patient', 'medication', 'drug', 'procedure', 'diagnosis',
            # general
            'property', 'unit', 'ticket', 'job', 'project',
        ]
        aggregate_regression_signals = [
            'total', 'sum of', 'sales', 'revenue', 'ltv',
            'popularity', 'count of', 'clicks', 'votes', 'ctr',
            'how much', 'how many', 'number of', 'mae', 'rmse',
            # media/music
            'streams', 'plays', 'listen', 'views', 'watch time', 'play count',
            # finance
            'balance', 'return', 'yield', 'spend', 'amount', 'exposure', 'volume',
            # healthcare
            'readmission', 'length of stay', 'lab value', 'mortality', 'dosage',
            # general
            'score', 'rating', 'duration', 'delay', 'distance', 'throughput',
        ]
        active_qualifier_keywords = [
            'active', 'recently', 'recent ', 'engaged', 'retained',
            'previously purchased', 'previously active',
        ]

        is_churn_task = any(kw in task_lower for kw in churn_keywords)
        is_link_task = any(kw in task_lower for kw in link_keywords)
        entity_is_catalog = any(kw in entity_col.lower() for kw in catalog_entity_signals)
        is_aggregate_regression = any(kw in task_lower for kw in aggregate_regression_signals)
        has_active_qualifier = any(kw in task_lower for kw in active_qualifier_keywords)
        is_all_entities_task = (
            entity_is_catalog and is_aggregate_regression
            and not is_churn_task and not has_active_qualifier
        )

        semantic = {
            "churn_signals_detected": is_churn_task,
            "link_prediction_signals_detected": is_link_task,
            "all_entity_regression_signals": is_all_entities_task,
            "active_qualifier_detected": has_active_qualifier,
        }

        # ================================================================
        # Section 4: Schema Hints
        # ================================================================
        event_table_cols = list(df.columns)
        potential_join_tables = []
        try:
            event_col_set = set(df.columns)
            for f in os.listdir(csv_dir):
                if f.endswith('.csv'):
                    other_name = f.replace('.csv', '')
                    if other_name in [event_table, entity_table]:
                        continue
                    try:
                        other_df = pd.read_csv(os.path.join(csv_dir, f), nrows=5)
                        shared = set(other_df.columns) & event_col_set
                        # Filter out generic columns that cause false join signals
                        GENERIC_COLS = {
                            'id', 'index', 'timestamp', 'date', 'time', 'type', 'status',
                            'name', 'description', 'value', 'created_at', 'updated_at',
                            'created', 'modified', 'updated',
                        }
                        shared = {c for c in shared if c.lower() not in GENERIC_COLS}
                        if shared:
                            potential_join_tables.append({
                                "table": other_name,
                                "shared_columns_with_event": list(shared),
                                "columns": list(other_df.columns),
                            })
                    except Exception:
                        pass
        except Exception:
            pass

        # ---- Categorical column profiling ----
        # Detect low-cardinality columns that may need filtering
        # (e.g., PostTypeId, VoteTypeId, StatusId, outcome_type)
        TYPE_NAME_SIGNALS = [
            'type', 'kind', 'category', 'status', 'class', 'group', 'mode',
            'level', 'tier', 'role', 'source', 'channel', 'method', 'reason',
        ]
        CARDINALITY_MIN = 2
        CARDINALITY_MAX = 20

        def _profile_categorical_columns(table_df, table_name, exclude_cols):
            profiles = []
            for col in table_df.columns:
                if col in exclude_cols:
                    continue
                try:
                    n_unique = table_df[col].nunique()
                except Exception:
                    continue
                if n_unique < CARDINALITY_MIN or n_unique > CARDINALITY_MAX:
                    continue
                vc = table_df[col].value_counts(dropna=False).head(20)
                value_dist = {str(k): int(v) for k, v in vc.items()}
                col_lower = col.lower()
                is_type_col = any(sig in col_lower for sig in TYPE_NAME_SIGNALS)
                profiles.append({
                    "column": col,
                    "table": table_name,
                    "n_distinct": int(n_unique),
                    "value_distribution": value_dist,
                    "is_likely_type_column": is_type_col,
                })
            return profiles

        exclude_event = {entity_col, time_col}
        event_cat_profiles = _profile_categorical_columns(df, event_table, exclude_event)

        entity_cat_profiles = []
        if entity_table and entity_source["has_dedicated_entity_table"]:
            try:
                ent_file = os.path.join(csv_dir, f"{entity_table}.csv")
                ent_df = pd.read_csv(ent_file)
                ent_pk = _find_entity_pk(ent_df, entity_col)
                exclude_ent = {ent_pk} if ent_pk else set()
                if entity_creation_col:
                    exclude_ent.add(entity_creation_col)
                entity_cat_profiles = _profile_categorical_columns(
                    ent_df, entity_table, exclude_ent
                )
            except Exception:
                pass

        # ---- Sentinel value detection on entity ID columns ----
        sentinel_warnings = []
        if entity_table and entity_source["has_dedicated_entity_table"]:
            try:
                ent_file = os.path.join(csv_dir, f"{entity_table}.csv")
                ent_df = pd.read_csv(ent_file)
                ent_pk = _find_entity_pk(ent_df, entity_col)
                if ent_pk and ent_pk in ent_df.columns:
                    id_series = ent_df[ent_pk]
                    null_count = int(id_series.isna().sum())
                    if null_count > 0:
                        sentinel_warnings.append({
                            "table": entity_table, "column": ent_pk,
                            "issue": "null_entity_ids", "count": null_count,
                            "recommendation": f"Filter: {ent_pk} IS NOT NULL",
                        })
                    if pd.api.types.is_numeric_dtype(id_series):
                        for sentinel in [-1, 0]:
                            sc = int((id_series == sentinel).sum())
                            if 0 < sc < len(id_series) * 0.1:
                                sentinel_warnings.append({
                                    "table": entity_table, "column": ent_pk,
                                    "issue": f"sentinel_value_{sentinel}", "count": sc,
                                    "recommendation": f"Filter: {ent_pk} != {sentinel}",
                                })
            except Exception:
                pass

        # Check entity_col in event table for FK sentinels
        if entity_col in df.columns:
            id_ev = df[entity_col]
            null_ev = int(id_ev.isna().sum())
            if 0 < null_ev < len(df) * 0.5:
                sentinel_warnings.append({
                    "table": event_table, "column": entity_col,
                    "issue": "null_fk_entity_ids", "count": null_ev,
                    "recommendation": f"Filter: {entity_col} IS NOT NULL",
                })
            if pd.api.types.is_numeric_dtype(id_ev):
                for sentinel in [-1, 0]:
                    sc = int((id_ev == sentinel).sum())
                    if 0 < sc < len(df) * 0.1:
                        sentinel_warnings.append({
                            "table": event_table, "column": entity_col,
                            "issue": f"sentinel_fk_value_{sentinel}", "count": sc,
                            "recommendation": f"Filter: {entity_col} != {sentinel}",
                        })

        schema_hints = {
            "event_table_columns": event_table_cols,
            "entity_table_columns": entity_table_cols,
            "potential_join_tables": potential_join_tables,
            "categorical_columns": event_cat_profiles + entity_cat_profiles,
            "sentinel_warnings": sentinel_warnings,
        }

        # ================================================================
        # Section 5: Ranked Pattern Candidates
        # ================================================================
        candidates: List[Dict[str, Any]] = []
        has_entity_table = entity_source["has_dedicated_entity_table"]
        has_creation_date = entity_source["entity_table_has_creation_date"]

        # Link prediction: highest priority
        if is_link_task:
            candidates.append({
                "pattern": "Link",
                "confidence": 0.95,
                "lookback": None,
                "reason": "Link prediction keywords detected. Use LEFT JOIN event table + LIST(DISTINCT).",
            })

        # Churn/absence: second priority
        if is_churn_task:
            candidates.append({
                "pattern": "A",
                "confidence": 0.95,
                "lookback": "self.timedelta",
                "reason": (
                    f"Churn/absence keywords detected. Lookback MUST equal self.timedelta "
                    f"(={timedelta_days}d) for symmetric active/inactive window."
                ),
            })

        # Pattern D: entity with creation date
        if has_creation_date and not is_all_entities_task:
            conf = 0.85 if not is_churn_task else 0.3
            candidates.append({
                "pattern": "D",
                "confidence": conf,
                "lookback": None,
                "reason": (
                    f"Entity table '{entity_table}' has creation date column "
                    f"'{entity_creation_col}'. Use LEFT JOIN entity ON "
                    f"{entity_creation_col} <= timestamp."
                ),
            })

        # Pattern C: all-entity regression
        if is_all_entities_task:
            candidates.append({
                "pattern": "C",
                "confidence": 0.85,
                "lookback": None,
                "reason": (
                    f"All-entity regression: entity_col='{entity_col}' is a catalog entity "
                    f"and target is aggregate. Zero is valid (no filter). Use COALESCE(agg, 0)."
                ),
            })

        # Pattern A vs B for non-churn, non-link, non-all-entity, non-creation-date tasks
        if not is_link_task and not is_churn_task and not is_all_entities_task:
            if not has_creation_date or is_all_entities_task:
                # The key fix: use gap analysis instead of frequency
                if max_gap_exceeds_timedelta or not has_entity_table:
                    # Large gaps or no entity table -> Pattern B
                    candidates.append({
                        "pattern": "B",
                        "confidence": 0.80,
                        "lookback": f"'{suggested_lookback_interval}'",
                        "reason": (
                            f"P90 inter-event gap ({inter_event_gap_p90:.0f}d) "
                            f"{'exceeds' if max_gap_exceeds_timedelta else 'and no entity table: entities derived from events.'} "
                            f"{'timedelta (' + str(timedelta_days) + 'd). EXISTS with self.timedelta would miss entities during long gaps.' if max_gap_exceeds_timedelta else ''} "
                            f"Use WHERE IN with {suggested_lookback_interval} lookback."
                        ),
                    })
                    if has_entity_table:
                        candidates.append({
                            "pattern": "A",
                            "confidence": 0.30,
                            "lookback": "self.timedelta",
                            "reason": (
                                f"Entity table exists, but p90 gap ({inter_event_gap_p90:.0f}d) > "
                                f"timedelta ({timedelta_days}d) suggests B is more appropriate."
                            ),
                        })
                else:
                    # Small gaps + entity table -> Pattern A
                    candidates.append({
                        "pattern": "A",
                        "confidence": 0.75,
                        "lookback": "self.timedelta",
                        "reason": (
                            f"Entity table exists and p90 gap ({inter_event_gap_p90:.0f}d) <= "
                            f"timedelta ({timedelta_days}d). Events are regular enough for "
                            f"EXISTS with self.timedelta."
                        ),
                    })
                    candidates.append({
                        "pattern": "B",
                        "confidence": 0.35,
                        "lookback": f"'{suggested_lookback_interval}'",
                        "reason": (
                            f"Could use Pattern B with {suggested_lookback_interval} lookback, "
                            f"but entity table + small gaps favor Pattern A."
                        ),
                    })

        # Sort by confidence descending
        candidates.sort(key=lambda c: c["confidence"], reverse=True)

        # ================================================================
        # Section 6: Building Block Suggestions
        # ================================================================
        # Detect nested JOIN need: when entity table has an ID that is shared
        # with the event table (direct or via case-insensitive match), suggesting
        # the entity-event pre-join pattern (e.g., UserInfo LEFT JOIN VisitStream)
        entity_col_lower = entity_col.lower()
        entity_col_in_event = (
            entity_col in event_table_cols
            or any(c.lower() == entity_col_lower for c in event_table_cols)
        )
        # needs_nested_join: entity table exists and shares the entity_col with
        # event table (suggesting entity-event pre-join for temporal filtering)
        _needs_nested_join = (
            has_entity_table
            and entity_col_in_event
            and len(potential_join_tables) > 0
        )

        # Detect CTE need: multiple event-like tables (>= 2 tables share columns
        # with the event table) or multiple related event sources
        _needs_cte = len(potential_join_tables) >= 2

        _has_categorical = bool(event_cat_profiles or entity_cat_profiles)

        building_blocks = {
            "needs_cte": _needs_cte,
            "needs_nested_join": _needs_nested_join,
            "needs_quality_filter": (
                any(
                    kw in task_lower for kw in [
                        'rating', 'star', 'quality', 'detailed', 'specific',
                        'click', 'successful', 'primary', 'severe',
                    ]
                )
                or _has_categorical
            ),
            "needs_having": any(
                kw in task_lower for kw in [
                    'assuming', 'given that', 'if active', 'has at least',
                    'with at least', 'only for', 'who have', 'that have',
                ]
            ),
            "categorical_filter_hint": (
                "Low-cardinality type/category columns detected in entity or event tables. "
                "Review schema_hints.categorical_columns to determine if the task requires "
                "filtering by specific values (e.g., only questions, only upvotes, only primary outcomes). "
                "Also check schema_hints.sentinel_warnings for entity ID sentinel values to exclude."
                if (_has_categorical or sentinel_warnings)
                else None
            ),
        }

        # ================================================================
        # Build the top-level recommendation (for backward compatibility)
        # ================================================================
        top_candidate = candidates[0] if candidates else {"pattern": "A", "lookback": "self.timedelta"}
        top_pattern = top_candidate["pattern"]
        top_lookback = top_candidate.get("lookback")

        pattern_descriptions = {
            "A": "Cross join entity table + EXISTS filter with self.timedelta lookback.",
            "B": "LEFT JOIN event table + WHERE IN subquery with data-driven lookback.",
            "C": "Cross join entity table + LEFT JOIN events + COALESCE(agg, 0). No activity filter.",
            "D": "LEFT JOIN entity table ON creation_date <= timestamp + LEFT JOIN events for forward window.",
            "Link": "LEFT JOIN event table + GROUP BY source entity + LIST(DISTINCT dest entity).",
        }

        # Recommend num_eval_timestamps when events are broadly sparse (not
        # just outlier max gaps).  Use p90 gap only — max gap is too sensitive
        # to outliers (e.g., Avito's 25d max gap in a 25d range with p90=0).
        # p90 > 2x timedelta means the majority of entities have gaps exceeding
        # the prediction window, so a single eval timestamp will likely land in
        # a dead period → empty eval tables → -inf metrics.
        events_broadly_sparse = inter_event_gap_p90 > timedelta_days * 2
        recommended_num_eval_timestamps = 40 if events_broadly_sparse else 1

        return {
            "status": "success",
            # Top-level recommendation (backward compat)
            "recommended_pattern": top_pattern,
            "lookback_window": top_lookback,
            "pattern_description": pattern_descriptions.get(top_pattern, ""),
            "reasoning": top_candidate.get("reason", ""),
            # num_eval_timestamps recommendation (1 = omit, 40 = set explicitly)
            "recommended_num_eval_timestamps": recommended_num_eval_timestamps,
            # Rich analysis sections
            "entity_source": entity_source,
            "temporal": temporal,
            "semantic": semantic,
            "schema_hints": schema_hints,
            "pattern_candidates": candidates,
            "building_blocks": building_blocks,
            # Backward compat fields
            "is_churn_task": is_churn_task,
            "is_all_entities_task": is_all_entities_task,
            "entity_creation_col": entity_creation_col,
            "evidence": {
                "total_events": int(total_events),
                "unique_entities": unique_entities,
                "events_per_day": round(events_per_day, 2),
                "median_events_per_entity": round(median_events_per_entity, 2),
                "median_events_per_window": round(float(median_events_per_entity / num_windows), 4),
                "inter_event_gap_p90_days": round(inter_event_gap_p90, 1),
                "max_gap_exceeds_timedelta": max_gap_exceeds_timedelta,
                "date_range_days": date_range_days,
                "timedelta_days": timedelta_days,
            },
            "must_use_lookback": top_lookback,
            "must_use_pattern": top_pattern,
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}


@langchain_tool
def determine_lookback_window(
    csv_dir: str,
    event_table: str,
    entity_col: str,
    time_col: str,
    timedelta_days: int,
    task_description: str,
    entity_table: str = ""
) -> Dict[str, Any]:
    """
    Backward-compatible wrapper for analyze_task_structure().
    Calls analyze_task_structure and returns results in the same format.
    Prefer calling analyze_task_structure() directly for richer analysis.

    Args:
        csv_dir: Directory containing CSV files
        event_table: Name of the event/fact table CSV (without .csv extension)
        entity_col: Column name identifying the entity (e.g., 'customer_id')
        time_col: Column name for the temporal column in the event table
        timedelta_days: The prediction window in days
        task_description: Brief description of the task
        entity_table: Name of the entity/dimension table CSV (without .csv extension)

    Returns:
        Analysis results with recommended pattern and lookback window
    """
    return analyze_task_structure.invoke({
        "csv_dir": csv_dir,
        "event_table": event_table,
        "entity_col": entity_col,
        "time_col": time_col,
        "timedelta_days": timedelta_days,
        "task_description": task_description,
        "entity_table": entity_table,
    })


@langchain_tool
def test_sql_query(
    csv_dir: str,
    query: str
) -> Dict[str, Any]:
    """
    Test a SQL query against CSV files using DuckDB.
    
    Args:
        csv_dir: Directory containing CSV files
        query: SQL query to test
    
    Returns:
        Query results or error
    """
    import duckdb
    import os
    import pandas as pd
    
    try:
        # Convert to absolute path to ensure files are found
        csv_dir = os.path.abspath(csv_dir)
        
        if not os.path.exists(csv_dir):
            return {
                "status": "error",
                "error": f"CSV directory does not exist: {csv_dir}"
            }
        
        conn = duckdb.connect(':memory:')
        
        # Load all CSV files as tables
        for f in os.listdir(csv_dir):
            if f.endswith('.csv'):
                table_name = f.replace('.csv', '')
                file_path = os.path.join(csv_dir, f)
                conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM read_csv_auto('{file_path}')")
        
        # Create a data-aware dummy timestamp_df for SQL validation.
        # At runtime, the real timestamp_df will be provided by the task framework.
        # Here we derive a representative date from the loaded tables so that
        # temporal JOINs find matching rows, producing a more realistic test.
        all_dates = []
        for f in os.listdir(csv_dir):
            if f.endswith('.csv'):
                try:
                    tbl = conn.execute(f"SELECT * FROM {f.replace('.csv', '')} LIMIT 0").fetchdf()
                    for col in tbl.columns:
                        col_lower = col.lower()
                        if col_lower.endswith('id') or col_lower == 'id':
                            continue
                        try:
                            sample = conn.execute(
                                f"SELECT DISTINCT \"{col}\" FROM {f.replace('.csv', '')} "
                                f"WHERE \"{col}\" IS NOT NULL LIMIT 500"
                            ).fetchdf()
                            parsed = pd.to_datetime(sample[col], errors='coerce').dropna()
                            if len(parsed) > len(sample) * 0.3:
                                all_dates.extend(parsed.tolist())
                        except Exception:
                            pass
                except Exception:
                    pass

        if all_dates:
            sorted_dates = sorted(all_dates)
            mid = sorted_dates[len(sorted_dates) // 2]
            base_ts = pd.Timestamp(mid).normalize()
        else:
            base_ts = pd.Timestamp('2020-01-01')

        timestamps_dummy = pd.DataFrame({
            'timestamp': pd.date_range(base_ts, periods=3, freq='7D')
        })
        conn.register("timestamp_df", timestamps_dummy)
        
        result = conn.execute(query).fetchdf()
        
        return {
            "status": "success",
            "columns": list(result.columns),
            "row_count": len(result),
            "sample_data": result.head(10).to_dict(orient='records')
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@langchain_tool
def register_task_code(
    code: str,
    class_name: str,
    file_path: str,
    task_type: str
) -> Dict[str, str]:
    """
    Register generated Task class code.
    
    Args:
        code: Python code for the Task class
        class_name: Name of the Task class
        file_path: Full path where the code will be saved (e.g., workdir/session-xxx/task.py)
        task_type: Type of task (regression, binary_classification, multiclass_classification)
    
    Returns:
        Registration status
    """
    import os
    import ast
    
    # Normalize the file path
    file_path = os.path.normpath(os.path.abspath(file_path))
    
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Sanitize the code - handle escaped characters from JSON serialization
    sanitized_code = code
    
    # Check if the code has JSON-style escaping (e.g., \\n instead of real newlines)
    # This typically happens when LLM output gets double-serialized
    if '\\n' in code and '\n' not in code:
        # Looks like it's been JSON-escaped - unescape it
        import json
        try:
            # Wrap in quotes and parse as JSON string to unescape
            sanitized_code = json.loads(f'"{code}"')
        except json.JSONDecodeError:
            # If that fails, try manual unescaping of common sequences
            sanitized_code = code.replace('\\n', '\n')
            sanitized_code = sanitized_code.replace('\\t', '\t')
            sanitized_code = sanitized_code.replace('\\"', '"')
            sanitized_code = sanitized_code.replace("\\'", "'")
    
    # Additional fix: handle backslash-escaped triple quotes that break f-strings
    # Pattern: f\"\"\" should become f"""
    if '\\"\\"\\"' in sanitized_code:
        sanitized_code = sanitized_code.replace('\\"\\"\\"', '"""')
    
    syntax_warnings = []
    try:
        ast.parse(sanitized_code)
    except SyntaxError as e:
        import logging
        logging.warning(f"Generated code has syntax error: {e}")
        syntax_warnings.append(f"Syntax error at line {e.lineno}: {e.msg}")

    with open(file_path, 'w') as f:
        f.write(sanitized_code)

    result = {
        "status": "registered_with_warnings" if syntax_warnings else "registered",
        "class_name": class_name,
        "file_path": file_path,
        "task_type": task_type,
        "code": code,
    }
    if syntax_warnings:
        result["warnings"] = syntax_warnings
    return result


@langchain_tool
def validate_dataset_timestamps(
    dataset_file_path: str,
    csv_dir: str,
    timedelta_days: int = 0
) -> Dict[str, Any]:
    """
    Validate that dataset timestamps are correctly set.

    Checks:
    1. val_timestamp and test_timestamp exist
    2. Timestamps are real dates within the data range (not Unix epoch)
    3. Gap between val and test is >= timedelta_days (prevents the ValueError
       'timedelta cannot be larger than the difference between val and test timestamps')

    Args:
        dataset_file_path: Path to dataset.py file
        csv_dir: Directory containing CSV files for temporal range check
        timedelta_days: The planned prediction window in days (e.g., 7, 30, 60).
            If provided, the tool verifies that test_timestamp - val_timestamp >= timedelta_days.
            This MUST be provided to prevent runtime ValueError.

    Returns:
        Validation results with status and any issues found
    """
    import os
    import ast
    import pandas as pd

    try:
        # Read the dataset file
        if not os.path.exists(dataset_file_path):
            return {
                "status": "error",
                "error": f"Dataset file not found: {dataset_file_path}"
            }

        with open(dataset_file_path, 'r') as f:
            dataset_code = f.read()

        # Parse to find val_timestamp and test_timestamp
        tree = ast.parse(dataset_code)
        val_ts = None
        test_ts = None

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if target.id == 'val_timestamp':
                            if isinstance(node.value, ast.Call):
                                if len(node.value.args) > 0:
                                    if isinstance(node.value.args[0], ast.Constant):
                                        val_ts = node.value.args[0].value
                        elif target.id == 'test_timestamp':
                            if isinstance(node.value, ast.Call):
                                if len(node.value.args) > 0:
                                    if isinstance(node.value.args[0], ast.Constant):
                                        test_ts = node.value.args[0].value

        if not val_ts or not test_ts:
            return {
                "status": "error",
                "error": "Could not find val_timestamp or test_timestamp in dataset.py"
            }

        # Parse timestamps
        val_timestamp = pd.Timestamp(val_ts)
        test_timestamp = pd.Timestamp(test_ts)

        issues = []

        # Check if timestamps are suspiciously close to Unix epoch (1970-01-01)
        epoch = pd.Timestamp("1970-01-01")
        if abs((val_timestamp - epoch).days) < 365:
            issues.append(f"val_timestamp ({val_ts}) is suspiciously close to Unix epoch (1970-01-01)")
        if abs((test_timestamp - epoch).days) < 365:
            issues.append(f"test_timestamp ({test_ts}) is suspiciously close to Unix epoch (1970-01-01)")

        # Check gap between val and test
        time_diff = (test_timestamp - val_timestamp).days

        if time_diff < 1:
            issues.append(
                f"Gap between val_timestamp and test_timestamp is only {time_diff} days. "
                f"Timestamps appear to be the same or inverted."
            )

        # Check gap >= timedelta (the root cause of the ValueError)
        if timedelta_days > 0 and time_diff < timedelta_days:
            issues.append(
                f"CRITICAL: Gap between val_timestamp and test_timestamp ({time_diff} days) "
                f"is smaller than the planned timedelta ({timedelta_days} days). "
                f"This will cause a ValueError at task initialization. "
                f"Fix: choose timestamps so that test_timestamp - val_timestamp >= {timedelta_days} days."
            )

        # Check against actual data range if CSV files available
        data_min_date = None
        data_max_date = None
        if csv_dir and os.path.exists(csv_dir):
            for f in os.listdir(csv_dir):
                if f.endswith('.csv'):
                    try:
                        df = pd.read_csv(os.path.join(csv_dir, f))
                        for col in df.columns:
                            col_lower = col.lower()
                            if col_lower.endswith('id') or col_lower == 'id':
                                continue
                            try:
                                dates = pd.to_datetime(df[col], errors='coerce')
                                if dates.notna().sum() > len(df) * 0.5:
                                    col_min = dates.min()
                                    col_max = dates.max()
                                    if pd.notna(col_min) and col_min.year >= 1900:
                                        if data_min_date is None or col_min < data_min_date:
                                            data_min_date = col_min
                                    if pd.notna(col_max) and col_max.year <= 2100:
                                        if data_max_date is None or col_max > data_max_date:
                                            data_max_date = col_max
                            except Exception:
                                pass
                    except Exception:
                        pass

            if data_min_date and data_max_date:
                if val_timestamp < data_min_date or val_timestamp > data_max_date:
                    issues.append(
                        f"val_timestamp ({val_ts}) is outside the data range "
                        f"({data_min_date.strftime('%Y-%m-%d')} to {data_max_date.strftime('%Y-%m-%d')})"
                    )
                if test_timestamp < data_min_date or test_timestamp > data_max_date:
                    issues.append(
                        f"test_timestamp ({test_ts}) is outside the data range "
                        f"({data_min_date.strftime('%Y-%m-%d')} to {data_max_date.strftime('%Y-%m-%d')})"
                    )

        if issues:
            # Build a recommendation with suggested fix timestamps
            recommendation = (
                "Dataset timestamps must be fixed before creating tasks. "
                "Use fix_dataset_timestamps() to set timestamps where "
                f"test_timestamp - val_timestamp >= {max(timedelta_days, 1)} days "
                "and both timestamps fall within the data range."
            )
            if data_max_date and timedelta_days > 0:
                suggested_test = data_max_date - pd.Timedelta(days=max(timedelta_days, 7))
                suggested_val = suggested_test - pd.Timedelta(days=timedelta_days)
                recommendation += (
                    f" Suggested: val_timestamp='{suggested_val.strftime('%Y-%m-%d')}', "
                    f"test_timestamp='{suggested_test.strftime('%Y-%m-%d')}'."
                )

            return {
                "status": "invalid",
                "val_timestamp": val_ts,
                "test_timestamp": test_ts,
                "time_diff_days": time_diff,
                "timedelta_days": timedelta_days,
                "issues": issues,
                "recommendation": recommendation,
            }

        return {
            "status": "valid",
            "val_timestamp": val_ts,
            "test_timestamp": test_ts,
            "time_diff_days": time_diff,
            "timedelta_days": timedelta_days,
            "message": f"Dataset timestamps are valid (gap={time_diff}d >= timedelta={timedelta_days}d)"
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@langchain_tool
def fix_dataset_timestamps(
    dataset_file_path: str,
    val_timestamp: str,
    test_timestamp: str
) -> Dict[str, Any]:
    """
    Fix the val_timestamp and test_timestamp in an existing dataset.py file.
    
    Use this tool when validate_dataset_timestamps returns invalid status.
    The tool will update the timestamps in the dataset.py file.
    
    Args:
        dataset_file_path: Path to dataset.py file
        val_timestamp: New val_timestamp in YYYY-MM-DD format
        test_timestamp: New test_timestamp in YYYY-MM-DD format (must be after val_timestamp)
    
    Returns:
        Status of the fix operation
    """
    import os
    import re
    import pandas as pd
    
    try:
        # Validate timestamps
        val_ts = pd.Timestamp(val_timestamp)
        test_ts = pd.Timestamp(test_timestamp)
        
        if test_ts <= val_ts:
            return {
                "status": "error",
                "error": "test_timestamp must be after val_timestamp"
            }
        
        # Note: We don't enforce a minimum gap here because different tasks
        # have different timedelta requirements (e.g., 7 days for weekly churn,
        # 30 days for monthly predictions). The gap should be >= task's timedelta.
        
        # Read the dataset file
        if not os.path.exists(dataset_file_path):
            return {
                "status": "error",
                "error": f"Dataset file not found: {dataset_file_path}"
            }
        
        with open(dataset_file_path, 'r') as f:
            content = f.read()
        
        # Replace val_timestamp
        val_pattern = r'val_timestamp\s*=\s*pd\.Timestamp\(["\'][^"\']*["\']\)'
        val_replacement = f'val_timestamp = pd.Timestamp("{val_timestamp}")'
        new_content = re.sub(val_pattern, val_replacement, content)
        
        # Replace test_timestamp
        test_pattern = r'test_timestamp\s*=\s*pd\.Timestamp\(["\'][^"\']*["\']\)'
        test_replacement = f'test_timestamp = pd.Timestamp("{test_timestamp}")'
        new_content = re.sub(test_pattern, test_replacement, new_content)
        
        # Check if replacements were made
        if new_content == content:
            return {
                "status": "error",
                "error": "Could not find val_timestamp or test_timestamp patterns in dataset.py"
            }
        
        # Write the updated content
        with open(dataset_file_path, 'w') as f:
            f.write(new_content)
        
        return {
            "status": "success",
            "message": f"Updated timestamps in {dataset_file_path}",
            "val_timestamp": val_timestamp,
            "test_timestamp": test_timestamp,
            "time_diff_days": (test_ts - val_ts).days
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }
