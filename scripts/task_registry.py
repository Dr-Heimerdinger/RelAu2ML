"""
Shared dataset-task mapping for author table scripts.

Since register_task/register_dataset are defined but never called (registries are
empty), this module provides a hardcoded mapping from dataset/task names to their
classes.
"""

import os
import sys
import types

# Bypass plexe/__init__.py which imports langgraph (heavy/optional dependency).
# We only need plexe.relbench.*, so we register a lightweight plexe stub if
# the real one can't be imported.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

if "plexe" not in sys.modules:
    try:
        import plexe  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        _stub = types.ModuleType("plexe")
        _stub.__path__ = [os.path.join(_PROJECT_ROOT, "plexe")]
        sys.modules["plexe"] = _stub

from plexe.relbench.datasets.f1 import F1Dataset
from plexe.relbench.datasets.amazon import AmazonDataset
from plexe.relbench.datasets.hm import HMDataset
from plexe.relbench.datasets.stack import StackDataset
from plexe.relbench.datasets.trial import TrialDataset
from plexe.relbench.datasets.event import EventDataset
from plexe.relbench.datasets.avito import AvitoDataset

from plexe.relbench.tasks.f1 import (
    DriverPositionTask,
    DriverDNFTask,
    DriverTop3Task,
    DriverRaceCompeteTask,
)
from plexe.relbench.tasks.amazon import (
    UserChurnTask as AmazonUserChurnTask,
    UserLTVTask as AmazonUserLTVTask,
    ItemChurnTask,
    ItemLTVTask,
    UserItemPurchaseTask as AmazonUserItemPurchaseTask,
    UserItemRateTask,
    UserItemReviewTask,
)
from plexe.relbench.tasks.hm import (
    UserItemPurchaseTask as HMUserItemPurchaseTask,
    UserChurnTask as HMUserChurnTask,
    ItemSalesTask,
)
from plexe.relbench.tasks.stack import (
    UserEngagementTask,
    PostVotesTask,
    UserBadgeTask,
    UserPostCommentTask,
    PostPostRelatedTask,
)
from plexe.relbench.tasks.trial import (
    StudyOutcomeTask,
    StudyAdverseTask,
    SiteSuccessTask,
    ConditionSponsorRunTask,
    SiteSponsorRunTask,
)
from plexe.relbench.tasks.event import (
    UserAttendanceTask,
    UserRepeatTask,
    UserIgnoreTask,
)
from plexe.relbench.tasks.avito import (
    AdCTRTask,
    UserVisitsTask,
    UserClicksTask,
    UserAdVisitTask,
)

DATASET_TASK_MAP = {
    "rel-f1": {
        "DatasetClass": F1Dataset,
        "tasks": {
            "driver-position": DriverPositionTask,
            "driver-dnf": DriverDNFTask,
            "driver-top3": DriverTop3Task,
            "driver-race-compete": DriverRaceCompeteTask,
        },
    },
    "rel-amazon": {
        "DatasetClass": AmazonDataset,
        "tasks": {
            "user-churn": AmazonUserChurnTask,
            "user-ltv": AmazonUserLTVTask,
            "item-churn": ItemChurnTask,
            "item-ltv": ItemLTVTask,
            "user-item-purchase": AmazonUserItemPurchaseTask,
            "user-item-rate": UserItemRateTask,
            "user-item-review": UserItemReviewTask,
        },
    },
    "rel-hm": {
        "DatasetClass": HMDataset,
        "tasks": {
            "user-item-purchase": HMUserItemPurchaseTask,
            "user-churn": HMUserChurnTask,
            "item-sales": ItemSalesTask,
        },
    },
    "rel-stack": {
        "DatasetClass": StackDataset,
        "tasks": {
            "user-engagement": UserEngagementTask,
            "post-votes": PostVotesTask,
            "user-badge": UserBadgeTask,
            "user-post-comment": UserPostCommentTask,
            "post-post-related": PostPostRelatedTask,
        },
    },
    "rel-trial": {
        "DatasetClass": TrialDataset,
        "tasks": {
            "study-outcome": StudyOutcomeTask,
            "study-adverse": StudyAdverseTask,
            "site-success": SiteSuccessTask,
            "condition-sponsor-run": ConditionSponsorRunTask,
            "site-sponsor-run": SiteSponsorRunTask,
        },
    },
    "rel-event": {
        "DatasetClass": EventDataset,
        "tasks": {
            "user-attendance": UserAttendanceTask,
            "user-repeat": UserRepeatTask,
            "user-ignore": UserIgnoreTask,
        },
    },
    "rel-avito": {
        "DatasetClass": AvitoDataset,
        "tasks": {
            "ad-ctr": AdCTRTask,
            "user-visits": UserVisitsTask,
            "user-clicks": UserClicksTask,
            "user-ad-visit": UserAdVisitTask,
        },
    },
}


def get_dataset_task_pairs(dataset_name=None, task_name=None):
    """Yield (dataset_name, task_name, DatasetClass, TaskClass) tuples
    filtered by optional dataset/task name."""
    items = DATASET_TASK_MAP.items()
    if dataset_name:
        items = [(k, v) for k, v in items if k == dataset_name]
        if not items:
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. "
                f"Available: {list(DATASET_TASK_MAP.keys())}"
            )

    for ds_name, ds_info in items:
        tasks = ds_info["tasks"].items()
        if task_name:
            tasks = [(k, v) for k, v in tasks if k == task_name]
            if not tasks and dataset_name:
                raise ValueError(
                    f"Unknown task '{task_name}' for dataset '{ds_name}'. "
                    f"Available: {list(ds_info['tasks'].keys())}"
                )
        for t_name, t_class in tasks:
            yield ds_name, t_name, ds_info["DatasetClass"], t_class
