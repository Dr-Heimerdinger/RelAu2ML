import os, sys
import numpy as np
import pandas as pd
from typing import Optional
import duckdb
from plexe.relbench.base import Database, Table, EntityTask, TaskType, Dataset
from plexe.relbench.metrics import accuracy, f1, roc_auc, average_precision

from workdir.rel_f1_driver_dnf.dataset import GenDataset
from workdir.rel_f1_driver_dnf.task import GenTask

from plexe.relbench.tasks.f1 import DriverDNFTask

csv_dir = '/home/ta/kl/plexe-clone/workdir/rel_f1_driver_dnf/csv_files'
dataset = GenDataset(csv_dir=csv_dir)
gen_task = GenTask(dataset)
root_task = DriverDNFTask(dataset)

db = dataset.get_db()
print("Training Table from GenTask:", gen_task.get_table("train"))
print("Training Table from DriverDNFTask:", root_task.get_table("train"))