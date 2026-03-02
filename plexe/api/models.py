import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/models", tags=["models"])

WORKDIR = Path(os.environ.get("PLEXE_WORKDIR", "./workdir"))


class RenameRequest(BaseModel):
    new_name: str


class InferRequest(BaseModel):
    timestamp: str
    entity_ids: Optional[List[str]] = None


def _scan_models() -> list:
    if not WORKDIR.exists():
        return []
    models = []
    for folder in sorted(WORKDIR.iterdir()):
        if not folder.is_dir():
            continue
        model_path = folder / "best_model.pt"
        if not model_path.exists():
            continue
        info = _read_model_info(folder)
        if info:
            models.append(info)
    return models


def _read_model_info(folder: Path) -> Optional[dict]:
    model_path = folder / "best_model.pt"
    if not model_path.exists():
        return None

    info = {
        "id": folder.name,
        "name": folder.name,
        "path": str(folder),
        "model_size": model_path.stat().st_size,
        "created_at": model_path.stat().st_mtime,
    }

    results_path = folder / "training_results.json"
    if results_path.exists():
        try:
            with open(results_path) as f:
                results = json.load(f)
            info["training_results"] = results
            info["tune_metric"] = results.get("tune_metric", "")
            info["epochs_trained"] = results.get("epochs_trained", 0)
            info["val_metrics"] = results.get("val_metrics", {})
            info["test_metrics"] = results.get("test_metrics", {})
        except Exception as e:
            logger.warning(f"Failed to read training results for {folder.name}: {e}")

    task_path = folder / "task.py"
    if task_path.exists():
        try:
            task_meta = _parse_task_file(task_path)
            info["task_meta"] = task_meta
        except Exception as e:
            logger.warning(f"Failed to parse task.py for {folder.name}: {e}")

    dataset_path = folder / "dataset.py"
    if dataset_path.exists():
        info["has_dataset"] = True

    csv_dir = folder / "csv_files"
    if csv_dir.exists():
        info["csv_files"] = [f.name for f in csv_dir.iterdir() if f.suffix == ".csv"]

    return info


def _parse_task_file(task_path: Path) -> dict:
    import re

    content = task_path.read_text()
    meta = {}

    match = re.search(r'task_type\s*=\s*TaskType\.(\w+)', content)
    if match:
        meta["task_type"] = match.group(1).lower()

    match = re.search(r'entity_col\s*=\s*["\'](\w+)["\']', content)
    if match:
        meta["entity_col"] = match.group(1)

    match = re.search(r'entity_table\s*=\s*["\'](\w+)["\']', content)
    if match:
        meta["entity_table"] = match.group(1)

    match = re.search(r'time_col\s*=\s*["\'](\w+)["\']', content)
    if match:
        meta["time_col"] = match.group(1)

    match = re.search(r'target_col\s*=\s*["\'](\w+)["\']', content)
    if match:
        meta["target_col"] = match.group(1)

    match = re.search(r'timedelta\s*=\s*pd\.Timedelta\(days=(\d+)\)', content)
    if match:
        meta["timedelta_days"] = int(match.group(1))

    metrics = re.findall(r'from plexe\.relbench\.metrics import (.+)', content)
    if metrics:
        meta["metrics"] = [m.strip() for m in metrics[0].split(",")]

    return meta


def _get_input_schema(folder: Path) -> dict:
    task_meta = {}
    task_path = folder / "task.py"
    if task_path.exists():
        task_meta = _parse_task_file(task_path)

    csv_dir = folder / "csv_files"
    tables = {}
    if csv_dir.exists():
        for csv_file in csv_dir.iterdir():
            if csv_file.suffix == ".csv":
                try:
                    df = pd.read_csv(csv_file, nrows=3)
                    tables[csv_file.stem] = {
                        "columns": list(df.columns),
                        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                        "sample_rows": df.head(3).to_dict(orient="records"),
                    }
                except Exception:
                    tables[csv_file.stem] = {"columns": [], "dtypes": {}, "sample_rows": []}

    entity_table = task_meta.get("entity_table", "")
    entity_col = task_meta.get("entity_col", "")

    return {
        "task_meta": task_meta,
        "tables": tables,
        "entity_table": entity_table,
        "entity_col": entity_col,
        "description": (
            f"This model predicts on the '{entity_table}' table using entity column "
            f"'{entity_col}'. Provide a timestamp and optionally a list of entity IDs "
            f"to run inference."
        ),
    }


@router.get("")
async def list_models():
    models = _scan_models()
    return {"models": models, "total": len(models)}


@router.get("/{model_id}")
async def get_model(model_id: str):
    folder = WORKDIR / model_id
    if not folder.exists() or not (folder / "best_model.pt").exists():
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    info = _read_model_info(folder)
    if not info:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    return info


@router.delete("/{model_id}")
async def delete_model(model_id: str):
    folder = WORKDIR / model_id
    if not folder.exists():
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    try:
        shutil.rmtree(folder)
        return {"success": True, "message": f"Model '{model_id}' deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {e}")


@router.put("/{model_id}/rename")
async def rename_model(model_id: str, req: RenameRequest):
    folder = WORKDIR / model_id
    if not folder.exists():
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    new_name = req.new_name.strip()
    if not new_name:
        raise HTTPException(status_code=400, detail="New name cannot be empty")
    new_folder = WORKDIR / new_name
    if new_folder.exists():
        raise HTTPException(status_code=400, detail=f"Name '{new_name}' already exists")
    try:
        folder.rename(new_folder)
        return {"success": True, "old_name": model_id, "new_name": new_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to rename: {e}")


@router.get("/{model_id}/schema")
async def get_model_schema(model_id: str):
    folder = WORKDIR / model_id
    if not folder.exists() or not (folder / "best_model.pt").exists():
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    return _get_input_schema(folder)


@router.post("/{model_id}/infer")
async def run_inference(model_id: str, file: UploadFile = File(...)):
    folder = WORKDIR / model_id
    if not folder.exists() or not (folder / "best_model.pt").exists():
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    task_path = folder / "task.py"
    dataset_path = folder / "dataset.py"
    model_path = folder / "best_model.pt"
    csv_dir = folder / "csv_files"

    if not task_path.exists() or not dataset_path.exists():
        raise HTTPException(
            status_code=400,
            detail="Model is missing task.py or dataset.py required for inference",
        )

    task_meta = _parse_task_file(task_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        upload_path = Path(tmpdir) / file.filename
        contents = await file.read()
        with open(upload_path, "wb") as f:
            f.write(contents)

        try:
            input_df = pd.read_csv(upload_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")

        entity_col = task_meta.get("entity_col", "id")
        time_col = task_meta.get("time_col", "timestamp")

        if entity_col not in input_df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Input CSV must contain entity column '{entity_col}'. Found columns: {list(input_df.columns)}",
            )
        if time_col not in input_df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Input CSV must contain time column '{time_col}'. Found columns: {list(input_df.columns)}",
            )

        infer_script = _generate_inference_script(
            folder=str(folder),
            csv_dir=str(csv_dir),
            model_path=str(model_path),
            input_csv=str(upload_path),
            output_csv=os.path.join(tmpdir, "predictions.csv"),
            task_meta=task_meta,
        )

        script_path = os.path.join(tmpdir, "run_inference.py")
        with open(script_path, "w") as f:
            f.write(infer_script)

        project_root = str(Path(__file__).parent.parent.parent.resolve())
        env = {**os.environ}
        existing_pypath = env.get("PYTHONPATH", "")
        paths = [str(folder), project_root]
        if existing_pypath:
            paths.append(existing_pypath)
        env["PYTHONPATH"] = os.pathsep.join(paths)

        try:
            result = subprocess.run(
                [sys.executable, script_path],
                cwd=str(folder),
                env=env,
                capture_output=True,
                text=True,
                timeout=600,
            )
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=504, detail="Inference timed out (600s)")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference execution failed: {e}")

        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"Inference script failed:\n{result.stderr[-2000:]}",
            )

        output_csv = os.path.join(tmpdir, "predictions.csv")
        if not os.path.exists(output_csv):
            raise HTTPException(
                status_code=500,
                detail=f"Inference did not produce output. stdout:\n{result.stdout[-1000:]}",
            )

        try:
            result_df = pd.read_csv(output_csv)
            records = result_df.to_dict(orient="records")
            columns = list(result_df.columns)
            return {
                "success": True,
                "predictions": records,
                "columns": columns,
                "total": len(records),
                "task_type": task_meta.get("task_type", "unknown"),
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read predictions: {e}")


def _generate_inference_script(
    folder: str,
    csv_dir: str,
    model_path: str,
    input_csv: str,
    output_csv: str,
    task_meta: dict,
) -> str:
    entity_col = task_meta.get("entity_col", "id")
    time_col = task_meta.get("time_col", "timestamp")
    entity_table = task_meta.get("entity_table", "users")
    task_type = task_meta.get("task_type", "regression")
    hidden_channels = 128
    num_gnn_layers = 2
    out_channels = 1

    return f'''import os
import sys
import json
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "{folder}")

from dataset import GenDataset
from task import GenTask
from plexe.relbench.modeling.graph import make_pkey_fkey_graph, get_node_train_table_input
from plexe.relbench.modeling.nn import HeteroEncoder, HeteroTemporalEncoder, HeteroGraphSAGE
from plexe.relbench.modeling.utils import get_stype_proposal
from plexe.relbench.base import Table
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import MLP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

text_embedder_cfg = None
try:
    from typing import List, Optional as Opt
    from sentence_transformers import SentenceTransformer
    from torch import Tensor
    from torch_frame.config.text_embedder import TextEmbedderConfig

    class GloveTextEmbedding:
        def __init__(self, device_: Opt[torch.device] = None):
            self.model = SentenceTransformer(
                "sentence-transformers/average_word_embeddings_glove.6B.300d",
                device=device_,
            )
        def __call__(self, sentences: List[str]) -> Tensor:
            return torch.from_numpy(self.model.encode(sentences))

    text_embedder_cfg = TextEmbedderConfig(
        text_embedder=GloveTextEmbedding(device_=device), batch_size=256
    )
except Exception:
    pass

csv_dir = "{csv_dir}"
dataset = GenDataset(csv_dir=csv_dir)
task = GenTask(dataset)
db = dataset.get_db()

col_to_stype_dict = get_stype_proposal(db)
data, col_stats_dict = make_pkey_fkey_graph(
    db,
    col_to_stype_dict=col_to_stype_dict,
    text_embedder_cfg=text_embedder_cfg,
    cache_dir="{folder}/cache/",
)

entity_table = "{entity_table}"

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
                node_type for node_type in data.node_types if "time" in data[node_type]
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
        self.head = MLP(
            channels,
            out_channels=out_channels,
            norm="batch_norm",
            num_layers=1,
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        self.gnn.reset_parameters()
        self.head.reset_parameters()

    def forward(self, batch, entity_table):
        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)
        rel_time_dict = self.temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict
        )
        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time
        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
            batch.num_sampled_nodes_dict,
            batch.num_sampled_edges_dict,
        )
        return self.head(x_dict[entity_table][:seed_time.size(0)])

model = Model(
    data=data,
    col_stats_dict=col_stats_dict,
    num_layers={num_gnn_layers},
    channels={hidden_channels},
    out_channels={out_channels},
).to(device)

state_dict = torch.load("{model_path}", map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model.eval()

input_df = pd.read_csv("{input_csv}")
input_df["{time_col}"] = pd.to_datetime(input_df["{time_col}"])
input_df["{entity_col}"] = pd.to_numeric(input_df["{entity_col}"], errors="coerce")

timestamps = input_df["{time_col}"]
entity_ids = input_df["{entity_col}"]

unique_timestamps = timestamps.unique()
all_preds = []
all_entity_ids = []
all_timestamps = []

for ts in unique_timestamps:
    ts_mask = timestamps == ts
    ts_entities = entity_ids[ts_mask].values
    ts_series = pd.Series([ts] * len(ts_entities))

    infer_table = task.make_table(db, ts_series)
    infer_df = infer_table.df

    target_col = "{task_meta.get('target_col', 'target')}"
    if target_col in infer_df.columns:
        infer_df = infer_df.drop(columns=[target_col])
    infer_df[target_col] = 0.0

    fkey_map = {{"{entity_col}": "{entity_table}"}}
    infer_table_obj = Table(
        df=infer_df,
        fkey_col_to_pkey_table=fkey_map,
        pkey_col=None,
        time_col="{time_col}",
    )

    table_input = get_node_train_table_input(table=infer_table_obj, task=task)
    loader = NeighborLoader(
        data,
        num_neighbors=[128] * {num_gnn_layers},
        time_attr="time",
        input_nodes=table_input.nodes,
        input_time=table_input.time,
        transform=table_input.transform,
        batch_size=512,
        temporal_strategy="uniform",
        shuffle=False,
        num_workers=0,
        persistent_workers=False,
    )

    pred_list = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch, entity_table)
            pred = pred.view(-1) if pred.size(1) == 1 else pred
            pred_list.append(pred.detach().cpu())

    preds = torch.cat(pred_list, dim=0).numpy()

    task_type = "{task_type}"
    if task_type == "binary_classification":
        proba = 1.0 / (1.0 + np.exp(-preds))
        labels = (proba >= 0.5).astype(int)
        for i, eid in enumerate(infer_df["{entity_col}"].values[:len(preds)]):
            all_preds.append({{"entity_id": eid, "timestamp": str(ts), "probability": float(proba[i]), "prediction": int(labels[i])}})
    elif task_type == "multiclass_classification":
        class_preds = np.argmax(preds, axis=1) if preds.ndim > 1 else preds
        for i, eid in enumerate(infer_df["{entity_col}"].values[:len(preds)]):
            all_preds.append({{"entity_id": eid, "timestamp": str(ts), "prediction": int(class_preds[i])}})
    else:
        for i, eid in enumerate(infer_df["{entity_col}"].values[:len(preds)]):
            all_preds.append({{"entity_id": eid, "timestamp": str(ts), "prediction": float(preds[i])}})

result_df = pd.DataFrame(all_preds)
result_df.to_csv("{output_csv}", index=False)
print(f"Inference complete: {{len(result_df)}} predictions saved")
'''
