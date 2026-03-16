import logging
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import psycopg2

# from plexe.agents.feature_generator import FeatureGeneratorAgent

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

router = APIRouter(prefix="/api", tags=["datasets"])
UPLOADS_DIR = Path("./data/uploads")
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_DIR = Path("./data/database")
DATABASE_DIR.mkdir(parents=True, exist_ok=True)


class PostgresConnection(BaseModel):
    """PostgreSQL connection configuration"""

    host: str
    port: int
    database: str
    username: str
    password: str


class CombineDatasetsRequest(BaseModel):
    tables: List[str]
    relationships: List[dict]
    connection: PostgresConnection


@router.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Upload one or more dataset files

    Supported formats: CSV, XLSX, JSON, Parquet
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    uploaded_files = []
    errors = []

    # Allowed file extensions
    allowed_extensions = {".csv", ".xlsx", ".xls", ".json", ".parquet", ".pq"}

    for file in files:
        try:
            # Validate file extension
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in allowed_extensions:
                errors.append(f"{file.filename}: Unsupported file format")
                continue

            # Save file
            file_path = UPLOADS_DIR / file.filename

            # Read and save file content
            contents = await file.read()
            with open(file_path, "wb") as f:
                f.write(contents)

            uploaded_files.append({"filename": file.filename, "size": len(contents), "path": str(file_path)})

        except Exception as e:
            errors.append(f"{file.filename}: {str(e)}")

    if not uploaded_files and errors:
        raise HTTPException(status_code=400, detail="; ".join(errors))

    return {"uploaded": uploaded_files, "errors": errors, "total": len(uploaded_files), "failed": len(errors)}


@router.post("/postgres/test")
async def test_postgres_connection(config: PostgresConnection):
    """
    Test PostgreSQL connection without saving
    """
    try:
        import psycopg2

        # Build connection string
        conn_str = (
            f"dbname={config.database} "
            f"user={config.username} "
            f"password={config.password} "
            f"host={config.host} "
            f"port={config.port}"
        )

        # Test connection
        conn = psycopg2.connect(conn_str)
        conn.close()

        return {
            "success": True,
            "message": "Connection successful",
            "host": config.host,
            "port": config.port,
            "database": config.database,
        }

    except ImportError:
        raise HTTPException(status_code=500, detail="psycopg2 not installed. Install with: pip install psycopg2-binary")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Connection failed: {str(e)}")


@router.post("/postgres/execute")
async def execute_postgres_query(config: PostgresConnection):
    """
    Execute a query on a PostgreSQL database
    TODO: Store in database or secure config file
    """
    try:
        import psycopg2

        # Build connection string
        conn_str = (
            f"dbname={config.database} "
            f"user={config.username} "
            f"password={config.password} "
            f"host={config.host} "
            f"port={config.port}"
        )

        # Test connection first
        conn = psycopg2.connect(conn_str)

        # Fetch tables from the public schema
        tables = []
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """
            )
            for row in cursor.fetchall():
                tables.append(row[0])

        # Fetch relationships from the public schema
        relationships = []
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    tc.table_name, 
                    kcu.column_name, 
                    ccu.table_name AS foreign_table_name,
                    ccu.column_name AS foreign_column_name
                FROM 
                    information_schema.table_constraints AS tc 
                    JOIN information_schema.key_column_usage AS kcu
                      ON tc.constraint_name = kcu.constraint_name
                    JOIN information_schema.constraint_column_usage AS ccu
                      ON ccu.constraint_name = tc.constraint_name
                WHERE constraint_type = 'FOREIGN KEY' AND tc.table_schema = 'public';
            """
            )
            for row in cursor.fetchall():
                relationships.append(
                    {
                        "table_name": row[0],
                        "column_name": row[1],
                        "foreign_table_name": row[2],
                        "foreign_column_name": row[3],
                    }
                )

        conn.close()

        # TODO: Save to secure config file or database
        # For now, just return success and the list of tables
        return {
            "success": True,
            "message": "Query executed successfully",
            "host": config.host,
            "port": config.port,
            "database": config.database,
            "tables": tables,
            "relationships": relationships,
        }

    except ImportError:
        raise HTTPException(status_code=500, detail="psycopg2 not installed. Install with: pip install psycopg2-binary")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to execute query: {str(e)}")


@router.get("/datasets")
async def get_datasets():
    """
    Get list of uploaded datasets
    """
    try:
        datasets = []

        if UPLOADS_DIR.exists():
            for file_path in UPLOADS_DIR.iterdir():
                if file_path.is_file():
                    datasets.append(
                        {
                            "id": file_path.stem,
                            "filename": file_path.name,
                            "size": file_path.stat().st_size,
                            "created_at": file_path.stat().st_ctime,
                            "path": str(file_path),
                        }
                    )

        return {"datasets": datasets, "total": len(datasets)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get datasets: {str(e)}")


@router.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """
    Delete a dataset by ID (filename without extension)
    """
    try:
        # Find and delete the file
        for file_path in UPLOADS_DIR.iterdir():
            if file_path.stem == dataset_id and file_path.is_file():
                file_path.unlink()
                return {"success": True, "message": f"Dataset '{dataset_id}' deleted successfully"}

        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete dataset: {str(e)}")


@router.get("/datasets/{dataset_id}/download")
async def download_dataset(dataset_id: str):
    """
    Download a dataset by ID (filename without extension)
    """
    try:
        for file_path in UPLOADS_DIR.iterdir():
            if file_path.stem == dataset_id and file_path.is_file():
                return FileResponse(path=file_path, filename=file_path.name, media_type="application/octet-stream")
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download dataset: {str(e)}")


@router.post("/datasets/combine")
async def combine_datasets_endpoint(data: CombineDatasetsRequest):
    """
    Combine datasets using featuretools
    """
    logger.info(f"Received request to combine datasets with data: {data}")
    try:
        # connect to postgres and export data to csv
        conn_str = (
            f"dbname={data.connection.database} "
            f"user={data.connection.username} "
            f"password={data.connection.password} "
            f"host={data.connection.host} "
            f"port={data.connection.port}"
        )
        conn = psycopg2.connect(conn_str)

        db_dir = DATABASE_DIR / data.connection.database
        db_dir.mkdir(parents=True, exist_ok=True)

        table_paths = {}
        for table_name in data.tables:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            file_path = db_dir / f"{table_name}.csv"
            df.to_csv(file_path, index=False, date_format="%Y-%m-%d %H:%M:%S")
            table_paths[table_name] = str(file_path.resolve())

        conn.close()


        message = "Feature generation is being migrated to a new multi-agent system."

        logger.info(message)
        return {"success": True, "message": message}

    except Exception as e:
        logger.error(f"Failed to combine datasets: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to combine datasets: {str(e)}")
