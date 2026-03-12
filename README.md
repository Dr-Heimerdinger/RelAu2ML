# Plexe: Agentic ML Framework with MCP Integration

Plexe is a multi-agent framework built on **LangGraph** that automates the entire machine learning pipeline from natural language descriptions. It transforms user requests into trained Graph Neural Network (GNN) models through a coordinated sequence of specialized agents.

The system integrates **Model Context Protocol (MCP)** to connect with external academic and data tools (Google Scholar, Kaggle, arXiv, Semantic Scholar) for hyperparameter optimization and research.

## Architecture

### Agent Pipeline

```
User Request
    |
    v
[ConversationalAgent] -- Gathers requirements, validates data sources
    |
    v
[EDAAgent] -- Schema analysis, data export, exploratory data analysis
    |
    v
[DatasetBuilderAgent] -- Generates RelBench Dataset class (dataset.py)
    |
    v
[TaskBuilderAgent] -- Generates RelBench Task class with SQL (task.py)
    |
    v
[GNNSpecialistAgent] -- Generates training script with HPO (train_script.py)
    |
    v
[OperationAgent] -- Executes training, reports metrics (best_model.pt)
```

Each agent inherits from `BaseAgent` and is orchestrated by `PlexeOrchestrator` using a LangGraph `StateGraph` with conditional edges and checkpoint-based state persistence.

### Error Handling

The pipeline uses a structured error handling system with:

- **Categorized errors**: Errors are classified as `transient` (retry with backoff), `permanent` (escalate), or `recoverable` (auto-fix).
- **Targeted retry routing**: On failure, the error handler routes directly back to the failed agent rather than restarting the entire pipeline.
- **Per-agent retry limits**: Each pipeline phase has independent retry budgets (e.g., `schema_analysis: 3`, `gnn_training: 1`).
- **Circuit breaker**: MCP tool calls use a circuit breaker that opens after 3 consecutive failures and auto-resets after 5 minutes.
- **Clearable error state**: Active errors are cleared between retry cycles to prevent stale errors from blocking retries.

### MCP Integration

External knowledge sources are accessed via MCP servers configured in `mcp_config.json`:

| Server | Purpose |
|--------|---------|
| `hpo-search` | Heuristic hyperparameter optimization |
| `google-scholar` | Academic paper search |
| `kaggle` | Dataset and competition search |
| `semantic-scholar` | Scholarly article metadata |
| `arxiv` | Preprint search |

MCP tools are automatically loaded by `MCPManager` and made available to all agents.

## Key Directories

```
plexe/
  langgraph/
    agents/          # Agent implementations (base, conversational, eda, etc.)
    prompts/         # System prompts for each agent
    tools/           # Tool functions (SQL testing, code registration, etc.)
    utils/           # Emitters, logging, callbacks
    orchestrator.py  # LangGraph StateGraph orchestration
    state.py         # Pipeline state definition
    config.py        # Agent and model configuration
    mcp_manager.py   # MCP server lifecycle and tool conversion
  relbench/          # RelBench base classes and reference tasks
    base/            # Database, Table, Dataset, Task base classes
    tasks/           # Reference task implementations (F1, H&M, Amazon, etc.)
    modeling/        # GNN model architecture, graph construction, training
  api/               # FastAPI routes (datasets, models, inference)
  server.py          # WebSocket server and session management
  main.py            # Application entry point
```

## Setup

### Yêu cầu hệ thống

- Python 3.11 hoặc 3.12
- [uv](https://docs.astral.sh/uv/) (trình quản lý package và venv)
- Docker + Docker Compose (cho triển khai container)
- NVIDIA GPU + CUDA 12.8 (cho `docker-compose.gpu.yml`)

Cài uv nếu chưa có:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 1. Cấu hình biến môi trường

Sao chép file mẫu và điền API keys:
```bash
cp .env.example .env
```

Các biến bắt buộc trong `.env`:
```env
# Ít nhất một LLM API key
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
ANTHROPIC_API_KEY=...

# Kaggle (cần thiết cho MCP tools tìm dataset)
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key

# Model cho từng agent (mặc định gemini-2.0-flash)
PLEXE_ORCHESTRATOR_MODEL=gemini/gemini-2.0-flash-exp
PLEXE_CONVERSATIONAL_MODEL=gemini/gemini-2.0-flash-exp
PLEXE_EDA_MODEL=gemini/gemini-2.0-flash-exp
PLEXE_GNN_SPECIALIST_MODEL=openai/gpt-4o

# Pipeline settings
PLEXE_AGENT_TEMPERATURE=0.1
PLEXE_VERBOSE=true
```

### 2. Cài đặt local (không Docker)

```bash
# Clone repo
git clone <repo-url>
cd plexe

# Cài đặt toàn bộ dependencies (uv tự tạo .venv)
uv sync --extra all

# Hoặc chỉ cài bản tối giản (không có transformers/chatui)
uv sync

# Kích hoạt venv
source .venv/bin/activate

# Chạy server
python -m plexe
```

> **Tip:** Sau lần đầu, chạy `uv lock` để tạo `uv.lock` rồi commit vào repo.
> Khi clone ở máy khác, `uv sync` sẽ cài đúng phiên bản đã lock, đảm bảo reproducibility.

Tùy chọn extras:
| Lệnh | Bao gồm |
|------|---------|
| `uv sync` | Chỉ deps cơ bản |
| `uv sync --extra chatui` | + FastAPI, uvicorn, websockets |
| `uv sync --extra transformers` | + HuggingFace transformers, sentence-transformers |
| `uv sync --extra all` | Tất cả các trên |
| `uv sync --group dev` | + pytest, ruff, jupyterlab |

### 3. Docker — GPU (Production)

Yêu cầu: NVIDIA GPU, [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

```bash
# Build và start toàn bộ stack
docker compose -f docker-compose.gpu.yml up -d

# Xem logs
docker compose -f docker-compose.gpu.yml logs -f backend
```

Services khởi động:
| Service | Port | Mô tả |
|---------|------|--------|
| backend | 8100 | FastAPI + WebSocket (GPU) |
| frontend | 3000 | React/Vite chat UI |
| mlflow | 5000 | Experiment tracking |
| postgres | 5432 | MLflow backend store |
| pgadmin | 8080 | Quản lý database |

### 4. Docker — CPU (Development)

```bash
docker compose -f docker-compose.dev.yml up -d
```

Backend chạy ở port **8000** (thay vì 8100), dùng CPU torch để build nhanh hơn.

### 5. Chuẩn bị khi cài đặt tại máy mới

```bash
# 1. Clone repo
git clone <repo-url> && cd plexe

# 2. Tạo .env từ mẫu
cp .env.example .env
# Điền API keys vào .env

# 3a. Cài local (nếu không dùng Docker)
uv sync --extra all          # uv.lock đảm bảo cài đúng phiên bản
source .venv/bin/activate
python -m plexe

# 3b. Chạy Docker GPU
docker compose -f docker-compose.gpu.yml up -d

# 3c. Chạy Docker CPU (dev)
docker compose -f docker-compose.dev.yml up -d
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve frontend |
| `/ws` | WebSocket | Real-time chat communication |
| `/health` | GET | Health check |
| `/api/upload` | POST | Upload CSV/XLSX/JSON/Parquet files |
| `/api/postgres/test` | POST | Test PostgreSQL connection |
| `/api/models/` | GET | List trained models |
| `/api/models/{id}/infer` | POST | Run inference on a trained model |
| `/api/models/{id}/download` | GET | Download trained model |

## How It Works

1. **Describe your task** in natural language (e.g., "Predict whether F1 drivers will finish in the top 3")
2. **Upload data** as CSV files or provide a PostgreSQL connection string
3. The pipeline automatically:
   - Analyzes schema, relationships, and temporal patterns
   - Generates a `Dataset` class that loads and links your tables
   - Generates a `Task` class with SQL that defines the prediction target
   - Searches academic literature and Kaggle for optimal hyperparameters
   - Trains a Graph Neural Network and reports metrics
4. **Download** the trained model or run inference via the API

## Technology Stack

- **Orchestration**: LangGraph, LangChain
- **LLM Providers**: OpenAI, Anthropic, Google Gemini (via LiteLLM)
- **ML**: PyTorch, PyTorch Geometric, scikit-learn
- **Data**: DuckDB (SQL execution), pandas, pyarrow
- **Serving**: FastAPI, uvicorn, WebSockets
- **Tracking**: MLflow
- **External Tools**: MCP (Model Context Protocol)

## License

Apache 2.0
