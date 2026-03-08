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

### 1. Environment Variables

Create a `.env` file or configure `docker-compose.gpu.yml`:

```env
# LLM API Keys (at least one required)
OPENAI_API_KEY=your_key
GOOGLE_API_KEY=your_key
ANTHROPIC_API_KEY=your_key

# Kaggle (required for Kaggle MCP tools)
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key

# Agent model selection (optional, defaults to openai/gpt-4o)
PLEXE_CONVERSATIONAL_MODEL=openai/gpt-4o
PLEXE_EDA_MODEL=gemini/gemini-2.5-flash
PLEXE_DATASET_BUILDER_MODEL=openai/gpt-4o
PLEXE_TASK_BUILDER_MODEL=openai/gpt-4o
PLEXE_GNN_SPECIALIST_MODEL=openai/gpt-4o
PLEXE_OPERATION_MODEL=openai/gpt-4o

# Pipeline settings
PLEXE_AGENT_TEMPERATURE=0.1
PLEXE_MAX_RETRIES=3
PLEXE_VERBOSE=true
```

### 2. Docker (GPU Production)

```bash
docker compose -f docker-compose.gpu.yml up -d
```

This starts:
- **Backend** (port 8100): FastAPI + WebSocket server with GPU support
- **Frontend** (port 3000): React/Vite chat UI
- **MLflow** (port 5000): Experiment tracking
- **PostgreSQL**: MLflow backend store
- **pgAdmin** (port 8080): Database administration

### 3. Docker (CPU Development)

```bash
docker compose -f docker-compose.dev.yml up -d
```

### 4. Local Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
python -m plexe
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
