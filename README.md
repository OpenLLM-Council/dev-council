# Dev Council 🎯

> **A Multi-Agent LLM Framework for Collaborative Software Development**

An innovative open-source framework that implements a council of AI agents working together to analyze, plan, and design software projects. Through structured debate and consensus-based decision-making, the agents produce comprehensive software specifications, milestone plans, and system architecture diagrams—all before a single line of production code is written.

[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)
[![Node.js](https://img.shields.io/badge/node-18+-green)](https://nodejs.org/)

## Table of Contents

- [Features](#features)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Contributing](#contributing)
- [License](#license)

## Features

✨ **Multi-Agent Collaboration** - Multiple specialized LLM agents work together to analyze requirements and design solutions

📋 **Automated SRS Generation** - Creates IEEE 830-compliant Software Requirements Specification documents

🎯 **Milestone Planning** - Automatically generates project milestones with task breakdowns and LLM assignments

🔄 **System Architecture Diagrams** - Generates Mermaid-based flow diagrams for system design visualization

🤖 **Local LLM Support** - Runs on local Ollama instances (Qwen, DeepSeek, Mistral, etc.) - no API keys required

📦 **End-to-End Workflow** - From user request to comprehensive project documentation in minutes

## How It Works

Dev Council's workflow demonstrates how multiple AI agents can collaborate to solve complex planning problems:

```
User Request
    ↓
┌─────────────────────────────────────────┐
│   Manager Agent (Orchestrator)          │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ Project Lead Agent                      │
│ └─ Analyzes requirements                │
│ └─ Creates SRS document                 │
│ └─ Breaks down into subtasks            │
│ └─ Assigns each to specialized LLMs     │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ Milestone Agent                         │
│ └─ Extracts milestones from SRS         │
│ └─ Creates planning timeline            │
│ └─ Assigns LLMs to milestones           │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ Flow Diagram Agent                      │
│ └─ Visualizes system architecture       │
│ └─ Generates Mermaid diagrams           │
└─────────────────────────────────────────┘
    ↓
Comprehensive Project Documentation
```

## Project Structure

```
dev-council/
├── backend/                          # Python backend with LLM agents
│   ├── app/
│   │   ├── agents/                   # Agent implementations
│   │   │   ├── manager.py            # Orchestration agent
│   │   │   ├── project_lead.py       # SRS generation
│   │   │   ├── milestone.py          # Milestone planning
│   │   │   ├── flow_diagram.py       # Architecture diagrams
│   │   │   └── manager.py
│   │   ├── core/
│   │   │   └── config.py             # Configuration settings
│   │   ├── structured_outputs/       # Output schemas
│   │   ├── tools/                    # Utility tools
│   │   │   ├── llm_resources.py      # LLM discovery
│   │   │   ├── mermaid.py            # Diagram generation
│   │   │   └── save_file.py          # File operations
│   │   └── main.py                   # Entry point
│   ├── outputs/                      # Generated documentation
│   ├── requirements.txt
│   └── pyproject.toml
│
├── frontend/                         # Next.js web interface
│   ├── app/                          # Next.js app directory
│   ├── package.json
│   └── tsconfig.json
│
└── README.md

```

## Tech Stack

### Backend
- **Framework**: LangChain, LangGraph
- **LLM Engines**: Ollama (local inference), supports Qwen, DeepSeek, Mistral
- **Language**: Python 3.10+
- **Key Libraries**:
  - `langchain` - Agent creation and orchestration
  - `langgraph` - Agent workflow management
  - `mermaidian` - Diagram generation
  - `markdown-pdf` - Document conversion

### Frontend
- **Framework**: Next.js 16
- **UI**: React 19, TypeScript
- **Styling**: Tailwind CSS 4
- **Tooling**: ESLint, PostCSS

## Installation

### Prerequisites
- Python 3.10 or higher
- Node.js 18 or higher
- Ollama installed and running (for LLM inference)

### Backend Setup

```bash
cd backend

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file with your configuration
cp .env.example .env  # or manually create .env
```

**Configure `.env`** (backend/.env):
```env
GPT_LLM=qwen2.5:1.5b
QWEN_LLM=qwen2.5:1.5b
DEEPSEEK_LLM=deepseek-r1:14b
MISTRAL_LLM=mistral-small:24b
OLLAMA_URL=http://localhost:11434
OLLAMA_TEMPERATURE=0
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Build frontend (optional)
npm run build
```

## Quick Start

### 1. Start Ollama
Make sure Ollama is running and loaded with the models specified in your `.env` file:

```bash
ollama serve
```

### 2. Run the Backend

```bash
cd backend
python main.py
```

Enter your project request when prompted:
```
User request: Create a real-time collaborative document editing application with user authentication, conflict resolution, and offline support
```

The system will:
1. Analyze the request and generate an SRS document
2. Break down requirements into milestones
3. Create system architecture diagrams
4. Save all outputs to `outputs/` directory as `.md`, `.pdf` files

### 3. View Generated Outputs

Check the `backend/outputs/` directory for:
- `project_plan.md` / `project_plan.pdf` - Complete SRS document
- `milestone.md` / `milestone.pdf` - Milestone planning table
- Flow diagrams (Mermaid format)

### 4. Run the Frontend

```bash
cd frontend
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to view the web interface.

## Architecture

### Agent-Based Design Pattern

Each agent in the council specializes in a specific aspect of software planning:

| Agent | Role | Responsibility |
|-------|------|-----------------|
| **Manager** | Orchestrator | Coordinates workflow between specialized agents |
| **Project Lead** | Analyst & Planner | Creates IEEE 830-compliant SRS documents |
| **Milestone Agent** | Timeline Planner | Breaks down work into logical milestones |
| **Flow Diagram Agent** | Architect | Generates system architecture visualizations |

### Key Design Principles

1. **Separation of Concerns** - Each agent focuses on its domain
2. **Local-First** - Uses Ollama for private, local LLM inference
3. **Structured Outputs** - Generates standardized documentation formats
4. **Reusable Tools** - Common utilities (file I/O, diagram generation, LLM discovery)
5. **Extensibility** - Easy to add new agents or modify existing ones

## Configuration

Edit `backend/app/core/config.py` to customize LLM models and Ollama settings:

```python
class Settings:
    GPT_LLM = os.getenv("GPT_LLM", "qwen2.5:1.5b")
    QWEN_LLM = os.getenv("QWEN_LLM", "qwen2.5:1.5b")
    DEEPSEEK_LLM = os.getenv("DEEPSEEK_LLM", "deepseek-r1:14b")
    MISTRAL_LLM = os.getenv("MISTRAL_LLM", "mistral-small:24b")
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
    OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", 0))
```

## Contributing

We welcome contributions! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Development Setup

```bash
# Backend development
cd backend
pip install -r requirements.txt
# Make your changes and test

# Frontend development
cd frontend
npm install
npm run dev
# Make your changes and test the UI
```

## Roadmap

### Phase 1: Planning & Analysis ✅ (Complete)
- [x] Manager Agent - Orchestration and workflow coordination
- [x] Project Lead Agent - SRS generation and requirements analysis
- [x] Milestone Agent - Project breakdown and timeline planning
- [x] Flow Diagram Agent - System architecture visualization

### Phase 2: Code Generation 🔄 (In Progress)
- [ ] Code Generation Agent - Generate implementation code from SRS
- [ ] Language-specific code generators (Python, JavaScript, TypeScript, etc.)
- [ ] Architecture implementation templates
- [ ] Database schema generation

### Phase 3: Quality Assurance 📋 (Pending)
- [ ] Code Review Agent - Automated code analysis and best practices validation
- [ ] Test Case Generation Agent - Create unit and integration tests
- [ ] Bug detection and security vulnerability scanning
- [ ] Performance optimization recommendations

### Phase 4: Finalization & Integration 🎯 (Pending)
- [ ] Finalization Agent - Consolidate generated code and documentation
- [ ] API documentation generation
- [ ] Deployment configuration generation (Docker, K8s, etc.)
- [ ] Project structure finalization and cleanup

### Additional Features 🚀 (Backlog)
- [ ] Web UI for project submissions and result visualization
- [ ] Support for additional LLM providers (OpenAI, Claude, etc.)
- [ ] Integration with Git for version control
- [ ] Docker containerization for easy deployment
- [ ] Multi-agent debate and consensus framework
- [ ] Custom agent creation framework
- [ ] Batch processing for multiple projects
- [ ] Result caching and optimization


## Acknowledgments

- Built with [LangChain](https://langchain.com/) and [LangGraph](https://langchain-ai.github.io/langgraph/)
- LLM inference powered by [Ollama](https://ollama.ai/)
- UI built with [Next.js](https://nextjs.org/) and [Tailwind CSS](https://tailwindcss.com/)

## Support

If you have questions or run into issues:
- Check the [Issues](https://github.com/yourusername/dev-council/issues) page
- Review the [Documentation](./docs)
- Start a [Discussion](https://github.com/yourusername/dev-council/discussions)

---

**Made with ❤️ by the OpenLLM-Council**
