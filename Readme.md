# NovaTech Multi-Agent Support Graph with RAG & Evaluation

An advanced multi-agent customer support system powered by **LangGraph**, **Gemini 3.1 Flash Lite**, and **Qdrant Vector Database**. The system features a supervisor router that intelligently manages user requests between general customer support, technical documentation RAG lookups, and escalation management paths, tracking history across conversations. It includes full telemetry and automated evaluation suites integrated with **LangSmith**.

---

## 🛠️ Prerequisites

Before getting started, ensure you have the following installed on your machine:
* Python 3.11+
* A virtual environment manager (`venv`)

---

## 🚀 Setup Instructions

### 1. Clone and Navigate to the Project Workspace
```bash
cd "Capstone Project Gen AI"
```

### 2. Activate Your Virtual Environment
```bash
source "env/bin/activate"
```

### 3. Configure Environment Variables (`.env`)

a `.env` file in the root directory of the project and supply your active API credentials. This connects your graph execution to the Gemini model and activates LangSmith tracing:

```env
# Google Gemini API Settings
GEMINI_API_KEY=your_gemini_api_key_here

# LangSmith Automation & Observability Telemetry
LANGCHAIN_API_KEY=your_langsmith_api_key_here
```

---

## 🖥️ Running the Application

To build the knowledge base vectors and spin up the runtime server, run the main entrypoint file:

```bash
python main.py
```

### What this command does:

1. **Loads the RAG Database:** It parses your local policy documents, embeds them, and synchronizes your Qdrant Vector Knowledge Base.
2. **Starts Local Chat:** It bootstraps the Flask web engine application, allowing you to converse with the system via a browser front-end interface with active thread history.

---

## 📊 Automated Quality Evaluations (LangSmith)

The project includes an isolation testing framework (`evaluator.py`) to systematically measure agent precision, fallback reliability, and supervisor routing criteria.

To execute automated batch loop test metrics without running the entire web server:

```bash
python src/evaluator.py
```

### Features of the Evaluation Script:

* **Bootstrap Integration:** Checks your cloud workspace for a validation dataset. If missing, it builds and populates `NovaTech-Capstone-Validation-Suite` automatically.
* **Algorithmic Scoring:** Applies the custom `accuracy_intent_scorer` rule matrix against outputs to verify that the supervisor and specialists (Support, Retrieval, Escalation) are behaving optimally.
* **Terminal Summary Table:** Prints a beautifully formatted analytics chart and Accuracy KPI directly into your console workspace alongside generating full deep-dive trace trees in your LangSmith dashboard.
```

---