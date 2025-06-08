# TEBSarvis - Multi-Agent Incident Resolution System

TEBSarvis is an advanced multi-agent system designed for intelligent incident resolution and proactive system monitoring. It combines multiple specialized AI agents working in coordination to provide automated incident resolution, pattern detection, and interactive assistance.

## 🌟 Key Features

- **Multi-Agent Architecture**: Coordinated AI agents for different specialized tasks
- **Reactive & Proactive Capabilities**: Both response-based and predictive monitoring
- **Advanced RAG Pipeline**: Utilizes GPT-4 with enterprise knowledge base integration
- **Real-time Pattern Detection**: Automated anomaly and pattern recognition
- **Interactive Chat Interface**: Natural language interaction with the system
- **Azure Cloud Integration**: Scalable serverless architecture using Azure Functions

## 🏗️ System Architecture

The system is built on a modular architecture with several key components:

### 1. Core Agent System

- Base agent framework
- Inter-agent communication protocols
- Message bus and registry system
- Dynamic agent discovery and registration

### 2. Agent Types

- **Reactive Agents**:
  - Resolution Agent: RAG + GPT-4 based solution generation
  - Search Agent: Vector & semantic similarity search
  - Conversation Agent: Natural language interaction
  - Context Agent: Metadata and context enhancement

- **Proactive Agents**:
  - Pattern Detection Agent: Automated anomaly detection
  - Alerting Agent: Rule-based proactive monitoring

- **Orchestration Layer**:
  - Agent Coordinator: Central task routing
  - Task Dispatcher: Load-balanced distribution
  - Workflow Engine: Composable process execution
  - Collaboration Manager: Inter-agent communication

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or higher
- Node.js 18+ 
- Azure subscription (for cloud deployment)
- OpenAI API key

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/annuag2467/TEBSarvis.git
   cd TEBSarvis
   ```

2. Set up Python virtual environment:

   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   source venv/bin/activate # Linux/Mac
   ```

3. Install backend dependencies:

   ```bash
   cd tebsarvis-multi-agent
   pip install -r requirements.txt
   ```

4. Install frontend dependencies:

   ```bash
   cd frontend
   npm install
   ```

5. Configure environment variables:

   Create a `.env` file in the root directory with:

   ```env
   OPENAI_API_KEY=your_api_key
   AZURE_TENANT_ID=your_tenant_id
   AZURE_SUBSCRIPTION_ID=your_subscription_id
   ```

## 🔌 API Endpoints

### Agent Orchestrator

- `POST /api/orchestrator/process-incident`
  - Process new incidents through the multi-agent system
  - Body: `{ "description": "string", "severity": "number", "category": "string" }`

### Resolution Agent

- `POST /api/recommend-resolution`
  - Get AI-powered resolution recommendations
  - Body: `{ "incident_id": "string", "context": "object" }`

### Search Agent

- `GET /api/search-similar-incidents`
  - Find similar historical incidents
  - Query params: `q` (search query), `limit` (max results)

### Conversation Agent

- `POST /api/ask-assistant`
  - Interactive chat with the AI assistant
  - Body: `{ "message": "string", "conversation_id": "string" }`

### Pattern Detection

- `GET /api/detect-patterns`
  - Retrieve detected patterns and anomalies
  - Query params: `timeframe` (analysis window)

### Proactive Alerts

- `GET /api/proactive-alerts`
  - Get current system alerts
  - Query params: `severity` (alert level)

## ▶️ Running the System

1. Start the backend services:

   ```bash
   python scripts/setup_agents.py
   ```

2. Deploy Azure Functions:

   ```bash
   ./scripts/deploy_multi_agent.sh
   ```

3. Start the frontend development server:

   ```bash
   cd frontend
   npm run serve
   ```

## 🔄 Workflows

The system supports three main workflows:

1. **Incident Resolution Workflow**
   - Automated incident analysis and resolution
   - Knowledge base enrichment
   - Similar case matching

2. **Proactive Monitoring Workflow**
   - Continuous system monitoring
   - Pattern detection and analysis
   - Automated alert generation

3. **Conversation Workflow**
   - Interactive problem solving
   - Context-aware responses
   - Knowledge base integration

## 🧪 Testing

Run the test suite:

```bash
cd tests
python -m pytest
```

Run integration tests:

```bash
python -m pytest tests/integration/
```

## 📚 Documentation

For detailed documentation, please refer to:

- [Agent Specifications](docs/agent_specifications.md)
- [API Documentation](docs/api_documentation.md)
- [Deployment Guide](docs/deployment_guide.md)
- [Workflow Documentation](docs/workflow_documentation.md)

## 🛠️ Tech Stack

- **Backend**:
  - Python 3.11+
  - Azure Functions
  - OpenAI GPT-4
  - FAISS Vector Database
  
- **Frontend**:
  - Vue.js 3
  - Vuex State Management
  - Tailwind CSS
  
- **Cloud Services**:
  - Azure AI Search
  - Azure Cosmos DB
  - Azure Service Bus
  - Azure Monitor

## 📈 Performance Monitoring

Monitor system health and performance:

```bash
python scripts/monitor_agents.py
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

For more information about the architecture and components, please refer to the [tebsarvis_architecture.pdf](tebsarvis_architecture.pdf) document.
