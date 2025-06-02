# TEBSarvis

tebsarvis-multi-agent/
├── backend/
│   ├── agents/                                # All AI agents organized by type
│   │   ├── core/                              # Foundational agent components
│   │   │   ├── base_agent.py                  # Abstract base class for agents
│   │   │   ├── agent_communication.py         # Messaging protocols (inter-agent)
│   │   │   ├── message_types.py               # Message formats and enums
│   │   │   └── agent_registry.py              # Dynamic agent discovery/registration
│   │   ├── reactive/                          # Agents that respond to requests
│   │   │   ├── resolution_agent.py            # RAG + GPT-4 based resolution generator
│   │   │   ├── search_agent.py                # Vector & semantic similarity search
│   │   │   ├── conversation_agent.py          # Chat-based agent using LLM
│   │   │   └── context_agent.py               # Adds metadata/context for LLMs
│   │   ├── proactive/                         # Autonomous, monitoring-focused agents
│   │   │   ├── pattern_detection_agent.py     # Clustering/anomaly detection
│   │   │   └── alerting_agent.py              # Rule-based proactive alerts
│   │   └── orchestrator/                      # Coordinates multi-agent workflows
│   │       ├── agent_coordinator.py           # Main task router / controller
│   │       ├── task_dispatcher.py             # Load-balanced task dispatch
│   │       ├── workflow_engine.py             # Composable workflow executor
│   │       └── collaboration_manager.py       # Agent-to-agent collaboration protocols
│   ├── azure-functions/                       # Serverless Azure Function APIs
│   │   ├── agent-orchestrator/
│   │   │   ├── function_app.py                # Central orchestration entrypoint
│   │   │   └── host.json
│   │   ├── recommend-resolution/
│   │   │   ├── function_app.py                # ResolutionAgent API wrapper
│   │   │   └── host.json
│   │   ├── search-similar-incidents/
│   │   │   ├── function_app.py                # SearchAgent API wrapper
│   │   │   └── host.json
│   │   ├── ask-assistant/
│   │   │   ├── function_app.py                # ConversationAgent API wrapper
│   │   │   └── host.json
│   │   ├── detect-patterns/
│   │   │   ├── function_app.py                # PatternDetectionAgent API
│   │   │   └── host.json
│   │   ├── proactive-alerts/
│   │   │   ├── function_app.py                # AlertingAgent API endpoint
│   │   │   └── host.json
│   │   └── shared/
│   │       ├── azure_clients.py               # Azure service connectors (Cognitive, Search)
│   │       ├── rag_pipeline.py                # RAG logic integration
│   │       └── agent_utils.py                 # Utilities for all functions
│   ├── workflows/                             # Multi-agent workflow compositions
│   │   ├── incident_resolution_workflow.py    # Full resolution pipeline
│   │   ├── proactive_monitoring_workflow.py   # Auto alert/pattern detection
│   │   └── conversation_workflow.py           # Chat + search + resolve chain
│   ├── data-processing/
│   │   ├── data_ingestion.py                  # Converts Excel to structured JSON
│   │   ├── vector_embeddings.py               # Embedding generator (e.g. OpenAI/BGE)
│   │   └── knowledge_base_builder.py          # Index builder (FAISS / Azure AI Search)
│   └── config/
│       ├── agent_config.py                    # Agent-specific parameters
│       └── azure_config.py                    # Azure service credentials/settings
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── AgentDashboard.vue             # Overview dashboard of all agents
│   │   │   ├── ChatInterface.vue              # UI for talking to Conversation Agent
│   │   │   ├── IncidentForm.vue               # Incident submission form
│   │   │   ├── PatternInsights.vue            # UI for detected patterns
│   │   │   ├── AlertsPanel.vue                # Shows triggered alerts
│   │   │   └── AgentCollaboration.vue         # Visualizes agent interaction graph
│   │   ├── services/
│   │   │   ├── agent_api.js                   # Agent API interface
│   │   │   ├── websocket.js                   # Live updates from agents
│   │   │   └── workflow_client.js             # Workflow execution handler
│   │   ├── store/
│   │   │   ├── agents.js                      # Vuex store: agents
│   │   │   ├── workflows.js                   # Vuex store: workflows
│   │   │   └── incidents.js                   # Vuex store: incident tracking
│   │   └── App.vue                            # Main app entry
│   └── package.json                           # Vue app config
├── data/
│   ├── raw/
│   │   └── CaseDataWithResolution.xlsx        # Original incident records
│   ├── processed/
│   │   ├── incidents.json                     # Cleaned incident data
│   │   ├── embeddings.json                    # Precomputed vector embeddings
│   │   └── knowledge_base.json                # Structured KB for search
│   └── agent-training/
│       ├── resolution_patterns.json           # Prompt engineering data
│       ├── conversation_examples.json         # LLM chat finetuning examples
│       └── pattern_samples.json               # Samples for pattern detection
├── scripts/
│   ├── setup_agents.py                        # Registers all agents
│   ├── deploy_multi_agent.sh                  # Deploy script (CLI/Azure)
│   ├── test_agent_communication.py            # Simulates agent chat/test
│   └── monitor_agents.py                      # Health checks and logs
├── tests/
│   ├── unit/
│   │   ├── test_agents/
│   │   │   ├── test_resolution_agent.py
│   │   │   ├── test_search_agent.py
│   │   │   ├── test_conversation_agent.py
│   │   │   ├── test_context_agent.py
│   │   │   ├── test_pattern_agent.py
│   │   │   └── test_alerting_agent.py
│   │   └── test_orchestrator/
│   │       ├── test_coordinator.py
│   │       └── test_workflows.py
│   └── integration/
│       ├── test_multi_agent_workflows.py
│       └── test_agent_collaboration.py
├── docs/
│   ├── agent_specifications.md                # Role and flow of each agent
│   ├── workflow_documentation.md             # Step-by-step workflow details
│   ├── api_documentation.md                  # Azure Function endpoints
│   └── deployment_guide.md                   # Guide to deploy all modules
├── .env.example                               # Sample env file
├── requirements.txt                           # Python deps
└── README.md                                  # Project overview (You're here!)
