# TEBSarvis
tebsarvis-multi-agent/
├── backend/
│   ├── agents/
│   │   ├── core/
│   │   │   ├── base_agent.py                    # Abstract base class for all agents
│   │   │   ├── agent_communication.py           # Inter-agent messaging protocol
│   │   │   ├── message_types.py                 # Message schemas and types
│   │   │   └── agent_registry.py                # Agent discovery and registration
│   │   ├── reactive/
│   │   │   ├── resolution_agent.py              # GPT-4 + RAG solution generator
│   │   │   ├── search_agent.py                  # Vector + semantic search engine
│   │   │   ├── conversation_agent.py            # NLP conversation handler
│   │   │   └── context_agent.py                 # Metadata enrichment processor
│   │   ├── proactive/
│   │   │   ├── pattern_detection_agent.py       # ML clustering and trend analysis
│   │   │   └── alerting_agent.py                # Rule-based alerting system
│   │   └── orchestrator/
│   │       ├── agent_coordinator.py             # Central coordination hub
│   │       ├── task_dispatcher.py               # Task routing and load balancing
│   │       ├── workflow_engine.py               # Multi-agent workflow management
│   │       └── collaboration_manager.py         # Agent collaboration protocols
│   ├── azure-functions/
│   │   ├── agent-orchestrator/
│   │   │   ├── function_app.py                  # Main orchestration endpoint
│   │   │   └── host.json
│   │   ├── recommend-resolution/
│   │   │   ├── function_app.py                  # Resolution Agent API endpoint
│   │   │   └── host.json
│   │   ├── search-similar-incidents/
│   │   │   ├── function_app.py                  # Search Agent API endpoint
│   │   │   └── host.json
│   │   ├── ask-assistant/
│   │   │   ├── function_app.py                  # Conversation Agent API endpoint
│   │   │   └── host.json
│   │   ├── detect-patterns/
│   │   │   ├── function_app.py                  # Pattern Detection Agent API
│   │   │   └── host.json
│   │   ├── proactive-alerts/
│   │   │   ├── function_app.py                  # Alerting Agent API endpoint
│   │   │   └── host.json
│   │   └── shared/
│   │       ├── azure_clients.py                 # Azure service connectors
│   │       ├── rag_pipeline.py                  # RAG implementation
│   │       └── agent_utils.py                   # Shared agent utilities
│   ├── workflows/
│   │   ├── incident_resolution_workflow.py      # Multi-agent incident handling
│   │   ├── proactive_monitoring_workflow.py     # Pattern detection pipeline
│   │   └── conversation_workflow.py             # Chat interaction flow
│   ├── data-processing/
│   │   ├── data_ingestion.py                    # Excel data processor
│   │   ├── vector_embeddings.py                 # Embedding generation
│   │   └── knowledge_base_builder.py            # Search index creation
│   └── config/
│       ├── agent_config.py                      # Agent configuration settings
│       └── azure_config.py                      # Azure service settings
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── AgentDashboard.vue               # Multi-agent status dashboard
│   │   │   ├── ChatInterface.vue                # Conversation Agent UI
│   │   │   ├── IncidentForm.vue                 # Resolution workflow UI
│   │   │   ├── PatternInsights.vue              # Pattern Detection Agent UI
│   │   │   ├── AlertsPanel.vue                  # Alerting Agent UI
│   │   │   └── AgentCollaboration.vue           # Agent interaction visualization
│   │   ├── services/
│   │   │   ├── agent_api.js                     # Agent API client
│   │   │   ├── websocket.js                     # Real-time agent communication
│   │   │   └── workflow_client.js               # Workflow execution client
│   │   ├── store/
│   │   │   ├── agents.js                        # Agent state management
│   │   │   ├── workflows.js                     # Workflow state management
│   │   │   └── incidents.js                     # Incident data management
│   │   └── App.vue
│   └── package.json
├── data/
│   ├── raw/
│   │   └── CaseDataWithResolution.xlsx
│   ├── processed/
│   │   ├── incidents.json
│   │   ├── embeddings.json
│   │   └── knowledge_base.json
│   └── agent-training/
│       ├── resolution_patterns.json
│       ├── conversation_examples.json
│       └── pattern_samples.json
├── scripts/
│   ├── setup_agents.py                          # Agent initialization script
│   ├── deploy_multi_agent.sh                    # Multi-agent deployment
│   ├── test_agent_communication.py              # Agent interaction testing
│   └── monitor_agents.py                        # Agent health monitoring
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
│   ├── agent_specifications.md
│   ├── workflow_documentation.md
│   ├── api_documentation.md
│   └── deployment_guide.md
├── .env.example
├── requirements.txt
└── README.md

