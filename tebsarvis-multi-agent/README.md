Directory structure:
└── annuag2467-tebsarvis/
    ├── README.md
    └── tebsarvis-multi-agent/
        ├── README.md
        ├── requirements.txt
        ├── .env.example
        ├── backend/
        │   ├── agents/
        │   │   ├── core/
        │   │   │   ├── agent_communication.py
        │   │   │   ├── agent_registry.py
        │   │   │   ├── agent_system.py
        │   │   │   ├── base_agent.py
        │   │   │   ├── message_bus_manager.py
        │   │   │   └── message_types.py
        │   │   ├── orchestrator/
        │   │   │   ├── agent_coordinator.py
        │   │   │   ├── collaboration_manager.py
        │   │   │   ├── orchestration_health.py
        │   │   │   ├── orchestration_manager.py
        │   │   │   ├── task_dispatcher.py
        │   │   │   └── workflow_engine.py
        │   │   ├── proactive/
        │   │   │   ├── alerting_agent.py
        │   │   │   └── pattern_detection_agent.py
        │   │   └── reactive/
        │   │       ├── context_agent.py
        │   │       ├── conversation_agent.py
        │   │       ├── resolution_agent.py
        │   │       └── search_agent.py
        │   ├── azure-functions/
        │   │   ├── agent-orchestrator/
        │   │   │   ├── function_app.py
        │   │   │   └── host.json
        │   │   ├── ask-assistant/
        │   │   │   ├── function_app.py
        │   │   │   └── host.json
        │   │   ├── detect-patterns/
        │   │   │   ├── function_app.py
        │   │   │   └── host.json
        │   │   ├── health-check/
        │   │   │   ├── function_app.py
        │   │   │   └── host.json
        │   │   ├── proactive-alerts/
        │   │   │   ├── function_app.py
        │   │   │   └── host.json
        │   │   ├── recommend-resolution/
        │   │   │   ├── function_app.py
        │   │   │   └── host.json
        │   │   ├── search-similar-incidents/
        │   │   │   ├── function_app.py
        │   │   │   └── host.json
        │   │   └── shared/
        │   │       ├── agent_utils.py
        │   │       ├── azure_clients.py
        │   │       ├── function_utils.py
        │   │       └── rag_pipeline.py
        │   ├── config/
        │   │   ├── agent_config.py
        │   │   └── azure_config.py
        │   ├── data-processing/
        │   │   ├── data_ingestion.py
        │   │   ├── knowledge_base_builder.py
        │   │   └── vector_embeddings.py
        │   └── workflows/
        │       ├── conversation_workflow.py
        │       ├── incident_resolution_workflow.py
        │       └── proactive_monitoring_workflow.py
     