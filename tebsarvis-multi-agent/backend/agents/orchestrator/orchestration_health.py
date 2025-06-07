class OrchestrationHealthMonitor:
    """Monitor health of orchestration components"""
    
    def __init__(self, orchestration_manager: OrchestrationManager):
        self.orchestration_manager = orchestration_manager
        self.logger = logging.getLogger("orchestration_health")
    
    async def perform_health_check(self) -> Dict[str, Any]:
        """Comprehensive health check of orchestration system"""
        health_status = {
            'overall_health': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'components': {}
        }
        
        try:
            # Check coordinator health
            coord_metrics = self.orchestration_manager.coordinator.get_coordination_metrics()
            health_status['components']['coordinator'] = {
                'status': 'healthy' if coord_metrics['success_rate'] > 80 else 'degraded',
                'success_rate': coord_metrics['success_rate'],
                'active_workflows': coord_metrics['active_workflows']
            }
            
            # Check task dispatcher health
            dispatcher_status = self.orchestration_manager.task_dispatcher.get_dispatcher_status()
            queue_health = 'healthy' if dispatcher_status['total_queue_size'] < 100 else 'warning'
            health_status['components']['task_dispatcher'] = {
                'status': queue_health,
                'queue_size': dispatcher_status['total_queue_size'],
                'strategy': dispatcher_status['load_balancing_strategy']
            }
            
            # Check collaboration manager health
            collab_stats = self.orchestration_manager.collaboration_manager.get_manager_status()
            collab_health = 'healthy' if collab_stats['active_sessions'] < 15 else 'warning'
            health_status['components']['collaboration_manager'] = {
                'status': collab_health,
                'active_sessions': collab_stats['active_sessions'],
                'success_rate': collab_stats['statistics'].get('success_rate', 0)
            }
            
            # Check workflow engine health
            workflow_status = self.orchestration_manager.workflow_engine.get_engine_status()
            workflow_health = 'healthy' if workflow_status['active_executions'] < 40 else 'warning'
            health_status['components']['workflow_engine'] = {
                'status': workflow_health,
                'active_executions': workflow_status['active_executions'],
                'success_rate': workflow_status['success_rate']
            }
            
            # Determine overall health
            component_statuses = [comp['status'] for comp in health_status['components'].values()]
            if 'degraded' in component_statuses:
                health_status['overall_health'] = 'degraded'
            elif component_statuses.count('warning') > 1:
                health_status['overall_health'] = 'warning'
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"Error performing health check: {str(e)}")
            return {
                'overall_health': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }



async def resolve_incident_intelligently(orchestration_manager: OrchestrationManager, 
                                        incident_data: Dict[str, Any]) -> Dict[str, Any]:
    """Example of intelligent incident resolution using full orchestration"""
    
    workflow_request = {
        'type': 'incident_resolution',
        'complexity': 'high',
        'requires_consensus': True,
        'agents_required': ['context_agent', 'search_agent', 'resolution_agent', 'conversation_agent'],
        'incident_data': incident_data,
        'shared_data': {
            'incident_summary': incident_data.get('summary', ''),
            'incident_category': incident_data.get('category', ''),
            'priority_level': incident_data.get('priority', 'normal')
        },
        'timeout_minutes': 45
    }
    
    # Execute with intelligent orchestration
    result = await orchestration_manager.execute_intelligent_workflow(workflow_request)
    
    # Monitor execution progress
    execution_id = result.get('execution_id') or result.get('session_id') or result.get('workflow_id')
    
    if execution_id:
        # Wait for completion (simplified)
        await asyncio.sleep(30)  # In real implementation, use proper monitoring
        
        # Get final status
        if result.get('execution_method') == 'workflow_engine':
            final_status = await orchestration_manager.workflow_engine.get_execution_status(execution_id)
        elif result.get('execution_method') == 'collaboration_manager':
            final_status = await orchestration_manager.collaboration_manager.get_collaboration_status(execution_id)
        else:
            final_status = await orchestration_manager.coordinator.get_workflow_status(execution_id)
        
        return {
            'incident_resolution': final_status,
            'orchestration_method': result.get('execution_method'),
            'execution_id': execution_id
        }
    
    return result


async def optimize_orchestration_performance(orchestration_manager: OrchestrationManager):
    """Example of system-wide performance optimization"""
    
    # Perform health check
    health_monitor = OrchestrationHealthMonitor(orchestration_manager)
    health_status = await health_monitor.perform_health_check()
    
    # Optimize based on health status
    if health_status['overall_health'] in ['warning', 'degraded']:
        optimizations = await orchestration_manager.optimize_system_performance()
        return {
            'health_status': health_status,
            'optimizations': optimizations,
            'recommendation': 'System optimized based on current performance metrics'
        }
    
    return {
        'health_status': health_status,
        'recommendation': 'System performing optimally, no changes needed'
    }