"""
Resolution Agent for TEBSarvis Multi-Agent System
Generates ranked solution recommendations using GPT-4 and RAG pipeline.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from ..core.base_agent import BaseAgent, AgentCapability
from ..shared.azure_clients import AzureClientManager
# from ...azure-functions.shared.azure_clients import AzureClientManager
from ...config.agent_config import get_agent_config, AgentType
class ResolutionAgent(BaseAgent):
    """
    Resolution Agent that generates solution recommendations for incidents.
    Uses GPT-4 with RAG to provide contextual, ranked solutions.
    """
    
    def __init__(self, agent_id: str = "resolution_agent", agent_system=None):
        capabilities = [
            AgentCapability(
                name="incident_resolution",
                description="Generate ranked solution recommendations for incidents",
                input_types=["incident_data", "resolution_request"],
                output_types=["solution_recommendations", "resolution_steps"],
                dependencies=["azure_openai", "search_agent"]
            ),
            AgentCapability(
                name="solution_ranking",
                description="Rank and score solution alternatives",
                input_types=["solution_candidates", "incident_context"],
                output_types=["ranked_solutions"],
                dependencies=["azure_openai"]
            ),
            AgentCapability(
                name="resolution_validation",
                description="Validate and improve solution quality",
                input_types=["proposed_solution", "incident_data"],
                output_types=["validation_result", "improved_solution"],
                dependencies=["azure_openai"]
            )
        ]

        config_manager = get_agent_config()
        agent_config = config_manager.get_agent_config(AgentType.RESOLUTION)
        capabilities = agent_config.get('capabilities', [])
        
        super().__init__(agent_id, "resolution", capabilities, agent_system)

        performance_config = config_manager.get_agent_performance_config(AgentType.RESOLUTION)
        self.max_concurrent_tasks = performance_config.max_concurrent_tasks
        self.task_timeout = performance_config.task_timeout
        
        self.azure_manager = AzureClientManager()
        self.max_similar_incidents = 5
        self.confidence_threshold = 0.6
        self.solution_templates = self._load_solution_templates()
        
        # Initialize agent
        asyncio.create_task(self._initialize())
    
    async def _initialize(self):
        """Initialize the agent with proper error handling"""
        try:
            # Initialize Azure manager
            await self.azure_manager.initialize()
            
            # Verify Azure services are accessible
            health_status = await self.azure_manager.get_health_status()
            
            unhealthy_services = [
                service for service, status in health_status.items() 
                if status != 'healthy' and service != 'timestamp'
            ]
            
            if unhealthy_services:
                self.logger.warning(f"Some Azure services are unhealthy: {unhealthy_services}")
            
            self.logger.info(f"{self.agent_type.title()} Agent initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize {self.agent_type} Agent: {str(e)}")
            raise
    
    def get_capabilities(self) -> List[AgentCapability]:
        """Return agent capabilities"""
        return self.capabilities
    
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process resolution tasks.
        
        Args:
            task: Task data containing incident information
            
        Returns:
            Resolution recommendations with confidence scores
        """
        task_type = task.get('type', 'incident_resolution')
        
        if task_type == 'incident_resolution':
            return await self._generate_resolution_recommendations(task)
        elif task_type == 'solution_ranking':
            return await self._rank_solutions(task)
        elif task_type == 'resolution_validation':
            return await self._validate_resolution(task)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    async def _generate_resolution_recommendations(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate resolution recommendations for an incident.
        
        Args:
            task: Task containing incident data
            
        Returns:
            Dictionary with resolution recommendations
        """
        try:
            incident_data = task.get('incident_data', {})
            
            # Extract incident information
            incident_summary = incident_data.get('summary', '')
            incident_description = incident_data.get('description', '')
            category = incident_data.get('category', '')
            severity = incident_data.get('severity', '')
            priority = incident_data.get('priority', '')
            
            # Step 1: Retrieve similar incidents using RAG
            similar_incidents = await self._retrieve_similar_incidents(
                incident_summary, incident_description, category
            )
            
            # Step 2: Generate solutions using GPT-4
            solutions = await self._generate_solutions_with_context(
                incident_data, similar_incidents
            )
            
            # Step 3: Rank and score solutions
            ranked_solutions = await self._rank_and_score_solutions(
                solutions, incident_data, similar_incidents
            )
            
            # Step 4: Add implementation details
            detailed_solutions = await self._add_implementation_details(
                ranked_solutions, incident_data
            )
            
            # Step 5: Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(detailed_solutions, similar_incidents)
            
            return {
                'incident_id': incident_data.get('id'),
                'solutions': detailed_solutions,
                'similar_incidents_used': len(similar_incidents),
                'overall_confidence': overall_confidence,
                'processing_metadata': {
                    'agent_id': self.agent_id,
                    'timestamp': datetime.now().isoformat(),
                    'method': 'rag_gpt4'
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating resolution recommendations: {str(e)}")
            raise
    
    async def _retrieve_similar_incidents(self, summary: str, description: str, 
                                        category: str) -> List[Dict[str, Any]]:
        """
        Retrieve similar incidents using hybrid search.
        
        Args:
            summary: Incident summary
            description: Incident description
            category: Incident category
            
        Returns:
            List of similar resolved incidents
        """
        try:
            # Combine summary and description for search
            search_query = f"{summary} {description}".strip()
            
            # Search for similar incidents
            search_results = await self.azure_manager.search_similar_incidents(
                query_text=search_query,
                top_k=self.max_similar_incidents
            )
            
            # Filter for resolved incidents with solutions
            similar_incidents = []
            for result in search_results:
                metadata = result.get('metadata', {})
                if (metadata.get('resolution') and 
                    metadata.get('resolution').strip() and
                    result.get('score', 0) > 0.3):  # Minimum similarity threshold
                    
                    similar_incidents.append({
                        'id': result['id'],
                        'summary': result['content'].split(' ')[0:20],  # First 20 words
                        'resolution': metadata['resolution'],
                        'category': metadata.get('category'),
                        'severity': metadata.get('severity'),
                        'similarity_score': result['score']
                    })
            
            self.logger.info(f"Found {len(similar_incidents)} similar resolved incidents")
            return similar_incidents
            
        except Exception as e:
            self.logger.error(f"Error retrieving similar incidents: {str(e)}")
            return []
    
    async def _generate_solutions_with_context(self, incident_data: Dict[str, Any],
                                             similar_incidents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate solutions using GPT-4 with context from similar incidents.
        
        Args:
            incident_data: Current incident information
            similar_incidents: List of similar resolved incidents
            
        Returns:
            List of generated solutions
        """
        try:
            # Build context from similar incidents
            context = self._build_incident_context(similar_incidents)
            
            # Create system prompt
            system_prompt = f"""
            You are an expert IT support specialist with deep knowledge of incident resolution.
            Your task is to provide multiple solution recommendations for the given incident.
            
            Use the following resolved similar incidents as context:
            {context}
            
            Guidelines:
            - Provide 3-5 distinct solution approaches
            - Each solution should be specific and actionable
            - Include step-by-step instructions where applicable
            - Consider the incident's severity and priority
            - Explain the reasoning behind each solution
            - Estimate the complexity and time required
            """
            
            # Create user prompt with incident details
            user_prompt = f"""
            CURRENT INCIDENT:
            Summary: {incident_data.get('summary', '')}
            Description: {incident_data.get('description', '')}
            Category: {incident_data.get('category', '')}
            Severity: {incident_data.get('severity', '')}
            Priority: {incident_data.get('priority', '')}
            
            Please provide multiple solution recommendations in JSON format:
            {{
                "solutions": [
                    {{
                        "title": "Solution title",
                        "description": "Detailed description",
                        "steps": ["step1", "step2", "step3"],
                        "complexity": "low|medium|high",
                        "estimated_time": "time estimate",
                        "reasoning": "why this solution works",
                        "prerequisites": ["any prerequisites"],
                        "risks": ["potential risks"]
                    }}
                ]
            }}
            """
            
            messages = [
                {"role": "user", "content": user_prompt}
            ]
            
            # Get GPT-4 response
            response = await self.azure_manager.get_chat_completion(
                messages=messages,
                system_prompt=system_prompt,
                temperature=0.7,
                max_tokens=2000
            )
            
            # Parse JSON response
            try:
                solutions_data = json.loads(response)
                return solutions_data.get('solutions', [])
            except json.JSONDecodeError:
                # Fallback: extract solutions from text response
                return self._extract_solutions_from_text(response)
            
        except Exception as e:
            self.logger.error(f"Error generating solutions: {str(e)}")
            return []
    
    def _build_incident_context(self, similar_incidents: List[Dict[str, Any]]) -> str:
        """Build context string from similar incidents"""
        if not similar_incidents:
            return "No similar incidents found."
        
        context_parts = []
        for i, incident in enumerate(similar_incidents, 1):
            context_parts.append(f"""
            Similar Incident {i}:
            Summary: {' '.join(incident['summary'])}
            Resolution: {incident['resolution']}
            Similarity: {incident['similarity_score']:.2f}
            """)
        
        return "\n".join(context_parts)
    
    async def _rank_and_score_solutions(self, solutions: List[Dict[str, Any]], 
                                       incident_data: Dict[str, Any],
                                       similar_incidents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rank and score solution alternatives.
        
        Args:
            solutions: List of generated solutions
            incident_data: Original incident data
            similar_incidents: Similar resolved incidents
            
        Returns:
            List of ranked solutions with scores
        """
        try:
            if not solutions:
                return []
            
            # Create ranking prompt
            ranking_prompt = f"""
            Rank the following solutions for this incident based on:
            1. Likelihood of success (40%)
            2. Implementation complexity (20%)
            3. Time to resolution (20%)
            4. Risk level (20%)
            
            Incident Details:
            Severity: {incident_data.get('severity', '')}
            Priority: {incident_data.get('priority', '')}
            Category: {incident_data.get('category', '')}
            
            Solutions to rank:
            {json.dumps(solutions, indent=2)}
            
            Provide ranking scores (0-100) for each solution in JSON format:
            {{
                "rankings": [
                    {{
                        "solution_index": 0,
                        "overall_score": 85,
                        "success_likelihood": 90,
                        "complexity_score": 80,
                        "time_score": 85,
                        "risk_score": 85,
                        "confidence": 0.8,
                        "justification": "reasoning for this ranking"
                    }}
                ]
            }}
            """
            
            response = await self.azure_manager.get_chat_completion(
                messages=[{"role": "user", "content": ranking_prompt}],
                temperature=0.3,
                max_tokens=1500
            )
            
            # Parse ranking response
            try:
                ranking_data = json.loads(response)
                rankings = ranking_data.get('rankings', [])
                
                # Apply rankings to solutions
                ranked_solutions = []
                for ranking in rankings:
                    solution_index = ranking.get('solution_index', 0)
                    if solution_index < len(solutions):
                        solution = solutions[solution_index].copy()
                        solution['ranking'] = ranking
                        ranked_solutions.append(solution)
                
                # Sort by overall score (descending)
                ranked_solutions.sort(
                    key=lambda x: x.get('ranking', {}).get('overall_score', 0),
                    reverse=True
                )
                
                return ranked_solutions
                
            except json.JSONDecodeError:
                # Fallback: return solutions with basic scoring
                return self._apply_basic_scoring(solutions)
            
        except Exception as e:
            self.logger.error(f"Error ranking solutions: {str(e)}")
            return solutions
    
    async def _add_implementation_details(self, solutions: List[Dict[str, Any]],
                                        incident_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Add detailed implementation information to solutions.
        
        Args:
            solutions: Ranked solutions
            incident_data: Incident information
            
        Returns:
            Solutions with implementation details
        """
        try:
            for solution in solutions:
                # Add implementation timeline
                solution['implementation'] = {
                    'estimated_duration': solution.get('estimated_time', 'Unknown'),
                    'complexity': solution.get('complexity', 'medium'),
                    'required_skills': self._determine_required_skills(solution, incident_data),
                    'tools_needed': self._determine_tools_needed(solution, incident_data),
                    'validation_steps': self._generate_validation_steps(solution)
                }
                
                # Add escalation information
                solution['escalation'] = {
                    'escalate_if_failed': True,
                    'escalation_criteria': self._define_escalation_criteria(solution),
                    'next_level_contact': 'L2 Support Team'
                }
            
            return solutions
            
        except Exception as e:
            self.logger.error(f"Error adding implementation details: {str(e)}")
            return solutions
    
    def _determine_required_skills(self, solution: Dict[str, Any], incident_data: Dict[str, Any]) -> List[str]:
        """Determine required skills for solution implementation"""
        skills = ['Basic IT Support']
        
        category = incident_data.get('category', '').lower()
        complexity = solution.get('complexity', 'medium').lower()
        
        if 'network' in category:
            skills.append('Network Administration')
        if 'security' in category:
            skills.append('Security Knowledge')
        if 'database' in category:
            skills.append('Database Management')
        if complexity == 'high':
            skills.append('Advanced Troubleshooting')
        
        return skills
    
    def _determine_tools_needed(self, solution: Dict[str, Any], incident_data: Dict[str, Any]) -> List[str]:
        """Determine tools needed for solution implementation"""
        tools = ['Remote Access Tools']
        
        category = incident_data.get('category', '').lower()
        
        if 'network' in category:
            tools.extend(['Network Monitoring Tools', 'Command Line Interface'])
        if 'security' in category:
            tools.extend(['Security Scanning Tools', 'Log Analysis Tools'])
        if 'lms' in category.lower():
            tools.append('LMS Admin Panel')
        
        return tools
    
    def _generate_validation_steps(self, solution: Dict[str, Any]) -> List[str]:
        """Generate validation steps for a solution"""
        return [
            'Verify the issue symptoms are resolved',
            'Test core functionality affected by the incident',
            'Monitor for 15-30 minutes to ensure stability',
            'Document the resolution steps taken',
            'Follow up with the user to confirm satisfaction'
        ]
    
    def _define_escalation_criteria(self, solution: Dict[str, Any]) -> List[str]:
        """Define when to escalate the solution"""
        return [
            'Solution does not resolve the issue within estimated time',
            'New issues arise during implementation',
            'User reports continued problems after implementation',
            'Security concerns are identified during resolution'
        ]
    
    def _calculate_overall_confidence(self, solutions: List[Dict[str, Any]], 
                                    similar_incidents: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence in the recommendations"""
        if not solutions:
            return 0.0
        
        # Base confidence on number of similar incidents and solution scores
        base_confidence = min(len(similar_incidents) * 0.15, 0.6)
        
        # Add confidence from top solution ranking
        if solutions and solutions[0].get('ranking'):
            top_solution_confidence = solutions[0]['ranking'].get('confidence', 0.5)
            base_confidence += top_solution_confidence * 0.4
        
        return min(base_confidence, 1.0)
    
    def _extract_solutions_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract solutions from text when JSON parsing fails"""
        # Simple text parsing fallback
        solutions = []
        lines = text.split('\n')
        
        current_solution = {}
        for line in lines:
            line = line.strip()
            if line.startswith('Solution') or line.startswith('##'):
                if current_solution:
                    solutions.append(current_solution)
                current_solution = {
                    'title': line,
                    'description': '',
                    'steps': [],
                    'complexity': 'medium',
                    'estimated_time': '30-60 minutes'
                }
            elif current_solution and line:
                if not current_solution['description']:
                    current_solution['description'] = line
                elif line.startswith('-') or line.startswith('•'):
                    current_solution['steps'].append(line.lstrip('-•').strip())
        
        if current_solution:
            solutions.append(current_solution)
        
        return solutions
    
    def _apply_basic_scoring(self, solutions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply basic scoring when AI ranking fails"""
        for i, solution in enumerate(solutions):
            # Simple scoring based on complexity and position
            complexity_score = {
                'low': 90,
                'medium': 75,
                'high': 60
            }.get(solution.get('complexity', 'medium'), 75)
            
            position_penalty = i * 5  # Slight penalty for later solutions
            overall_score = max(complexity_score - position_penalty, 30)
            
            solution['ranking'] = {
                'solution_index': i,
                'overall_score': overall_score,
                'confidence': 0.6,
                'justification': 'Basic scoring applied'
            }
        
        return solutions
    
    def _load_solution_templates(self) -> Dict[str, Any]:
        """Load solution templates for common incident types"""
        return {
            'password_reset': {
                'steps': [
                    'Navigate to password reset interface',
                    'Enter user credentials',
                    'Generate new temporary password',
                    'Send password to user via secure channel',
                    'Instruct user to change password on first login'
                ],
                'complexity': 'low',
                'estimated_time': '5-10 minutes'
            },
            'access_denied': {
                'steps': [
                    'Verify user permissions in system',
                    'Check group memberships',
                    'Review access control settings',
                    'Update permissions as needed',
                    'Test access with user'
                ],
                'complexity': 'medium',
                'estimated_time': '15-30 minutes'
            },
            'system_error': {
                'steps': [
                    'Review system logs for error details',
                    'Identify root cause of error',
                    'Apply appropriate fix or workaround',
                    'Monitor system stability',
                    'Document resolution for future reference'
                ],
                'complexity': 'high',
                'estimated_time': '30-90 minutes'
            }
        }