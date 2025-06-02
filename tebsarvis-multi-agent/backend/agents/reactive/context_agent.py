"""
Context Agent for TEBSarvis Multi-Agent System
Handles metadata enrichment and environmental context processing.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import re
from collections import Counter

from ..core.base_agent import BaseAgent, AgentCapability
from ..shared.azure_clients import AzureClientManager

class ContextAgent(BaseAgent):
    """
    Context Agent that enriches incident data with metadata and environmental context.
    Provides context analysis, metadata extraction, and environmental factor integration.
    """
    
    def __init__(self, agent_id: str = "context_agent"):
        capabilities = [
            AgentCapability(
                name="metadata_enrichment",
                description="Extract and enrich metadata from incident data",
                input_types=["incident_data", "raw_text"],
                output_types=["enriched_metadata", "extracted_entities"],
                dependencies=["azure_openai"]
            ),
            AgentCapability(
                name="environmental_analysis",
                description="Analyze environmental factors affecting incidents",
                input_types=["incident_context", "system_data"],
                output_types=["environmental_factors", "context_insights"],
                dependencies=["azure_openai"]
            ),
            AgentCapability(
                name="context_correlation",
                description="Correlate incidents with historical and environmental context",
                input_types=["incident_data", "historical_context"],
                output_types=["correlation_analysis", "context_patterns"],
                dependencies=["cosmos_db", "azure_openai"]
            ),
            AgentCapability(
                name="data_validation",
                description="Validate and cleanse incident data",
                input_types=["raw_incident_data"],
                output_types=["validated_data", "quality_assessment"],
                dependencies=[]
            )
        ]
        
        super().__init__(agent_id, "context", capabilities)
        
        self.azure_manager = AzureClientManager()
        self.entity_patterns = self._load_entity_patterns()
        self.context_cache = {}  # Cache for environmental context
        self.cache_duration = 300  # 5 minutes
        
        # Initialize agent
        asyncio.create_task(self._initialize())
    
    async def _initialize(self):
        """Initialize the context agent"""
        try:
            await self.azure_manager.initialize()
            self.logger.info("Context Agent initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Context Agent: {str(e)}")
            raise
    
    def get_capabilities(self) -> List[AgentCapability]:
        """Return agent capabilities"""
        return self.capabilities
    
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process context enrichment tasks.
        
        Args:
            task: Task data containing context processing parameters
            
        Returns:
            Enriched context data with metadata
        """
        task_type = task.get('type', 'metadata_enrichment')
        
        if task_type == 'metadata_enrichment':
            return await self._enrich_metadata(task)
        elif task_type == 'environmental_analysis':
            return await self._analyze_environmental_factors(task)
        elif task_type == 'context_correlation':
            return await self._correlate_context(task)
        elif task_type == 'data_validation':
            return await self._validate_data(task)
        elif task_type == 'context_extraction':
            return await self._extract_context(task)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    async def _enrich_metadata(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich incident data with extracted metadata.
        
        Args:
            task: Task containing incident data to enrich
            
        Returns:
            Dictionary with enriched metadata
        """
        try:
            incident_data = task.get('incident_data', {})
            enrichment_level = task.get('enrichment_level', 'standard')  # minimal, standard, comprehensive
            
            # Extract basic metadata
            basic_metadata = await self._extract_basic_metadata(incident_data)
            
            # Extract entities using pattern matching and AI
            entities = await self._extract_entities(incident_data)
            
            # Analyze temporal context
            temporal_context = await self._analyze_temporal_context(incident_data)
            
            # Get system context if available
            system_context = await self._extract_system_context(incident_data)
            
            # Comprehensive enrichment includes additional analysis
            additional_context = {}
            if enrichment_level in ['standard', 'comprehensive']:
                additional_context = await self._perform_additional_enrichment(incident_data)
            
            # Combine all enrichment data
            enriched_metadata = {
                'basic_metadata': basic_metadata,
                'entities': entities,
                'temporal_context': temporal_context,
                'system_context': system_context,
                'additional_context': additional_context,
                'enrichment_metadata': {
                    'agent_id': self.agent_id,
                    'timestamp': datetime.now().isoformat(),
                    'enrichment_level': enrichment_level,
                    'confidence_score': self._calculate_enrichment_confidence(
                        basic_metadata, entities, temporal_context, system_context
                    )
                }
            }
            
            return enriched_metadata
            
        except Exception as e:
            self.logger.error(f"Error enriching metadata: {str(e)}")
            raise
    
    async def _analyze_environmental_factors(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze environmental factors that might affect incidents.
        
        Args:
            task: Task containing incident context for analysis
            
        Returns:
            Dictionary with environmental analysis
        """
        try:
            incident_context = task.get('incident_context', {})
            analysis_scope = task.get('analysis_scope', ['temporal', 'system', 'user'])
            
            environmental_factors = {}
            
            # Temporal environmental factors
            if 'temporal' in analysis_scope:
                environmental_factors['temporal'] = await self._analyze_temporal_environment(incident_context)
            
            # System environmental factors
            if 'system' in analysis_scope:
                environmental_factors['system'] = await self._analyze_system_environment(incident_context)
            
            # User environmental factors
            if 'user' in analysis_scope:
                environmental_factors['user'] = await self._analyze_user_environment(incident_context)
            
            # Network environmental factors
            if 'network' in analysis_scope:
                environmental_factors['network'] = await self._analyze_network_environment(incident_context)
            
            # Generate insights from environmental analysis
            insights = await self._generate_environmental_insights(environmental_factors)
            
            return {
                'environmental_factors': environmental_factors,
                'insights': insights,
                'risk_assessment': self._assess_environmental_risks(environmental_factors),
                'analysis_metadata': {
                    'agent_id': self.agent_id,
                    'timestamp': datetime.now().isoformat(),
                    'analysis_scope': analysis_scope
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing environmental factors: {str(e)}")
            raise
    
    async def _correlate_context(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Correlate incident with historical and environmental context.
        
        Args:
            task: Task containing incident data and historical context
            
        Returns:
            Dictionary with correlation analysis
        """
        try:
            incident_data = task.get('incident_data', {})
            correlation_types = task.get('correlation_types', ['temporal', 'categorical', 'similarity'])
            
            correlations = {}
            
            # Temporal correlations
            if 'temporal' in correlation_types:
                correlations['temporal'] = await self._find_temporal_correlations(incident_data)
            
            # Categorical correlations
            if 'categorical' in correlation_types:
                correlations['categorical'] = await self._find_categorical_correlations(incident_data)
            
            # Similarity correlations
            if 'similarity' in correlation_types:
                correlations['similarity'] = await self._find_similarity_correlations(incident_data)
            
            # Pattern correlations
            if 'pattern' in correlation_types:
                correlations['pattern'] = await self._find_pattern_correlations(incident_data)
            
            # Generate correlation insights
            correlation_insights = await self._generate_correlation_insights(correlations)
            
            return {
                'correlations': correlations,
                'insights': correlation_insights,
                'correlation_strength': self._calculate_correlation_strength(correlations),
                'correlation_metadata': {
                    'agent_id': self.agent_id,
                    'timestamp': datetime.now().isoformat(),
                    'correlation_types': correlation_types
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error correlating context: {str(e)}")
            raise
    
    async def _validate_data(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and cleanse incident data.
        
        Args:
            task: Task containing raw incident data
            
        Returns:
            Dictionary with validated data and quality assessment
        """
        try:
            raw_data = task.get('raw_incident_data', {})
            validation_rules = task.get('validation_rules', [])
            
            # Perform data validation
            validation_results = self._perform_data_validation(raw_data, validation_rules)
            
            # Cleanse data
            cleansed_data = self._cleanse_data(raw_data, validation_results)
            
            # Assess data quality
            quality_assessment = self._assess_data_quality(raw_data, cleansed_data, validation_results)
            
            return {
                'validated_data': cleansed_data,
                'validation_results': validation_results,
                'quality_assessment': quality_assessment,
                'validation_metadata': {
                    'agent_id': self.agent_id,
                    'timestamp': datetime.now().isoformat(),
                    'rules_applied': len(validation_rules)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error validating data: {str(e)}")
            raise
    
    async def _extract_context(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract comprehensive context from incident data.
        
        Args:
            task: Task containing incident data for context extraction
            
        Returns:
            Dictionary with extracted context
        """
        try:
            incident_data = task.get('incident_data', {})
            context_types = task.get('context_types', ['technical', 'business', 'user'])
            
            extracted_context = {}
            
            # Technical context
            if 'technical' in context_types:
                extracted_context['technical'] = await self._extract_technical_context(incident_data)
            
            # Business context
            if 'business' in context_types:
                extracted_context['business'] = await self._extract_business_context(incident_data)
            
            # User context
            if 'user' in context_types:
                extracted_context['user'] = await self._extract_user_context(incident_data)
            
            # Historical context
            if 'historical' in context_types:
                extracted_context['historical'] = await self._extract_historical_context(incident_data)
            
            # Priority context
            if 'priority' in context_types:
                extracted_context['priority'] = await self._extract_priority_context(incident_data)
            
            return {
                'extracted_context': extracted_context,
                'context_summary': await self._generate_context_summary(extracted_context),
                'extraction_metadata': {
                    'agent_id': self.agent_id,
                    'timestamp': datetime.now().isoformat(),
                    'context_types': context_types
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting context: {str(e)}")
            raise
    
    async def _extract_basic_metadata(self, incident_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract basic metadata from incident data"""
        try:
            summary = incident_data.get('summary', '')
            description = incident_data.get('description', '')
            
            # Extract basic information
            metadata = {
                'text_length': len(summary) + len(description),
                'word_count': len((summary + ' ' + description).split()),
                'has_error_codes': bool(re.search(r'\b(error|code|exception)\s*:?\s*\w+', 
                                                 summary + ' ' + description, re.IGNORECASE)),
                'has_numbers': bool(re.search(r'\d+', summary + ' ' + description)),
                'urgency_indicators': self._extract_urgency_indicators(summary + ' ' + description),
                'language_detected': 'en',  # Simple assumption
                'complexity_score': self._calculate_text_complexity(summary + ' ' + description)
            }
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error extracting basic metadata: {str(e)}")
            return {}
    
    async def _extract_entities(self, incident_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract entities from incident data using patterns and AI"""
        try:
            text_content = f"{incident_data.get('summary', '')} {incident_data.get('description', '')}"
            
            # Pattern-based entity extraction
            pattern_entities = self._extract_entities_with_patterns(text_content)
            
            # AI-based entity extraction
            ai_entities = await self._extract_entities_with_ai(text_content)
            
            # Combine and deduplicate entities
            combined_entities = self._combine_entities(pattern_entities, ai_entities)
            
            return combined_entities
            
        except Exception as e:
            self.logger.error(f"Error extracting entities: {str(e)}")
            return {}
    
    def _extract_entities_with_patterns(self, text: str) -> Dict[str, List[str]]:
        """Extract entities using predefined patterns"""
        entities = {
            'systems': [],
            'error_codes': [],
            'ip_addresses': [],
            'file_paths': [],
            'urls': [],
            'email_addresses': [],
            'software_versions': []
        }
        
        try:
            # System names
            for pattern in self.entity_patterns['systems']:
                matches = re.findall(pattern, text, re.IGNORECASE)
                entities['systems'].extend(matches)
            
            # Error codes
            error_patterns = [
                r'\berror\s*:?\s*(\w+\d+|\d+)',
                r'\bcode\s*:?\s*(\w+\d+|\d+)',
                r'\b(E\d{4,}|ERR_\w+|0x[0-9A-Fa-f]+)'
            ]
            for pattern in error_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                entities['error_codes'].extend([m if isinstance(m, str) else m[0] for m in matches])
            
            # IP addresses
            ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
            entities['ip_addresses'] = re.findall(ip_pattern, text)
            
            # File paths
            path_patterns = [
                r'[A-Za-z]:\\[\w\\\.-]+',  # Windows paths
                r'/[\w/\.-]+',  # Unix paths
            ]
            for pattern in path_patterns:
                entities['file_paths'].extend(re.findall(pattern, text))
            
            # URLs
            url_pattern = r'https?://[\w\.-]+(?:/[\w\.-]*)*'
            entities['urls'] = re.findall(url_pattern, text)
            
            # Email addresses
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            entities['email_addresses'] = re.findall(email_pattern, text)
            
            # Software versions
            version_pattern = r'\bv?\d+\.\d+(?:\.\d+)*\b'
            entities['software_versions'] = re.findall(version_pattern, text)
            
            # Remove duplicates and empty strings
            for key in entities:
                entities[key] = list(set([e for e in entities[key] if e.strip()]))
            
            return entities
            
        except Exception as e:
            self.logger.error(f"Error in pattern-based entity extraction: {str(e)}")
            return entities
    
    async def _extract_entities_with_ai(self, text: str) -> Dict[str, List[str]]:
        """Extract entities using AI"""
        try:
            entity_prompt = f"""
            Extract technical entities from this IT incident description:
            
            Text: "{text}"
            
            Extract the following types of entities:
            - Systems/Applications (e.g., "LMS", "Active Directory", "SQL Server")
            - Technologies (e.g., "Windows 10", "Chrome", "Office 365")
            - Error types (e.g., "login failure", "timeout", "connection error")
            - User actions (e.g., "password reset", "file access", "printing")
            
            Respond in JSON format:
            {{
                "systems": ["system1", "system2"],
                "technologies": ["tech1", "tech2"],
                "error_types": ["error1", "error2"],
                "user_actions": ["action1", "action2"]
            }}
            """
            
            response = await self.azure_manager.get_chat_completion(
                messages=[{"role": "user", "content": entity_prompt}],
                temperature=0.3,
                max_tokens=300
            )
            
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                return {}
                
        except Exception as e:
            self.logger.error(f"Error in AI entity extraction: {str(e)}")
            return {}
    
    def _combine_entities(self, pattern_entities: Dict[str, List[str]], 
                         ai_entities: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Combine entities from pattern and AI extraction"""
        combined = pattern_entities.copy()
        
        # Add AI entities
        for key, values in ai_entities.items():
            if key in combined:
                combined[key].extend(values)
            else:
                combined[key] = values
        
        # Remove duplicates
        for key in combined:
            combined[key] = list(set(combined[key]))
        
        return combined
    
    async def _analyze_temporal_context(self, incident_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze temporal context of the incident"""
        try:
            date_submitted = incident_data.get('date_submitted', '')
            
            # Parse date if string
            if isinstance(date_submitted, str):
                try:
                    incident_time = datetime.strptime(date_submitted, '%d-%m-%Y %H:%M')
                except ValueError:
                    incident_time = datetime.now()
            else:
                incident_time = datetime.now()
            
            temporal_context = {
                'day_of_week': incident_time.strftime('%A'),
                'hour_of_day': incident_time.hour,
                'is_business_hours': 9 <= incident_time.hour <= 17,
                'is_weekend': incident_time.weekday() >= 5,
                'quarter': (incident_time.month - 1) // 3 + 1,
                'season': self._get_season(incident_time.month),
                'time_classification': self._classify_time_period(incident_time),
                'relative_timing': self._get_relative_timing(incident_time)
            }
            
            return temporal_context
            
        except Exception as e:
            self.logger.error(f"Error analyzing temporal context: {str(e)}")
            return {}
    
    async def _extract_system_context(self, incident_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract system-related context"""
        try:
            category = incident_data.get('category', '')
            summary = incident_data.get('summary', '')
            description = incident_data.get('description', '')
            
            text_content = f"{category} {summary} {description}".lower()
            
            system_context = {
                'primary_system': category,
                'system_type': self._classify_system_type(category),
                'affected_components': self._identify_affected_components(text_content),
                'integration_dependencies': self._identify_dependencies(text_content),
                'system_criticality': self._assess_system_criticality(category),
                'technology_stack': self._identify_technology_stack(text_content)
            }
            
            return system_context
            
        except Exception as e:
            self.logger.error(f"Error extracting system context: {str(e)}")
            return {}
    
    async def _perform_additional_enrichment(self, incident_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform additional enrichment analysis"""
        try:
            additional_context = {}
            
            # Sentiment analysis
            text_content = f"{incident_data.get('summary', '')} {incident_data.get('description', '')}"
            additional_context['sentiment'] = await self._analyze_text_sentiment(text_content)
            
            # Complexity analysis
            additional_context['complexity'] = self._analyze_incident_complexity(incident_data)
            
            # Priority indicators
            additional_context['priority_indicators'] = self._extract_priority_indicators(incident_data)
            
            # Communication style
            additional_context['communication_style'] = self._analyze_communication_style(text_content)
            
            return additional_context
            
        except Exception as e:
            self.logger.error(f"Error in additional enrichment: {str(e)}")
            return {}
    
    async def _analyze_temporal_environment(self, incident_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze temporal environmental factors"""
        try:
            current_time = datetime.now()
            
            temporal_env = {
                'current_load_period': self._determine_load_period(current_time),
                'maintenance_window': self._check_maintenance_window(current_time),
                'peak_usage_period': self._is_peak_usage_period(current_time),
                'holiday_period': self._check_holiday_period(current_time),
                'end_of_period_effects': self._check_end_of_period_effects(current_time)
            }
            
            return temporal_env
            
        except Exception as e:
            self.logger.error(f"Error analyzing temporal environment: {str(e)}")
            return {}
    
    async def _analyze_system_environment(self, incident_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze system environmental factors"""
        try:
            # This would typically query system monitoring data
            # For now, we'll provide a basic structure
            
            system_env = {
                'system_load': 'normal',  # Would be queried from monitoring
                'recent_deployments': [],  # Would be queried from deployment systems
                'active_maintenance': False,
                'system_health_status': 'healthy',
                'resource_utilization': {
                    'cpu': 'normal',
                    'memory': 'normal',
                    'disk': 'normal',
                    'network': 'normal'
                }
            }
            
            return system_env
            
        except Exception as e:
            self.logger.error(f"Error analyzing system environment: {str(e)}")
            return {}
    
    async def _analyze_user_environment(self, incident_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze user environmental factors"""
        try:
            user_env = {
                'user_activity_level': 'normal',
                'concurrent_users': 'normal',
                'user_type_distribution': {
                    'internal': 70,
                    'external': 30
                },
                'geographic_distribution': ['local', 'regional'],
                'access_patterns': 'normal'
            }
            
            return user_env
            
        except Exception as e:
            self.logger.error(f"Error analyzing user environment: {str(e)}")
            return {}
    
    async def _analyze_network_environment(self, incident_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze network environmental factors"""
        try:
            network_env = {
                'network_latency': 'normal',
                'bandwidth_utilization': 'normal',
                'connection_quality': 'stable',
                'external_connectivity': 'healthy',
                'cdn_status': 'operational'
            }
            
            return network_env
            
        except Exception as e:
            self.logger.error(f"Error analyzing network environment: {str(e)}")
            return {}
    
    async def _find_temporal_correlations(self, incident_data: Dict[str, Any]) -> Dict[str, Any]:
        """Find temporal correlations with historical incidents"""
        try:
            # Query similar incidents from the same time periods
            query = """
            SELECT * FROM c 
            WHERE c.category = @category 
            AND c.date_submitted >= @start_date 
            AND c.date_submitted <= @end_date
            """
            
            category = incident_data.get('category')
            current_time = datetime.now()
            
            # Look for incidents in the same time frame over the past year
            correlations = {
                'same_day_of_week': [],
                'same_hour_of_day': [],
                'seasonal_patterns': [],
                'recent_similar': []
            }
            
            # This would be populated by actual database queries
            # For now, providing structure
            
            return correlations
            
        except Exception as e:
            self.logger.error(f"Error finding temporal correlations: {str(e)}")
            return {}
    
    async def _find_categorical_correlations(self, incident_data: Dict[str, Any]) -> Dict[str, Any]:
        """Find categorical correlations"""
        try:
            category = incident_data.get('category')
            severity = incident_data.get('severity')
            
            correlations = {
                'same_category_incidents': {'count': 0, 'trends': []},
                'cross_category_dependencies': [],
                'severity_patterns': [],
                'escalation_correlations': []
            }
            
            return correlations
            
        except Exception as e:
            self.logger.error(f"Error finding categorical correlations: {str(e)}")
            return {}
    
    async def _find_similarity_correlations(self, incident_data: Dict[str, Any]) -> Dict[str, Any]:
        """Find similarity-based correlations"""
        try:
            # This would use the search agent to find similar incidents
            summary = incident_data.get('summary', '')
            description = incident_data.get('description', '')
            
            search_text = f"{summary} {description}"
            
            # Use search agent (would be implemented with proper agent communication)
            similar_incidents = await self.azure_manager.search_similar_incidents(
                query_text=search_text,
                top_k=5
            )
            
            correlations = {
                'similar_incidents': similar_incidents,
                'common_patterns': self._extract_common_patterns(similar_incidents),
                'resolution_patterns': self._extract_resolution_patterns(similar_incidents)
            }
            
            return correlations
            
        except Exception as e:
            self.logger.error(f"Error finding similarity correlations: {str(e)}")
            return {}
    
    def _extract_urgency_indicators(self, text: str) -> List[str]:
        """Extract urgency indicators from text"""
        urgency_patterns = [
            r'\b(urgent|critical|emergency|asap|immediately|high priority)\b',
            r'\b(down|outage|failure|broken|not working)\b',
            r'\b(production|live|critical system)\b'
        ]
        
        indicators = []
        for pattern in urgency_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            indicators.extend(matches)
        
        return list(set(indicators))
    
    def _calculate_text_complexity(self, text: str) -> float:
        """Calculate text complexity score"""
        if not text:
            return 0.0
        
        words = text.split()
        if not words:
            return 0.0
        
        # Simple complexity metrics
        avg_word_length = sum(len(word) for word in words) / len(words)
        sentence_count = len(re.split(r'[.!?]+', text))
        avg_sentence_length = len(words) / max(sentence_count, 1)
        
        # Technical terms bonus
        technical_terms = len(re.findall(r'\b\w*(?:error|system|server|network|database)\w*\b', text, re.IGNORECASE))
        
        complexity = (avg_word_length / 10) + (avg_sentence_length / 20) + (technical_terms / 10)
        return min(complexity, 1.0)
    
    def _get_season(self, month: int) -> str:
        """Get season from month"""
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'autumn'
    
    def _classify_time_period(self, dt: datetime) -> str:
        """Classify time period"""
        hour = dt.hour
        if 6 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 18:
            return 'afternoon'
        elif 18 <= hour < 22:
            return 'evening'
        else:
            return 'night'
    
    def _get_relative_timing(self, dt: datetime) -> str:
        """Get relative timing description"""
        now = datetime.now()
        diff = now - dt
        
        if diff.days == 0:
            return 'today'
        elif diff.days == 1:
            return 'yesterday'
        elif diff.days <= 7:
            return 'this_week'
        elif diff.days <= 30:
            return 'this_month'
        else:
            return 'older'
    
    def _load_entity_patterns(self) -> Dict[str, List[str]]:
        """Load entity extraction patterns"""
        return {
            'systems': [
                r'\b(lms|learning management system)\b',
                r'\b(active directory|ad)\b',
                r'\b(sql server|database)\b',
                r'\b(exchange|email system)\b',
                r'\b(sharepoint|collaboration)\b'
            ],
            'applications': [
                r'\b(outlook|office|word|excel|powerpoint)\b',
                r'\b(chrome|firefox|edge|browser)\b',
                r'\b(teams|skype|zoom)\b'
            ]
        }
    
    def _calculate_enrichment_confidence(self, basic_metadata: Dict[str, Any], 
                                       entities: Dict[str, Any], 
                                       temporal_context: Dict[str, Any], 
                                       system_context: Dict[str, Any]) -> float:
        """Calculate confidence score for enrichment"""
        scores = []
        
        # Basic metadata score
        if basic_metadata:
            scores.append(0.8 if basic_metadata.get('word_count', 0) > 5 else 0.4)
        
        # Entities score
        entity_count = sum(len(entities.get(key, [])) for key in entities)
        scores.append(min(entity_count * 0.1, 1.0))
        
        # Context scores
        if temporal_context:
            scores.append(0.9)
        if system_context:
            scores.append(0.9)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    # Additional helper methods would continue here...
    # Including methods for data validation, quality assessment, etc.
    
    def _perform_data_validation(self, raw_data: Dict[str, Any], 
                                validation_rules: List[str]) -> Dict[str, Any]:
        """Perform data validation checks"""
        validation_results = {
            'passed': [],
            'failed': [],
            'warnings': []
        }
        
        # Basic validation checks
        required_fields = ['summary', 'category', 'priority']
        for field in required_fields:
            if field in raw_data and raw_data[field]:
                validation_results['passed'].append(f"Required field '{field}' present")
            else:
                validation_results['failed'].append(f"Required field '{field}' missing or empty")
        
        # Data type validation
        if 'date_submitted' in raw_data:
            try:
                datetime.strptime(raw_data['date_submitted'], '%d-%m-%Y %H:%M')
                validation_results['passed'].append("Date format valid")
            except ValueError:
                validation_results['warnings'].append("Date format may be invalid")
        
        return validation_results
    
    def _cleanse_data(self, raw_data: Dict[str, Any], 
                     validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Cleanse and normalize data"""
        cleansed_data = raw_data.copy()
        
        # Trim whitespace
        for key, value in cleansed_data.items():
            if isinstance(value, str):
                cleansed_data[key] = value.strip()
        
        # Normalize category names
        if 'category' in cleansed_data:
            category = cleansed_data['category'].lower()
            if 'lms' in category or 'learning' in category:
                cleansed_data['category'] = 'Learning Management System (LMS)'
        
        return cleansed_data
    
    def _assess_data_quality(self, raw_data: Dict[str, Any], 
                           cleansed_data: Dict[str, Any], 
                           validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall data quality"""
        total_checks = len(validation_results['passed']) + len(validation_results['failed'])
        passed_checks = len(validation_results['passed'])
        
        quality_score = passed_checks / total_checks if total_checks > 0 else 0.0
        
        return {
            'quality_score': quality_score,
            'completeness': self._calculate_completeness(cleansed_data),
            'consistency': self._calculate_consistency(cleansed_data),
            'recommendations': self._generate_quality_recommendations(validation_results)
        }
    
    def _calculate_completeness(self, data: Dict[str, Any]) -> float:
        """Calculate data completeness score"""
        expected_fields = ['summary', 'category', 'priority', 'severity', 'date_submitted']
        present_fields = sum(1 for field in expected_fields if data.get(field))
        return present_fields / len(expected_fields)
    
    def _calculate_consistency(self, data: Dict[str, Any]) -> float:
        """Calculate data consistency score"""
        # Simple consistency checks
        consistency_score = 1.0
        
        # Check if priority and severity are aligned
        priority = data.get('priority', '').lower()
        severity = data.get('severity', '').lower()
        
        if ('high' in priority and 'severity 3' in severity) or \
           ('low' in priority and 'severity 1' in severity):
            consistency_score -= 0.2
        
        return max(consistency_score, 0.0)
    
    def _generate_quality_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations for improving data quality"""
        recommendations = []
        
        if validation_results['failed']:
            recommendations.append("Complete missing required fields")
        
        if validation_results['warnings']:
            recommendations.append("Review and correct data format issues")
        
        return recommendations