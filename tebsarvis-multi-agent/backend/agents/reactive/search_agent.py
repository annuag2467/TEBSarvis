"""
Search Agent for TEBSarvis Multi-Agent System
Performs semantic similarity matching across knowledge bases using vector and hybrid search.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import json
import math

from ..core.base_agent import BaseAgent, AgentCapability
from ..shared.azure_clients import AzureClientManager
from ...config.agent_config import get_agent_config, AgentType

class SearchAgent(BaseAgent):
    """
    Search Agent that performs semantic and vector-based searches across incident knowledge base.
    Provides similarity matching, contextual search, and intelligent result ranking.
    """
    
    def __init__(self, agent_id: str = "search_agent", agent_system=None):
        capabilities = [
            AgentCapability(
                name="semantic_search",
                description="Perform semantic similarity search across incidents",
                input_types=["search_query", "text_input"],
                output_types=["search_results", "similarity_scores"],
                dependencies=["azure_search", "azure_openai"]
            ),
            AgentCapability(
                name="vector_search",
                description="Perform vector similarity search using embeddings",
                input_types=["query_vector", "search_parameters"],
                output_types=["vector_results"],
                dependencies=["azure_search"]
            ),
            AgentCapability(
                name="hybrid_search",
                description="Combine text and vector search for best results",
                input_types=["search_query", "search_context"],
                output_types=["ranked_results"],
                dependencies=["azure_search", "azure_openai"]
            ),
            AgentCapability(
                name="contextual_search",
                description="Search with additional context and filtering",
                input_types=["contextual_query"],
                output_types=["filtered_results"],
                dependencies=["azure_search"]
            )
        ]
        config_manager = get_agent_config()
        agent_config = config_manager.get_agent_config(AgentType.SEARCH)
        capabilities = agent_config.get('capabilities', [])
        super().__init__(agent_id, "search", capabilities, agent_system)
        performance_config = config_manager.get_agent_performance_config(AgentType.SEARCH)
        self.max_concurrent_tasks = performance_config.max_concurrent_tasks
        self.task_timeout = performance_config.task_timeout
        self.azure_manager = AzureClientManager()
        self.default_max_results = 10
        self.similarity_threshold = 0.3
        self.search_cache = {}  # Simple in-memory cache
        self.cache_ttl = 300  # 5 minutes
        
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
        Process search tasks.
        
        Args:
            task: Task data containing search parameters
            
        Returns:
            Search results with metadata
        """
        task_type = task.get('type', 'semantic_search')
        
        if task_type == 'semantic_search':
            return await self._perform_semantic_search(task)
        elif task_type == 'vector_search':
            return await self._perform_vector_search(task)
        elif task_type == 'hybrid_search':
            return await self._perform_hybrid_search(task)
        elif task_type == 'contextual_search':
            return await self._perform_contextual_search(task)
        elif task_type == 'similarity_analysis':
            return await self._analyze_similarity(task)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    async def _perform_semantic_search(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform semantic search using text query.
        
        Args:
            task: Task containing search query and parameters
            
        Returns:
            Dictionary with search results and metadata
        """
        try:
            search_query = task.get('query', '')
            max_results = task.get('max_results', self.default_max_results)
            filters = task.get('filters', {})
            include_metadata = task.get('include_metadata', True)
            
            if not search_query.strip():
                return {
                    'results': [],
                    'total_count': 0,
                    'error': 'Empty search query'
                }
            
            # Check cache first
            cache_key = self._generate_cache_key('semantic', search_query, filters, max_results)
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                return cached_result
            
            # Perform text search
            search_results = await self.azure_manager.search_client.text_search(
                query_text=search_query,
                top_k=max_results,
                filters=self._build_filter_string(filters)
            )
            
            # Process and enhance results
            processed_results = await self._process_search_results(
                search_results, search_query, include_metadata
            )
            
            result = {
                'results': processed_results,
                'total_count': len(processed_results),
                'search_type': 'semantic',
                'query': search_query,
                'filters_applied': filters,
                'processing_metadata': {
                    'agent_id': self.agent_id,
                    'timestamp': datetime.now().isoformat(),
                    'cache_hit': False
                }
            }
            
            # Cache the result
            self._cache_result(cache_key, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in semantic search: {str(e)}")
            raise
    
    async def _perform_vector_search(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform vector similarity search using embeddings.
        
        Args:
            task: Task containing query vector or text to convert
            
        Returns:
            Dictionary with vector search results
        """
        try:
            query_vector = task.get('query_vector')
            query_text = task.get('query_text', '')
            max_results = task.get('max_results', self.default_max_results)
            filters = task.get('filters', {})
            similarity_threshold = task.get('similarity_threshold', self.similarity_threshold)
            
            # Generate vector if not provided
            if not query_vector and query_text:
                query_vector = await self.azure_manager.get_embeddings(query_text)
            elif not query_vector:
                return {
                    'results': [],
                    'total_count': 0,
                    'error': 'No query vector or text provided'
                }
            
            # Check cache
            cache_key = self._generate_cache_key('vector', str(query_vector[:10]), filters, max_results)
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                return cached_result
            
            # Perform vector search
            search_results = await self.azure_manager.search_client.vector_search(
                query_vector=query_vector,
                top_k=max_results,
                filters=self._build_filter_string(filters)
            )
            
            # Filter by similarity threshold
            filtered_results = [
                result for result in search_results 
                if result.get('score', 0) >= similarity_threshold
            ]
            
            # Process results
            processed_results = await self._process_search_results(
                filtered_results, query_text or 'vector_query', True
            )
            
            result = {
                'results': processed_results,
                'total_count': len(processed_results),
                'search_type': 'vector',
                'similarity_threshold': similarity_threshold,
                'filters_applied': filters,
                'processing_metadata': {
                    'agent_id': self.agent_id,
                    'timestamp': datetime.now().isoformat(),
                    'cache_hit': False,
                    'vector_dimensions': len(query_vector)
                }
            }
            
            # Cache result
            self._cache_result(cache_key, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in vector search: {str(e)}")
            raise
    
    async def _perform_hybrid_search(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform hybrid search combining text and vector search.
        
        Args:
            task: Task containing search query and parameters
            
        Returns:
            Dictionary with hybrid search results
        """
        try:
            search_query = task.get('query', '')
            max_results = task.get('max_results', self.default_max_results)
            filters = task.get('filters', {})
            boost_recent = task.get('boost_recent', True)
            
            if not search_query.strip():
                return {
                    'results': [],
                    'total_count': 0,
                    'error': 'Empty search query'
                }
            
            # Check cache
            cache_key = self._generate_cache_key('hybrid', search_query, filters, max_results)
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                return cached_result
            
            # Generate query embedding
            query_vector = await self.azure_manager.get_embeddings(search_query)
            
            # Perform hybrid search
            search_results = await self.azure_manager.search_client.hybrid_search(
                query_text=search_query,
                query_vector=query_vector,
                top_k=max_results,
                filters=self._build_filter_string(filters)
            )
            
            # Apply additional ranking logic
            ranked_results = await self._apply_intelligent_ranking(
                search_results, search_query, boost_recent
            )
            
            # Process results
            processed_results = await self._process_search_results(
                ranked_results, search_query, True
            )
            
            result = {
                'results': processed_results,
                'total_count': len(processed_results),
                'search_type': 'hybrid',
                'query': search_query,
                'filters_applied': filters,
                'ranking_applied': True,
                'processing_metadata': {
                    'agent_id': self.agent_id,
                    'timestamp': datetime.now().isoformat(),
                    'cache_hit': False
                }
            }
            
            # Cache result
            self._cache_result(cache_key, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in hybrid search: {str(e)}")
            raise
    
    async def _perform_contextual_search(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform contextual search with additional filtering and context awareness.
        
        Args:
            task: Task containing contextual search parameters
            
        Returns:
            Dictionary with contextual search results
        """
        try:
            search_query = task.get('query', '')
            context = task.get('context', {})
            max_results = task.get('max_results', self.default_max_results)
            filters = task.get('filters', {})
            
            # Enhance query with context
            enhanced_query = await self._enhance_query_with_context(search_query, context)
            
            # Build contextual filters
            contextual_filters = self._build_contextual_filters(filters, context)
            
            # Perform search with enhanced query and filters
            search_task = {
                'query': enhanced_query,
                'max_results': max_results,
                'filters': contextual_filters,
                'boost_recent': context.get('prefer_recent', False)
            }
            
            # Use hybrid search for best results
            results = await self._perform_hybrid_search(search_task)
            
            # Apply context-specific ranking
            results['results'] = await self._apply_contextual_ranking(
                results['results'], context
            )
            
            results['search_type'] = 'contextual'
            results['context_applied'] = context
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in contextual search: {str(e)}")
            raise
    
    async def _analyze_similarity(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze similarity between items or find similar items.
        
        Args:
            task: Task containing similarity analysis parameters
            
        Returns:
            Dictionary with similarity analysis results
        """
        try:
            target_item = task.get('target_item', {})
            comparison_items = task.get('comparison_items', [])
            analysis_type = task.get('analysis_type', 'find_similar')
            
            if analysis_type == 'find_similar':
                return await self._find_similar_items(target_item, task.get('max_results', 10))
            elif analysis_type == 'compare_items':
                return await self._compare_items(target_item, comparison_items)
            elif analysis_type == 'cluster_analysis':
                return await self._perform_cluster_analysis(comparison_items)
            else:
                raise ValueError(f"Unknown analysis type: {analysis_type}")
                
        except Exception as e:
            self.logger.error(f"Error in similarity analysis: {str(e)}")
            raise
    
    async def _find_similar_items(self, target_item: Dict[str, Any], max_results: int) -> Dict[str, Any]:
        """Find items similar to the target item"""
        try:
            # Extract text content from target item
            text_content = self._extract_text_content(target_item)
            
            # Perform vector search
            search_task = {
                'query_text': text_content,
                'max_results': max_results,
                'similarity_threshold': 0.5
            }
            
            results = await self._perform_vector_search(search_task)
            
            # Add similarity explanations
            for result in results['results']:
                result['similarity_explanation'] = await self._explain_similarity(
                    target_item, result
                )
            
            return {
                'target_item': target_item,
                'similar_items': results['results'],
                'analysis_type': 'find_similar',
                'processing_metadata': {
                    'agent_id': self.agent_id,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error finding similar items: {str(e)}")
            raise
    
    async def _compare_items(self, target_item: Dict[str, Any], 
                           comparison_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare target item with a list of comparison items"""
        try:
            target_text = self._extract_text_content(target_item)
            target_vector = await self.azure_manager.get_embeddings(target_text)
            
            comparisons = []
            for item in comparison_items:
                item_text = self._extract_text_content(item)
                item_vector = await self.azure_manager.get_embeddings(item_text)
                
                # Calculate cosine similarity
                similarity = self._calculate_cosine_similarity(target_vector, item_vector)
                
                comparisons.append({
                    'item': item,
                    'similarity_score': similarity,
                    'similarity_explanation': await self._explain_similarity(target_item, item)
                })
            
            # Sort by similarity
            comparisons.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            return {
                'target_item': target_item,
                'comparisons': comparisons,
                'analysis_type': 'compare_items',
                'processing_metadata': {
                    'agent_id': self.agent_id,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error comparing items: {str(e)}")
            raise
    
    async def _process_search_results(self, search_results: List[Dict[str, Any]], 
                                    query: str, include_metadata: bool) -> List[Dict[str, Any]]:
        """Process and enhance search results"""
        processed_results = []
        
        for result in search_results:
            processed_result = {
                'id': result.get('id'),
                'content': result.get('content', ''),
                'score': result.get('score', 0.0),
                'highlights': result.get('highlights', {})
            }
            
            if include_metadata:
                metadata = result.get('metadata', {})
                processed_result['metadata'] = {
                    'category': metadata.get('category'),
                    'severity': metadata.get('severity'),
                    'priority': metadata.get('priority'),
                    'date_submitted': metadata.get('date_submitted'),
                    'resolution': metadata.get('resolution', '')
                }
                
                # Add relevance explanation
                processed_result['relevance_explanation'] = self._generate_relevance_explanation(
                    result, query
                )
            
            processed_results.append(processed_result)
        
        return processed_results
    
    async def _apply_intelligent_ranking(self, search_results: List[Dict[str, Any]], 
                                       query: str, boost_recent: bool) -> List[Dict[str, Any]]:
        """Apply intelligent ranking to search results"""
        try:
            for result in search_results:
                # Start with base search score
                final_score = result.get('score', 0.0)
                
                # Apply recency boost if requested
                if boost_recent:
                    recency_boost = self._calculate_recency_boost(result)
                    final_score *= (1 + recency_boost)
                
                # Apply resolution quality boost
                resolution_boost = self._calculate_resolution_quality_boost(result)
                final_score *= (1 + resolution_boost)
                
                # Apply category relevance boost
                category_boost = self._calculate_category_boost(result, query)
                final_score *= (1 + category_boost)
                
                result['final_score'] = final_score
            
            # Sort by final score
            search_results.sort(key=lambda x: x.get('final_score', 0), reverse=True)
            
            return search_results
            
        except Exception as e:
            self.logger.error(f"Error in intelligent ranking: {str(e)}")
            return search_results
    
    async def _enhance_query_with_context(self, query: str, context: Dict[str, Any]) -> str:
        """Enhance search query with contextual information"""
        enhanced_parts = [query]
        
        # Add category context
        if context.get('category'):
            enhanced_parts.append(f"category:{context['category']}")
        
        # Add severity context
        if context.get('severity'):
            enhanced_parts.append(f"severity:{context['severity']}")
        
        # Add user context
        if context.get('user_role'):
            enhanced_parts.append(f"role:{context['user_role']}")
        
        return " ".join(enhanced_parts)
    
    def _build_contextual_filters(self, base_filters: Dict[str, Any], 
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """Build contextual filters from base filters and context"""
        filters = base_filters.copy()
        
        # Add context-based filters
        if context.get('time_range'):
            filters['date_range'] = context['time_range']
        
        if context.get('exclude_categories'):
            filters['exclude_categories'] = context['exclude_categories']
        
        if context.get('minimum_resolution_quality'):
            filters['min_resolution_quality'] = context['minimum_resolution_quality']
        
        return filters
    
    async def _apply_contextual_ranking(self, results: List[Dict[str, Any]], 
                                      context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply context-specific ranking adjustments"""
        for result in results:
            context_score = result.get('score', 0.0)
            
            # Boost based on user experience level
            user_level = context.get('user_level', 'intermediate')
            if user_level == 'beginner':
                # Boost simpler solutions
                if 'simple' in result.get('content', '').lower():
                    context_score *= 1.2
            elif user_level == 'expert':
                # Boost technical solutions
                if 'advanced' in result.get('content', '').lower():
                    context_score *= 1.2
            
            # Boost based on urgency
            urgency = context.get('urgency', 'normal')
            if urgency == 'high':
                # Boost quick fixes
                metadata = result.get('metadata', {})
                if 'quick' in metadata.get('resolution', '').lower():
                    context_score *= 1.3
            
            result['context_score'] = context_score
        
        # Re-sort by context score
        results.sort(key=lambda x: x.get('context_score', 0), reverse=True)
        return results
    
    def _build_filter_string(self, filters: Dict[str, Any]) -> Optional[str]:
        """Build OData filter string from filter dictionary"""
        if not filters:
            return None
        
        filter_parts = []
        
        if filters.get('category'):
            filter_parts.append(f"metadata/category eq '{filters['category']}'")
        
        if filters.get('severity'):
            filter_parts.append(f"metadata/severity eq '{filters['severity']}'")
        
        if filters.get('date_range'):
            date_range = filters['date_range']
            if date_range.get('start'):
                filter_parts.append(f"metadata/date_submitted ge '{date_range['start']}'")
            if date_range.get('end'):
                filter_parts.append(f"metadata/date_submitted le '{date_range['end']}'")
        
        return " and ".join(filter_parts) if filter_parts else None
    
    def _calculate_recency_boost(self, result: Dict[str, Any]) -> float:
        """Calculate boost factor based on recency"""
        try:
            metadata = result.get('metadata', {})
            date_str = metadata.get('date_submitted')
            if not date_str:
                return 0.0
            
            # Simple recency calculation (boost recent items)
            # This would need proper date parsing in production
            return 0.1  # Placeholder boost
            
        except Exception:
            return 0.0
    
    def _calculate_resolution_quality_boost(self, result: Dict[str, Any]) -> float:
        """Calculate boost based on resolution quality"""
        try:
            metadata = result.get('metadata', {})
            resolution = metadata.get('resolution', '')
            
            # Simple quality heuristics
            if len(resolution) > 100:  # Detailed resolution
                return 0.15
            elif len(resolution) > 50:  # Moderate detail
                return 0.1
            elif resolution:  # Has resolution
                return 0.05
            else:  # No resolution
                return -0.2
                
        except Exception:
            return 0.0
    
    def _calculate_category_boost(self, result: Dict[str, Any], query: str) -> float:
        """Calculate boost based on category relevance to query"""
        try:
            metadata = result.get('metadata', {})
            category = metadata.get('category', '').lower()
            query_lower = query.lower()
            
            # Simple category matching
            if category in query_lower:
                return 0.2
            elif any(word in category for word in query_lower.split()):
                return 0.1
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _extract_text_content(self, item: Dict[str, Any]) -> str:
        """Extract text content from an item for embedding generation"""
        text_parts = []
        
        # Extract common text fields
        for field in ['summary', 'description', 'content', 'title']:
            if item.get(field):
                text_parts.append(str(item[field]))
        
        return " ".join(text_parts)
    
    def _calculate_cosine_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            # Simple cosine similarity calculation
            dot_product = sum(a * b for a, b in zip(vector1, vector2))
            magnitude1 = math.sqrt(sum(a * a for a in vector1))
            magnitude2 = math.sqrt(sum(b * b for b in vector2))
            
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            
            return dot_product / (magnitude1 * magnitude2)
            
        except Exception:
            return 0.0
    
    async def _explain_similarity(self, item1: Dict[str, Any], item2: Dict[str, Any]) -> str:
        """Generate explanation for why two items are similar"""
        # Simple explanation based on common keywords
        text1 = self._extract_text_content(item1).lower()
        text2 = self._extract_text_content(item2).lower()
        
        words1 = set(text1.split())
        words2 = set(text2.split())
        common_words = words1.intersection(words2)
        
        if len(common_words) > 3:
            return f"Similar keywords: {', '.join(list(common_words)[:5])}"
        else:
            return "Semantic similarity detected"
    
    def _generate_relevance_explanation(self, result: Dict[str, Any], query: str) -> str:
        """Generate explanation for result relevance"""
        explanations = []
        
        # Check for direct keyword matches
        content = result.get('content', '').lower()
        query_words = query.lower().split()
        
        matches = [word for word in query_words if word in content]
        if matches:
            explanations.append(f"Matches keywords: {', '.join(matches)}")
        
        # Check score
        score = result.get('score', 0)
        if score > 0.8:
            explanations.append("High semantic similarity")
        elif score > 0.5:
            explanations.append("Moderate semantic similarity")
        
        return "; ".join(explanations) if explanations else "General relevance"
    
    def _generate_cache_key(self, search_type: str, query: str, 
                          filters: Dict[str, Any], max_results: int) -> str:
        """Generate cache key for search results"""
        key_parts = [search_type, query, str(sorted(filters.items())), str(max_results)]
        return "|".join(key_parts)
    
    def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached search result if available and not expired"""
        if cache_key in self.search_cache:
            cached_item = self.search_cache[cache_key]
            if datetime.now().timestamp() - cached_item['timestamp'] < self.cache_ttl:
                cached_item['result']['processing_metadata']['cache_hit'] = True
                return cached_item['result']
            else:
                # Remove expired cache entry
                del self.search_cache[cache_key]
        return None
    
    def _cache_result(self, cache_key: str, result: Dict[str, Any]):
        """Cache search result"""
        self.search_cache[cache_key] = {
            'result': result,
            'timestamp': datetime.now().timestamp()
        }
        
        # Simple cache size management
        if len(self.search_cache) > 100:
            # Remove oldest entries
            oldest_key = min(self.search_cache.keys(), 
                           key=lambda k: self.search_cache[k]['timestamp'])
            del self.search_cache[oldest_key]