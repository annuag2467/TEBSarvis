"""
Pattern Detection Agent for TEBSarvis Multi-Agent System
Performs ML clustering and trend analysis for incident pattern identification.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import numpy as np
from collections import Counter, defaultdict
import statistics

from ..core.base_agent import BaseAgent, AgentCapability
from ..shared.azure_clients import AzureClientManager

class PatternDetectionAgent(BaseAgent):
    """
    Pattern Detection Agent that identifies incident clusters and recurring trends.
    Uses ML clustering, statistical analysis, and AI for pattern recognition.
    """
    
    def __init__(self, agent_id: str = "pattern_detection_agent"):
        capabilities = [
            AgentCapability(
                name="incident_clustering",
                description="Cluster incidents using ML algorithms",
                input_types=["incident_data_batch", "clustering_parameters"],
                output_types=["incident_clusters", "cluster_insights"],
                dependencies=["cosmos_db", "azure_ml"]
            ),
            AgentCapability(
                name="trend_analysis",
                description="Analyze trends in incident patterns over time",
                input_types=["time_series_data", "trend_parameters"],
                output_types=["trend_analysis", "forecasts"],
                dependencies=["cosmos_db"]
            ),
            AgentCapability(
                name="anomaly_detection",
                description="Detect anomalous patterns in incident data",
                input_types=["incident_metrics", "baseline_data"],
                output_types=["anomalies", "anomaly_analysis"],
                dependencies=["azure_ml"]
            ),
            AgentCapability(
                name="pattern_insights",
                description="Generate insights from detected patterns",
                input_types=["pattern_data", "historical_context"],
                output_types=["actionable_insights", "recommendations"],
                dependencies=["azure_openai"]
            ),
            AgentCapability(
                name="correlation_analysis",
                description="Find correlations between different incident attributes",
                input_types=["multi_dimensional_data"],
                output_types=["correlation_matrix", "correlation_insights"],
                dependencies=[]
            )
        ]
        
        super().__init__(agent_id, "pattern_detection", capabilities)
        
        self.azure_manager = AzureClientManager()
        self.analysis_cache = {}  # Cache for expensive computations
        self.cache_duration = 1800  # 30 minutes
        self.min_cluster_size = 3
        self.trend_window_days = 30
        
        # Initialize agent
        asyncio.create_task(self._initialize())
    
    async def _initialize(self):
        """Initialize the pattern detection agent"""
        try:
            await self.azure_manager.initialize()
            self.logger.info("Pattern Detection Agent initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Pattern Detection Agent: {str(e)}")
            raise
    
    def get_capabilities(self) -> List[AgentCapability]:
        """Return agent capabilities"""
        return self.capabilities
    
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process pattern detection tasks.
        
        Args:
            task: Task data containing pattern analysis parameters
            
        Returns:
            Pattern analysis results with insights
        """
        task_type = task.get('type', 'incident_clustering')
        
        if task_type == 'incident_clustering':
            return await self._perform_incident_clustering(task)
        elif task_type == 'trend_analysis':
            return await self._analyze_trends(task)
        elif task_type == 'anomaly_detection':
            return await self._detect_anomalies(task)
        elif task_type == 'pattern_insights':
            return await self._generate_pattern_insights(task)
        elif task_type == 'correlation_analysis':
            return await self._analyze_correlations(task)
        elif task_type == 'comprehensive_analysis':
            return await self._perform_comprehensive_analysis(task)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    async def _perform_incident_clustering(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform incident clustering using ML algorithms.
        
        Args:
            task: Task containing clustering parameters
            
        Returns:
            Dictionary with clustering results and insights
        """
        try:
            time_range = task.get('time_range', {'days': 30})
            clustering_method = task.get('clustering_method', 'semantic')  # semantic, categorical, mixed
            min_cluster_size = task.get('min_cluster_size', self.min_cluster_size)
            
            # Retrieve incident data
            incidents = await self._retrieve_incident_data(time_range)
            
            if len(incidents) < min_cluster_size:
                return {
                    'clusters': [],
                    'total_incidents': len(incidents),
                    'message': 'Insufficient data for clustering',
                    'analysis_metadata': {
                        'agent_id': self.agent_id,
                        'timestamp': datetime.now().isoformat()
                    }
                }
            
            # Perform clustering based on method
            if clustering_method == 'semantic':
                clusters = await self._semantic_clustering(incidents, min_cluster_size)
            elif clustering_method == 'categorical':
                clusters = await self._categorical_clustering(incidents, min_cluster_size)
            elif clustering_method == 'mixed':
                clusters = await self._mixed_clustering(incidents, min_cluster_size)
            else:
                raise ValueError(f"Unknown clustering method: {clustering_method}")
            
            # Generate cluster insights
            cluster_insights = await self._generate_cluster_insights(clusters, incidents)
            
            # Calculate clustering quality metrics
            clustering_metrics = self._calculate_clustering_metrics(clusters, incidents)
            
            return {
                'clusters': clusters,
                'cluster_insights': cluster_insights,
                'clustering_metrics': clustering_metrics,
                'total_incidents': len(incidents),
                'clustered_incidents': sum(len(cluster['incidents']) for cluster in clusters),
                'analysis_metadata': {
                    'agent_id': self.agent_id,
                    'timestamp': datetime.now().isoformat(),
                    'clustering_method': clustering_method,
                    'time_range': time_range
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in incident clustering: {str(e)}")
            raise
    
    async def _analyze_trends(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze trends in incident patterns over time.
        
        Args:
            task: Task containing trend analysis parameters
            
        Returns:
            Dictionary with trend analysis results
        """
        try:
            analysis_period = task.get('analysis_period', {'days': 90})
            trend_types = task.get('trend_types', ['volume', 'category', 'severity', 'resolution_time'])
            granularity = task.get('granularity', 'daily')  # hourly, daily, weekly
            
            # Retrieve time series data
            time_series_data = await self._retrieve_time_series_data(analysis_period, granularity)
            
            trends = {}
            
            # Volume trends
            if 'volume' in trend_types:
                trends['volume'] = await self._analyze_volume_trends(time_series_data)
            
            # Category trends
            if 'category' in trend_types:
                trends['category'] = await self._analyze_category_trends(time_series_data)
            
            # Severity trends
            if 'severity' in trend_types:
                trends['severity'] = await self._analyze_severity_trends(time_series_data)
            
            # Resolution time trends
            if 'resolution_time' in trend_types:
                trends['resolution_time'] = await self._analyze_resolution_time_trends(time_series_data)
            
            # Generate forecasts
            forecasts = await self._generate_trend_forecasts(trends, granularity)
            
            # Create trend summary
            trend_summary = await self._create_trend_summary(trends, forecasts)
            
            return {
                'trends': trends,
                'forecasts': forecasts,
                'trend_summary': trend_summary,
                'data_points': len(time_series_data),
                'analysis_metadata': {
                    'agent_id': self.agent_id,
                    'timestamp': datetime.now().isoformat(),
                    'analysis_period': analysis_period,
                    'granularity': granularity
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in trend analysis: {str(e)}")
            raise
    
    async def _detect_anomalies(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect anomalous patterns in incident data.
        
        Args:
            task: Task containing anomaly detection parameters
            
        Returns:
            Dictionary with detected anomalies
        """
        try:
            detection_window = task.get('detection_window', {'days': 7})
            baseline_period = task.get('baseline_period', {'days': 30})
            sensitivity = task.get('sensitivity', 'medium')  # low, medium, high
            anomaly_types = task.get('anomaly_types', ['volume', 'pattern', 'timing'])
            
            # Get baseline and current data
            baseline_data = await self._retrieve_baseline_data(baseline_period)
            current_data = await self._retrieve_current_data(detection_window)
            
            anomalies = {}
            
            # Volume anomalies
            if 'volume' in anomaly_types:
                anomalies['volume'] = await self._detect_volume_anomalies(
                    baseline_data, current_data, sensitivity
                )
            
            # Pattern anomalies
            if 'pattern' in anomaly_types:
                anomalies['pattern'] = await self._detect_pattern_anomalies(
                    baseline_data, current_data, sensitivity
                )
            
            # Timing anomalies
            if 'timing' in anomaly_types:
                anomalies['timing'] = await self._detect_timing_anomalies(
                    baseline_data, current_data, sensitivity
                )
            
            # Category distribution anomalies
            if 'category' in anomaly_types:
                anomalies['category'] = await self._detect_category_anomalies(
                    baseline_data, current_data, sensitivity
                )
            
            # Calculate anomaly scores
            anomaly_scores = self._calculate_anomaly_scores(anomalies)
            
            # Generate anomaly insights
            anomaly_insights = await self._generate_anomaly_insights(anomalies, anomaly_scores)
            
            return {
                'anomalies': anomalies,
                'anomaly_scores': anomaly_scores,
                'anomaly_insights': anomaly_insights,
                'overall_anomaly_level': self._determine_overall_anomaly_level(anomaly_scores),
                'detection_metadata': {
                    'agent_id': self.agent_id,
                    'timestamp': datetime.now().isoformat(),
                    'sensitivity': sensitivity,
                    'detection_window': detection_window
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in anomaly detection: {str(e)}")
            raise
    
    async def _generate_pattern_insights(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate actionable insights from detected patterns.
        
        Args:
            task: Task containing pattern data for insight generation
            
        Returns:
            Dictionary with actionable insights and recommendations
        """
        try:
            pattern_data = task.get('pattern_data', {})
            insight_types = task.get('insight_types', ['operational', 'strategic', 'preventive'])
            
            insights = {}
            
            # Operational insights
            if 'operational' in insight_types:
                insights['operational'] = await self._generate_operational_insights(pattern_data)
            
            # Strategic insights
            if 'strategic' in insight_types:
                insights['strategic'] = await self._generate_strategic_insights(pattern_data)
            
            # Preventive insights
            if 'preventive' in insight_types:
                insights['preventive'] = await self._generate_preventive_insights(pattern_data)
            
            # Process improvement insights
            if 'process_improvement' in insight_types:
                insights['process_improvement'] = await self._generate_process_insights(pattern_data)
            
            # Generate recommendations
            recommendations = await self._generate_pattern_based_recommendations(insights)
            
            # Prioritize insights
            prioritized_insights = self._prioritize_insights(insights, recommendations)
            
            return {
                'insights': insights,
                'recommendations': recommendations,
                'prioritized_insights': prioritized_insights,
                'insight_metadata': {
                    'agent_id': self.agent_id,
                    'timestamp': datetime.now().isoformat(),
                    'insight_types': insight_types
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating pattern insights: {str(e)}")
            raise
    
    async def _analyze_correlations(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze correlations between different incident attributes.
        
        Args:
            task: Task containing correlation analysis parameters
            
        Returns:
            Dictionary with correlation analysis results
        """
        try:
            data_attributes = task.get('attributes', ['category', 'severity', 'priority', 'resolution_time'])
            correlation_method = task.get('method', 'pearson')  # pearson, spearman, kendall
            time_range = task.get('time_range', {'days': 60})
            
            # Retrieve multi-dimensional data
            incident_data = await self._retrieve_multi_dimensional_data(data_attributes, time_range)
            
            # Calculate correlation matrix
            correlation_matrix = self._calculate_correlation_matrix(incident_data, correlation_method)
            
            # Find significant correlations
            significant_correlations = self._identify_significant_correlations(correlation_matrix)
            
            # Generate correlation insights
            correlation_insights = await self._generate_correlation_insights(
                correlation_matrix, significant_correlations
            )
            
            # Create correlation visualizations data
            visualization_data = self._prepare_correlation_visualizations(correlation_matrix)
            
            return {
                'correlation_matrix': correlation_matrix,
                'significant_correlations': significant_correlations,
                'correlation_insights': correlation_insights,
                'visualization_data': visualization_data,
                'analysis_metadata': {
                    'agent_id': self.agent_id,
                    'timestamp': datetime.now().isoformat(),
                    'method': correlation_method,
                    'attributes_analyzed': data_attributes
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in correlation analysis: {str(e)}")
            raise
    
    async def _perform_comprehensive_analysis(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive pattern analysis combining multiple techniques.
        
        Args:
            task: Task containing comprehensive analysis parameters
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        try:
            analysis_components = task.get('components', ['clustering', 'trends', 'anomalies', 'correlations'])
            time_range = task.get('time_range', {'days': 60})
            
            comprehensive_results = {}
            
            # Run each analysis component
            if 'clustering' in analysis_components:
                clustering_task = {'time_range': time_range, 'clustering_method': 'mixed'}
                comprehensive_results['clustering'] = await self._perform_incident_clustering(clustering_task)
            
            if 'trends' in analysis_components:
                trend_task = {'analysis_period': time_range, 'granularity': 'daily'}
                comprehensive_results['trends'] = await self._analyze_trends(trend_task)
            
            if 'anomalies' in analysis_components:
                anomaly_task = {
                    'detection_window': {'days': 7},
                    'baseline_period': time_range,
                    'sensitivity': 'medium'
                }
                comprehensive_results['anomalies'] = await self._detect_anomalies(anomaly_task)
            
            if 'correlations' in analysis_components:
                correlation_task = {'time_range': time_range}
                comprehensive_results['correlations'] = await self._analyze_correlations(correlation_task)
            
            # Generate integrated insights
            integrated_insights = await self._generate_integrated_insights(comprehensive_results)
            
            # Create executive summary
            executive_summary = await self._create_executive_summary(comprehensive_results, integrated_insights)
            
            return {
                'comprehensive_results': comprehensive_results,
                'integrated_insights': integrated_insights,
                'executive_summary': executive_summary,
                'analysis_metadata': {
                    'agent_id': self.agent_id,
                    'timestamp': datetime.now().isoformat(),
                    'components_analyzed': analysis_components,
                    'time_range': time_range
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive analysis: {str(e)}")
            raise
    
    async def _retrieve_incident_data(self, time_range: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve incident data for the specified time range"""
        try:
            # Calculate date range
            end_date = datetime.now()
            if 'days' in time_range:
                start_date = end_date - timedelta(days=time_range['days'])
            elif 'weeks' in time_range:
                start_date = end_date - timedelta(weeks=time_range['weeks'])
            else:
                start_date = end_date - timedelta(days=30)  # Default
            
            # Query Cosmos DB
            query = """
            SELECT * FROM c 
            WHERE c.date_submitted >= @start_date 
            AND c.date_submitted <= @end_date
            ORDER BY c.date_submitted DESC
            """
            
            parameters = [
                {"name": "@start_date", "value": start_date.strftime('%d-%m-%Y')},
                {"name": "@end_date", "value": end_date.strftime('%d-%m-%Y')}
            ]
            
            incidents = await self.azure_manager.query_incidents(query, parameters)
            
            return incidents
            
        except Exception as e:
            self.logger.error(f"Error retrieving incident data: {str(e)}")
            return []
    
    async def _semantic_clustering(self, incidents: List[Dict[str, Any]], 
                                 min_cluster_size: int) -> List[Dict[str, Any]]:
        """Perform semantic clustering using embeddings"""
        try:
            # Extract text content and generate embeddings
            incident_texts = []
            for incident in incidents:
                text = f"{incident.get('summary', '')} {incident.get('description', '')}"
                incident_texts.append(text)
            
            # Generate embeddings for all incidents
            embeddings = await self.azure_manager.get_embeddings(incident_texts)
            
            # Perform clustering (simplified clustering algorithm)
            clusters = self._simple_embedding_clustering(incidents, embeddings, min_cluster_size)
            
            return clusters
            
        except Exception as e:
            self.logger.error(f"Error in semantic clustering: {str(e)}")
            return []
    
    def _simple_embedding_clustering(self, incidents: List[Dict[str, Any]], 
                                   embeddings: List[List[float]], 
                                   min_cluster_size: int) -> List[Dict[str, Any]]:
        """Simple clustering algorithm using cosine similarity"""
        clusters = []
        used_indices = set()
        
        for i, incident in enumerate(incidents):
            if i in used_indices:
                continue
            
            # Start new cluster
            cluster_incidents = [incident]
            cluster_indices = {i}
            
            # Find similar incidents
            for j, other_incident in enumerate(incidents):
                if j <= i or j in used_indices:
                    continue
                
                # Calculate cosine similarity
                similarity = self._cosine_similarity(embeddings[i], embeddings[j])
                
                if similarity > 0.7:  # Similarity threshold
                    cluster_incidents.append(other_incident)
                    cluster_indices.add(j)
            
            # Add cluster if it meets minimum size
            if len(cluster_incidents) >= min_cluster_size:
                clusters.append({
                    'cluster_id': len(clusters),
                    'incidents': cluster_incidents,
                    'size': len(cluster_incidents),
                    'representative_incident': cluster_incidents[0],
                    'cluster_type': 'semantic'
                })
                used_indices.update(cluster_indices)
        
        return clusters
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            magnitude1 = sum(a * a for a in vec1) ** 0.5
            magnitude2 = sum(b * b for b in vec2) ** 0.5
            
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            
            return dot_product / (magnitude1 * magnitude2)
            
        except Exception:
            return 0.0
    
    async def _categorical_clustering(self, incidents: List[Dict[str, Any]], 
                                    min_cluster_size: int) -> List[Dict[str, Any]]:
        """Perform clustering based on categorical attributes"""
        try:
            # Group by category and severity
            category_groups = defaultdict(list)
            
            for incident in incidents:
                category = incident.get('category', 'Unknown')
                severity = incident.get('severity', 'Unknown')
                key = f"{category}_{severity}"
                category_groups[key].append(incident)
            
            clusters = []
            for key, group_incidents in category_groups.items():
                if len(group_incidents) >= min_cluster_size:
                    category, severity = key.rsplit('_', 1)
                    clusters.append({
                        'cluster_id': len(clusters),
                        'incidents': group_incidents,
                        'size': len(group_incidents),
                        'representative_incident': group_incidents[0],
                        'cluster_type': 'categorical',
                        'cluster_attributes': {
                            'category': category,
                            'severity': severity
                        }
                    })
            
            return clusters
            
        except Exception as e:
            self.logger.error(f"Error in categorical clustering: {str(e)}")
            return []
    
    async def _mixed_clustering(self, incidents: List[Dict[str, Any]], 
                              min_cluster_size: int) -> List[Dict[str, Any]]:
        """Perform mixed clustering combining semantic and categorical approaches"""
        try:
            # First cluster by category
            categorical_clusters = await self._categorical_clustering(incidents, min_cluster_size)
            
            # Then apply semantic clustering within each category cluster
            mixed_clusters = []
            
            for cat_cluster in categorical_clusters:
                cat_incidents = cat_cluster['incidents']
                if len(cat_incidents) >= min_cluster_size * 2:  # Worth sub-clustering
                    semantic_subclusters = await self._semantic_clustering(cat_incidents, min_cluster_size)
                    
                    for subcluster in semantic_subclusters:
                        subcluster['cluster_id'] = len(mixed_clusters)
                        subcluster['cluster_type'] = 'mixed'
                        subcluster['parent_category'] = cat_cluster.get('cluster_attributes', {})
                        mixed_clusters.append(subcluster)
                else:
                    # Keep as categorical cluster
                    cat_cluster['cluster_id'] = len(mixed_clusters)
                    mixed_clusters.append(cat_cluster)
            
            return mixed_clusters
            
        except Exception as e:
            self.logger.error(f"Error in mixed clustering: {str(e)}")
            return []
    
    async def _generate_cluster_insights(self, clusters: List[Dict[str, Any]], 
                                       all_incidents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate insights from clustering results"""
        try:
            if not clusters:
                return {}
            
            insights = {
                'cluster_summary': {
                    'total_clusters': len(clusters),
                    'largest_cluster_size': max(cluster['size'] for cluster in clusters),
                    'average_cluster_size': statistics.mean(cluster['size'] for cluster in clusters),
                    'coverage_percentage': (sum(cluster['size'] for cluster in clusters) / len(all_incidents)) * 100
                },
                'cluster_characteristics': [],
                'recommendations': []
            }
            
            # Analyze each cluster
            for cluster in clusters:
                characteristics = await self._analyze_cluster_characteristics(cluster)
                insights['cluster_characteristics'].append(characteristics)
            
            # Generate recommendations
            insights['recommendations'] = await self._generate_clustering_recommendations(clusters)
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error generating cluster insights: {str(e)}")
            return {}
    
    async def _analyze_cluster_characteristics(self, cluster: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze characteristics of a specific cluster"""
        try:
            incidents = cluster['incidents']
            
            # Category distribution
            categories = Counter(inc.get('category', 'Unknown') for inc in incidents)
            
            # Severity distribution
            severities = Counter(inc.get('severity', 'Unknown') for inc in incidents)
            
            # Time pattern analysis
            times = []
            for inc in incidents:
                date_str = inc.get('date_submitted', '')
                if date_str:
                    try:
                        dt = datetime.strptime(date_str, '%d-%m-%Y %H:%M')
                        times.append(dt)
                    except ValueError:
                        continue
            
            time_patterns = self._analyze_time_patterns(times) if times else {}
            
            # Common keywords
            all_text = ' '.join([
                f"{inc.get('summary', '')} {inc.get('description', '')}" 
                for inc in incidents
            ])
            common_keywords = self._extract_common_keywords(all_text)
            
            return {
                'cluster_id': cluster['cluster_id'],
                'size': cluster['size'],
                'category_distribution': dict(categories.most_common(3)),
                'severity_distribution': dict(severities),
                'time_patterns': time_patterns,
                'common_keywords': common_keywords[:10],
                'cluster_type': cluster.get('cluster_type', 'unknown')
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing cluster characteristics: {str(e)}")
            return {}
    
    def _analyze_time_patterns(self, times: List[datetime]) -> Dict[str, Any]:
        """Analyze temporal patterns in a cluster"""
        if not times:
            return {}
        
        # Day of week distribution
        dow_counts = Counter(dt.strftime('%A') for dt in times)
        
        # Hour distribution
        hour_counts = Counter(dt.hour for dt in times)
        
        # Time span
        min_time = min(times)
        max_time = max(times)
        time_span = max_time - min_time
        
        return {
            'most_common_day': dow_counts.most_common(1)[0][0] if dow_counts else None,
            'peak_hours': [hour for hour, count in hour_counts.most_common(3)],
            'time_span_days': time_span.days,
            'incident_frequency': len(times) / max(time_span.days, 1)
        }
    
    def _extract_common_keywords(self, text: str) -> List[str]:
        """Extract common keywords from cluster text"""
        import re
        
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter out common stop words
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'she', 'use', 'way', 'will'}
        
        filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
        
        # Count and return most common
        word_counts = Counter(filtered_words)
        return [word for word, count in word_counts.most_common(20) if count > 1]
    
    def _calculate_clustering_metrics(self, clusters: List[Dict[str, Any]], 
                                    all_incidents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate clustering quality metrics"""
        try:
            if not clusters:
                return {'coverage': 0.0, 'fragmentation': 1.0}
            
            clustered_count = sum(cluster['size'] for cluster in clusters)
            total_count = len(all_incidents)
            
            coverage = clustered_count / total_count if total_count > 0 else 0.0
            
            # Fragmentation (inverse of average cluster size)
            avg_cluster_size = clustered_count / len(clusters) if clusters else 0
            fragmentation = 1.0 / (1.0 + avg_cluster_size / 10)  # Normalized
            
            # Size distribution variance
            cluster_sizes = [cluster['size'] for cluster in clusters]
            size_variance = statistics.variance(cluster_sizes) if len(cluster_sizes) > 1 else 0
            
            return {
                'coverage': coverage,
                'fragmentation': fragmentation,
                'size_variance': size_variance,
                'cluster_count': len(clusters),
                'average_cluster_size': avg_cluster_size
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating clustering metrics: {str(e)}")
            return {}
    
    async def _retrieve_time_series_data(self, analysis_period: Dict[str, Any], 
                                       granularity: str) -> List[Dict[str, Any]]:
        """Retrieve time series data for trend analysis"""
        try:
            # Calculate date range
            end_date = datetime.now()
            if 'days' in analysis_period:
                start_date = end_date - timedelta(days=analysis_period['days'])
            elif 'weeks' in analysis_period:
                start_date = end_date - timedelta(weeks=analysis_period['weeks'])
            else:
                start_date = end_date - timedelta(days=90)  # Default
            
            # Generate time buckets based on granularity
            time_buckets = self._generate_time_buckets(start_date, end_date, granularity)
            
            # Query incidents and group by time buckets
            query = """
            SELECT * FROM c 
            WHERE c.date_submitted >= @start_date 
            AND c.date_submitted <= @end_date
            ORDER BY c.date_submitted
            """
            
            parameters = [
                {"name": "@start_date", "value": start_date.strftime('%d-%m-%Y')},
                {"name": "@end_date", "value": end_date.strftime('%d-%m-%Y')}
            ]
            
            incidents = await self.azure_manager.query_incidents(query, parameters)
            
            # Group incidents by time buckets
            time_series_data = self._group_incidents_by_time(incidents, time_buckets, granularity)
            
            return time_series_data
            
        except Exception as e:
            self.logger.error(f"Error retrieving time series data: {str(e)}")
            return []
    
    def _generate_time_buckets(self, start_date: datetime, end_date: datetime, 
                             granularity: str) -> List[datetime]:
        """Generate time buckets for the specified granularity"""
        buckets = []
        current = start_date
        
        if granularity == 'hourly':
            delta = timedelta(hours=1)
        elif granularity == 'daily':
            delta = timedelta(days=1)
        elif granularity == 'weekly':
            delta = timedelta(weeks=1)
        else:
            delta = timedelta(days=1)  # Default to daily
        
        while current <= end_date:
            buckets.append(current)
            current += delta
        
        return buckets
    
    def _group_incidents_by_time(self, incidents: List[Dict[str, Any]], 
                               time_buckets: List[datetime], 
                               granularity: str) -> List[Dict[str, Any]]:
        """Group incidents by time buckets"""
        time_series_data = []
        
        for i, bucket_time in enumerate(time_buckets):
            # Calculate bucket end time
            if i < len(time_buckets) - 1:
                next_bucket = time_buckets[i + 1]
            else:
                if granularity == 'hourly':
                    next_bucket = bucket_time + timedelta(hours=1)
                elif granularity == 'daily':
                    next_bucket = bucket_time + timedelta(days=1)
                elif granularity == 'weekly':
                    next_bucket = bucket_time + timedelta(weeks=1)
                else:
                    next_bucket = bucket_time + timedelta(days=1)
            
            # Filter incidents for this time bucket
            bucket_incidents = []
            for incident in incidents:
                try:
                    incident_time = datetime.strptime(incident.get('date_submitted', ''), '%d-%m-%Y %H:%M')
                    if bucket_time <= incident_time < next_bucket:
                        bucket_incidents.append(incident)
                except ValueError:
                    continue
            
            # Create time series data point
            time_series_data.append({
                'timestamp': bucket_time,
                'incidents': bucket_incidents,
                'count': len(bucket_incidents),
                'categories': Counter([inc.get('category', 'Unknown') for inc in bucket_incidents]),
                'severities': Counter([inc.get('severity', 'Unknown') for inc in bucket_incidents]),
                'priorities': Counter([inc.get('priority', 'Unknown') for inc in bucket_incidents])
            })
        
        return time_series_data
    
    async def _analyze_volume_trends(self, time_series_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze volume trends in the time series data"""
        try:
            if not time_series_data:
                return {}
            
            # Extract volume data
            volumes = [point['count'] for point in time_series_data]
            timestamps = [point['timestamp'] for point in time_series_data]
            
            # Calculate basic statistics
            avg_volume = statistics.mean(volumes) if volumes else 0
            max_volume = max(volumes) if volumes else 0
            min_volume = min(volumes) if volumes else 0
            
            # Calculate trend direction
            trend_direction = self._calculate_trend_direction(volumes)
            
            # Find peaks and valleys
            peaks, valleys = self._find_peaks_and_valleys(volumes, timestamps)
            
            # Calculate volatility
            volatility = statistics.stdev(volumes) if len(volumes) > 1 else 0
            
            return {
                'average_volume': avg_volume,
                'max_volume': max_volume,
                'min_volume': min_volume,
                'trend_direction': trend_direction,
                'volatility': volatility,
                'peaks': peaks,
                'valleys': valleys,
                'total_incidents': sum(volumes)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing volume trends: {str(e)}")
            return {}
    
    async def _analyze_category_trends(self, time_series_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze category distribution trends"""
        try:
            category_trends = {}
            
            # Collect all categories
            all_categories = set()
            for point in time_series_data:
                all_categories.update(point['categories'].keys())
            
            # Analyze trend for each category
            for category in all_categories:
                category_volumes = []
                for point in time_series_data:
                    category_volumes.append(point['categories'].get(category, 0))
                
                if sum(category_volumes) > 0:  # Only analyze categories with incidents
                    category_trends[category] = {
                        'total_incidents': sum(category_volumes),
                        'average_per_period': statistics.mean(category_volumes),
                        'trend_direction': self._calculate_trend_direction(category_volumes),
                        'peak_period': self._find_peak_period(category_volumes, time_series_data),
                        'percentage_of_total': (sum(category_volumes) / sum(point['count'] for point in time_series_data)) * 100
                    }
            
            # Find fastest growing and declining categories
            growing_categories = []
            declining_categories = []
            
            for category, trend in category_trends.items():
                if trend['trend_direction'] == 'increasing':
                    growing_categories.append(category)
                elif trend['trend_direction'] == 'decreasing':
                    declining_categories.append(category)
            
            return {
                'category_trends': category_trends,
                'growing_categories': growing_categories,
                'declining_categories': declining_categories,
                'most_common_category': max(category_trends.keys(), 
                                          key=lambda k: category_trends[k]['total_incidents']) 
                                          if category_trends else None
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing category trends: {str(e)}")
            return {}
    
    async def _analyze_severity_trends(self, time_series_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze severity distribution trends"""
        try:
            severity_trends = {}
            
            # Collect all severities
            all_severities = set()
            for point in time_series_data:
                all_severities.update(point['severities'].keys())
            
            # Analyze trend for each severity
            for severity in all_severities:
                severity_volumes = []
                for point in time_series_data:
                    severity_volumes.append(point['severities'].get(severity, 0))
                
                if sum(severity_volumes) > 0:
                    severity_trends[severity] = {
                        'total_incidents': sum(severity_volumes),
                        'average_per_period': statistics.mean(severity_volumes),
                        'trend_direction': self._calculate_trend_direction(severity_volumes),
                        'percentage_of_total': (sum(severity_volumes) / sum(point['count'] for point in time_series_data)) * 100
                    }
            
            return severity_trends
            
        except Exception as e:
            self.logger.error(f"Error analyzing severity trends: {str(e)}")
            return {}
    
    async def _analyze_resolution_time_trends(self, time_series_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze resolution time trends"""
        try:
            resolution_times = []
            
            for point in time_series_data:
                for incident in point['incidents']:
                    resolution_time = self._calculate_resolution_time(incident)
                    if resolution_time is not None:
                        resolution_times.append({
                            'timestamp': point['timestamp'],
                            'resolution_time': resolution_time
                        })
            
            if not resolution_times:
                return {'message': 'No resolution time data available'}
            
            # Group by time periods and calculate averages
            period_averages = []
            for point in time_series_data:
                period_resolution_times = [
                    rt['resolution_time'] for rt in resolution_times 
                    if rt['timestamp'] == point['timestamp']
                ]
                
                if period_resolution_times:
                    avg_resolution_time = statistics.mean(period_resolution_times)
                    period_averages.append(avg_resolution_time)
                else:
                    period_averages.append(0)
            
            # Calculate overall trend
            overall_avg = statistics.mean([rt['resolution_time'] for rt in resolution_times])
            trend_direction = self._calculate_trend_direction(period_averages)
            
            return {
                'overall_average_hours': overall_avg,
                'trend_direction': trend_direction,
                'period_averages': period_averages,
                'total_resolved_incidents': len(resolution_times)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing resolution time trends: {str(e)}")
            return {}
    
    def _calculate_resolution_time(self, incident: Dict[str, Any]) -> Optional[float]:
        """Calculate resolution time in hours for an incident"""
        try:
            date_submitted = incident.get('date_submitted', '')
            resolution_date = incident.get('resolution_date', '')
            
            if not date_submitted or not resolution_date:
                return None
            
            submitted_dt = datetime.strptime(date_submitted, '%d-%m-%Y %H:%M')
            resolved_dt = datetime.strptime(resolution_date, '%d-%m-%Y %H:%M')
            
            resolution_time = (resolved_dt - submitted_dt).total_seconds() / 3600  # Convert to hours
            return resolution_time
            
        except ValueError:
            return None
    
    def _calculate_trend_direction(self, values: List[float]) -> str:
        """Calculate trend direction for a series of values"""
        if len(values) < 2:
            return 'stable'
        
        # Simple linear trend calculation
        n = len(values)
        x_values = list(range(n))
        
        # Calculate slope using least squares
        x_mean = statistics.mean(x_values)
        y_mean = statistics.mean(values)
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)
        
        if denominator == 0:
            return 'stable'
        
        slope = numerator / denominator
        
        if slope > 0.1:
            return 'increasing'
        elif slope < -0.1:
            return 'decreasing'
        else:
            return 'stable'
    
    def _find_peaks_and_valleys(self, values: List[float], 
                               timestamps: List[datetime]) -> Tuple[List[Dict], List[Dict]]:
        """Find peaks and valleys in the data"""
        peaks = []
        valleys = []
        
        if len(values) < 3:
            return peaks, valleys
        
        for i in range(1, len(values) - 1):
            # Peak: higher than both neighbors
            if values[i] > values[i-1] and values[i] > values[i+1]:
                peaks.append({
                    'timestamp': timestamps[i],
                    'value': values[i],
                    'index': i
                })
            
            # Valley: lower than both neighbors
            elif values[i] < values[i-1] and values[i] < values[i+1]:
                valleys.append({
                    'timestamp': timestamps[i],
                    'value': values[i],
                    'index': i
                })
        
        return peaks, valleys
    
    def _find_peak_period(self, values: List[float], 
                         time_series_data: List[Dict[str, Any]]) -> Optional[datetime]:
        """Find the time period with peak values"""
        if not values:
            return None
        
        max_index = values.index(max(values))
        return time_series_data[max_index]['timestamp']
    
    async def _generate_trend_forecasts(self, trends: Dict[str, Any], 
                                      granularity: str) -> Dict[str, Any]:
        """Generate forecasts based on trend analysis"""
        try:
            forecasts = {}
            
            # Volume forecast
            if 'volume' in trends:
                volume_trend = trends['volume']
                forecasts['volume'] = self._generate_volume_forecast(volume_trend, granularity)
            
            # Category forecasts
            if 'category' in trends:
                category_trends = trends['category']
                forecasts['categories'] = self._generate_category_forecasts(category_trends, granularity)
            
            return forecasts
            
        except Exception as e:
            self.logger.error(f"Error generating forecasts: {str(e)}")
            return {}
    
    def _generate_volume_forecast(self, volume_trend: Dict[str, Any], 
                                granularity: str) -> Dict[str, Any]:
        """Generate volume forecast"""
        avg_volume = volume_trend.get('average_volume', 0)
        trend_direction = volume_trend.get('trend_direction', 'stable')
        volatility = volume_trend.get('volatility', 0)
        
        # Simple forecast logic
        if trend_direction == 'increasing':
            forecast_multiplier = 1.1
        elif trend_direction == 'decreasing':
            forecast_multiplier = 0.9
        else:
            forecast_multiplier = 1.0
        
        forecasted_volume = avg_volume * forecast_multiplier
        
        return {
            'forecasted_average': forecasted_volume,
            'confidence': 'high' if volatility < avg_volume * 0.3 else 'medium',
            'trend_direction': trend_direction,
            'forecast_period': f"next_{granularity}_period"
        }
    
    def _generate_category_forecasts(self, category_trends: Dict[str, Any], 
                                   granularity: str) -> Dict[str, Any]:
        """Generate category-specific forecasts"""
        category_forecasts = {}
        
        for category, trend_data in category_trends.items():
            trend_direction = trend_data.get('trend_direction', 'stable')
            avg_volume = trend_data.get('average_per_period', 0)
            
            if trend_direction == 'increasing':
                forecast = avg_volume * 1.15
                risk_level = 'increasing'
            elif trend_direction == 'decreasing':
                forecast = avg_volume * 0.85
                risk_level = 'decreasing'
            else:
                forecast = avg_volume
                risk_level = 'stable'
            
            category_forecasts[category] = {
                'forecasted_volume': forecast,
                'risk_level': risk_level,
                'trend_direction': trend_direction
            }
        
        return category_forecasts
    
    async def _create_trend_summary(self, trends: Dict[str, Any], 
                                  forecasts: Dict[str, Any]) -> Dict[str, Any]:
        """Create executive summary of trend analysis"""
        try:
            summary = {
                'key_findings': [],
                'alerts': [],
                'recommendations': []
            }
            
            # Volume insights
            if 'volume' in trends:
                volume_trend = trends['volume']
                if volume_trend.get('trend_direction') == 'increasing':
                    summary['key_findings'].append("Incident volume is trending upward")
                    summary['alerts'].append("Monitor capacity and resource allocation")
                elif volume_trend.get('trend_direction') == 'decreasing':
                    summary['key_findings'].append("Incident volume is trending downward")
                
                high_volatility = volume_trend.get('volatility', 0) > volume_trend.get('average_volume', 0) * 0.5
                if high_volatility:
                    summary['alerts'].append("High volatility in incident volume detected")
            
            # Category insights
            if 'category' in trends:
                category_data = trends['category']
                growing_categories = category_data.get('growing_categories', [])
                declining_categories = category_data.get('declining_categories', [])
                
                if growing_categories:
                    summary['key_findings'].append(f"Growing incident categories: {', '.join(growing_categories)}")
                    summary['recommendations'].append("Focus preventive measures on growing categories")
                
                if declining_categories:
                    summary['key_findings'].append(f"Declining incident categories: {', '.join(declining_categories)}")
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error creating trend summary: {str(e)}")
            return {}
    
    async def _retrieve_baseline_data(self, baseline_period: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve baseline data for anomaly detection"""
        return await self._retrieve_incident_data(baseline_period)
    
    async def _retrieve_current_data(self, detection_window: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve current data for anomaly detection"""
        return await self._retrieve_incident_data(detection_window)
    
    async def _detect_volume_anomalies(self, baseline_data: List[Dict[str, Any]], 
                                     current_data: List[Dict[str, Any]], 
                                     sensitivity: str) -> Dict[str, Any]:
        """Detect volume anomalies"""
        try:
            baseline_volume = len(baseline_data)
            current_volume = len(current_data)
            
            # Calculate baseline daily average
            baseline_days = 30  # Assuming 30-day baseline
            baseline_daily_avg = baseline_volume / baseline_days
            
            # Calculate current daily average
            current_days = 7  # Assuming 7-day detection window
            current_daily_avg = current_volume / current_days
            
            # Set thresholds based on sensitivity
            thresholds = {
                'low': 2.0,      # 100% increase
                'medium': 1.5,   # 50% increase  
                'high': 1.3      # 30% increase
            }
            
            threshold = thresholds.get(sensitivity, 1.5)
            
            volume_anomaly = {
                'anomaly_detected': current_daily_avg > baseline_daily_avg * threshold,
                'baseline_daily_avg': baseline_daily_avg,
                'current_daily_avg': current_daily_avg,
                'increase_factor': current_daily_avg / baseline_daily_avg if baseline_daily_avg > 0 else 0,
                'threshold_used': threshold,
                'severity': 'high' if current_daily_avg > baseline_daily_avg * 2 else 'medium'
            }
            
            return volume_anomaly
            
        except Exception as e:
            self.logger.error(f"Error detecting volume anomalies: {str(e)}")
            return {}
    
    async def _detect_pattern_anomalies(self, baseline_data: List[Dict[str, Any]], 
                                      current_data: List[Dict[str, Any]], 
                                      sensitivity: str) -> Dict[str, Any]:
        """Detect pattern anomalies"""
        try:
            # Analyze category distributions
            baseline_categories = Counter([inc.get('category', 'Unknown') for inc in baseline_data])
            current_categories = Counter([inc.get('category', 'Unknown') for inc in current_data])
            
            # Normalize to percentages
            baseline_total = sum(baseline_categories.values())
            current_total = sum(current_categories.values())
            
            anomalies = []
            
            for category in set(list(baseline_categories.keys()) + list(current_categories.keys())):
                baseline_pct = (baseline_categories.get(category, 0) / baseline_total * 100) if baseline_total > 0 else 0
                current_pct = (current_categories.get(category, 0) / current_total * 100) if current_total > 0 else 0
                
                # Check for significant changes
                if baseline_pct > 0:
                    change_factor = current_pct / baseline_pct
                    if change_factor > 2.0 or change_factor < 0.5:  # 100% increase or 50% decrease
                        anomalies.append({
                            'category': category,
                            'baseline_percentage': baseline_pct,
                            'current_percentage': current_pct,
                            'change_factor': change_factor,
                            'anomaly_type': 'increase' if change_factor > 1 else 'decrease'
                        })
            
            return {
                'pattern_anomalies': anomalies,
                'anomaly_count': len(anomalies)
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting pattern anomalies: {str(e)}")
            return {}
    
    async def _detect_timing_anomalies(self, baseline_data: List[Dict[str, Any]], 
                                     current_data: List[Dict[str, Any]], 
                                     sensitivity: str) -> Dict[str, Any]:
        """Detect timing anomalies"""
        try:
            # Analyze time distributions
            baseline_hours = []
            current_hours = []
            
            for incident in baseline_data:
                try:
                    dt = datetime.strptime(incident.get('date_submitted', ''), '%d-%m-%Y %H:%M')
                    baseline_hours.append(dt.hour)
                except ValueError:
                    continue
            
            for incident in current_data:
                try:
                    dt = datetime.strptime(incident.get('date_submitted', ''), '%d-%m-%Y %H:%M')
                    current_hours.append(dt.hour)
                except ValueError:
                    continue
            
            if not baseline_hours or not current_hours:
                return {'message': 'Insufficient timing data for analysis'}
            
            # Compare hour distributions
            baseline_hour_dist = Counter(baseline_hours)
            current_hour_dist = Counter(current_hours)
            
            timing_anomalies = []
            
            for hour in range(24):
                baseline_count = baseline_hour_dist.get(hour, 0)
                current_count = current_hour_dist.get(hour, 0)
                
                # Normalize by total incidents
                baseline_pct = (baseline_count / len(baseline_hours)) * 100
                current_pct = (current_count / len(current_hours)) * 100
                
                # Check for unusual timing patterns
                if baseline_pct > 5:  # Only check hours with significant baseline activity
                    if current_pct > baseline_pct * 3:  # 300% increase
                        timing_anomalies.append({
                            'hour': hour,
                            'baseline_percentage': baseline_pct,
                            'current_percentage': current_pct,
                            'anomaly_type': 'unusual_peak'
                        })
            
            return {
                'timing_anomalies': timing_anomalies,
                'anomaly_count': len(timing_anomalies)
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting timing anomalies: {str(e)}")
            return {}
    
    async def _detect_category_anomalies(self, baseline_data: List[Dict[str, Any]], 
                                       current_data: List[Dict[str, Any]], 
                                       sensitivity: str) -> Dict[str, Any]:
        """Detect category distribution anomalies"""
        # This is similar to pattern anomalies but focuses specifically on categories
        return await self._detect_pattern_anomalies(baseline_data, current_data, sensitivity)
    
    def _calculate_anomaly_scores(self, anomalies: Dict[str, Any]) -> Dict[str, float]:
        """Calculate overall anomaly scores"""
        scores = {}
        
        # Volume anomaly score
        if 'volume' in anomalies and anomalies['volume'].get('anomaly_detected'):
            increase_factor = anomalies['volume'].get('increase_factor', 1.0)
            scores['volume'] = min((increase_factor - 1.0) * 100, 100)  # Cap at 100
        else:
            scores['volume'] = 0
        
        # Pattern anomaly score
        if 'pattern' in anomalies:
            pattern_count = anomalies['pattern'].get('anomaly_count', 0)
            scores['pattern'] = min(pattern_count * 25, 100)  # 25 points per anomaly, cap at 100
        else:
            scores['pattern'] = 0
        
        # Timing anomaly score
        if 'timing' in anomalies:
            timing_count = anomalies['timing'].get('anomaly_count', 0)
            scores['timing'] = min(timing_count * 20, 100)  # 20 points per anomaly, cap at 100
        else:
            scores['timing'] = 0
        
        # Overall score (weighted average)
        overall_score = (scores['volume'] * 0.4 + scores['pattern'] * 0.4 + scores['timing'] * 0.2)
        scores['overall'] = overall_score
        
        return scores
    
    async def _generate_anomaly_insights(self, anomalies: Dict[str, Any], 
                                       anomaly_scores: Dict[str, float]) -> Dict[str, Any]:
        """Generate insights from detected anomalies"""
        try:
            insights = {
                'summary': '',
                'key_anomalies': [],
                'recommended_actions': [],
                'risk_assessment': ''
            }
            
            overall_score = anomaly_scores.get('overall', 0)
            
            # Generate summary
            if overall_score > 70:
                insights['summary'] = "Multiple significant anomalies detected requiring immediate attention"
                insights['risk_assessment'] = 'high'
            elif overall_score > 40:
                insights['summary'] = "Moderate anomalies detected, monitoring recommended"
                insights['risk_assessment'] = 'medium'
            else:
                insights['summary'] = "Minor or no anomalies detected, system appears normal"
                insights['risk_assessment'] = 'low'
            
            # Extract key anomalies
            if anomaly_scores.get('volume', 0) > 50:
                insights['key_anomalies'].append("Significant increase in incident volume")
                insights['recommended_actions'].append("Review system capacity and resource allocation")
            
            if anomaly_scores.get('pattern', 0) > 50:
                insights['key_anomalies'].append("Unusual patterns in incident categories")
                insights['recommended_actions'].append("Investigate root causes for pattern changes")
            
            if anomaly_scores.get('timing', 0) > 50:
                insights['key_anomalies'].append("Abnormal timing patterns detected")
                insights['recommended_actions'].append("Review recent system changes or external factors")
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error generating anomaly insights: {str(e)}")
            return {}
    
    def _determine_overall_anomaly_level(self, anomaly_scores: Dict[str, float]) -> str:
        """Determine overall anomaly level"""
        overall_score = anomaly_scores.get('overall', 0)
        
        if overall_score > 70:
            return 'critical'
        elif overall_score > 40:
            return 'warning'
        elif overall_score > 20:
            return 'caution'
        else:
            return 'normal'     