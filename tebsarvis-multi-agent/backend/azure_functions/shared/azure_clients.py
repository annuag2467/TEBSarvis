"""
Azure Serfrom azure.cosmos.aio import CosmosClient
from azure.cosmos import PartitionKey
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
from openai import OpenAI, AsyncOpenAI
import numpy as np
from ...config.azure_config import get_azure_configents for TEBSarvis Multi-Agent System
Provides unified access to Azure OpenAI, Cosmos DB, and Cognitive Search services.
"""

import os
import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import json

# Azure SDK imports
from azure.cosmos.aio import CosmosClient
from azure.cosmos import PartitionKey
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
from openai import AsyncAzureOpenAI
import numpy as np
from config.azure_config import get_azure_config

class OpenAIClient:

    """
    Async client for OpenAI services.
    Handles chat completions, embeddings, and other AI operations.
    """

    def __init__(self):
        config = get_azure_config()
        openai_config = config.openai
        
        self.client = AsyncOpenAI(
            api_key=openai_config.api_key
        )
        self.embedding_model = openai_config.embedding_model
        self.chat_model = openai_config.chat_model
        
    
    async def get_chat_completion(self, messages: List[Dict[str, str]], 
                                 temperature: float = 0.7, max_tokens: int = 1000,
                                 system_prompt: Optional[str] = None) -> str:
        """
        Get chat completion from Azure OpenAI.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens in response
            system_prompt: Optional system prompt to prepend
            
        Returns:
            Generated response text
        """
        try:
            # Add system prompt if provided
            if system_prompt:
                messages = [{"role": "system", "content": system_prompt}] + messages
            
            response = await self.client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"Chat completion error: {str(e)}")
            raise
    
    async def get_embeddings(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text string or list of texts
            
        Returns:
            Embedding vector(s)
        """
        try:
            # Ensure texts is a list
            if isinstance(texts, str):
                texts = [texts]
                return_single = True
            else:
                return_single = False
            
            response = await self.client.embeddings.create(
                model=self.embedding_model,
                input=texts
            )
            
            embeddings = [item.embedding for item in response.data]
            
            return embeddings[0] if return_single else embeddings
            
        except Exception as e:
            self.logger.error(f"Embedding generation error: {str(e)}")
            raise
    
    async def get_streaming_completion(self, messages: List[Dict[str, str]], 
                                     temperature: float = 0.7, max_tokens: int = 1000):
        """
        Get streaming chat completion from Azure OpenAI.
        
        Args:
            messages: List of message dictionaries
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            
        Yields:
            Response chunks as they arrive
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            
            async for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            self.logger.error(f"Streaming completion error: {str(e)}")
            raise

class AzureCosmosClient:
    """
    Async client for Azure Cosmos DB operations.
    Handles document storage and retrieval for incident data.
    """
    
    def __init__(self):
        self.client = CosmosClient(
            url=os.getenv('COSMOS_DB_URL'),
            credential=os.getenv('COSMOS_DB_KEY')
        )
        self.database_name = os.getenv('COSMOS_DB_DATABASE', 'tebsarvis')
        self.container_name = os.getenv('COSMOS_DB_CONTAINER', 'incidents')
        self.logger = logging.getLogger("azure.cosmos")
        
        # Initialize database and container
        asyncio.create_task(self._initialize_database())
    
    async def _initialize_database(self):
        """Initialize database and container if they don't exist"""
        try:
            # Create database if not exists
            database = await self.client.create_database_if_not_exists(
                id=self.database_name
            )
            
            # Create container if not exists
            await database.create_container_if_not_exists(
                id=self.container_name,
                partition_key=PartitionKey(path="/category"),
                offer_throughput=400
            )
            
            self.logger.info(f"Database '{self.database_name}' and container '{self.container_name}' ready")
            
        except Exception as e:
            self.logger.error(f"Database initialization error: {str(e)}")
            raise
    
    async def create_document(self, document: Dict[str, Any], partition_key: str) -> Dict[str, Any]:
        """
        Create a new document in Cosmos DB.
        
        Args:
            document: Document data to store
            partition_key: Partition key value
            
        Returns:
            Created document with metadata
        """
        try:
            database = self.client.get_database_client(self.database_name)
            container = database.get_container_client(self.container_name)
            
            # Add timestamp if not present
            if 'created_at' not in document:
                document['created_at'] = datetime.now().isoformat()
            
            result = await container.create_item(document)
            self.logger.debug(f"Document created with id: {result['id']}")
            return result
            
        except Exception as e:
            self.logger.error(f"Document creation error: {str(e)}")
            raise
    
    async def get_document(self, document_id: str, partition_key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document by ID and partition key.
        
        Args:
            document_id: Document identifier
            partition_key: Partition key value
            
        Returns:
            Document data or None if not found
        """
        try:
            database = self.client.get_database_client(self.database_name)
            container = database.get_container_client(self.container_name)
            
            result = await container.read_item(
                item=document_id,
                partition_key=partition_key
            )
            return result
            
        except Exception as e:
            self.logger.warning(f"Document not found: {document_id}, {str(e)}")
            return None
    
    async def query_documents(self, query: str, parameters: Optional[List] = None,
                            partition_key: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Query documents using SQL-like syntax.
        
        Args:
            query: SQL query string
            parameters: Query parameters
            partition_key: Optional partition key to limit scope
            
        Returns:
            List of matching documents
        """
        try:
            database = self.client.get_database_client(self.database_name)
            container = database.get_container_client(self.container_name)
            
            query_options = {
                'enable_cross_partition_query': partition_key is None,
                'max_item_count': 100
            }
            
            if partition_key:
                query_options['partition_key'] = partition_key
            
            items = []
            async for item in container.query_items(
                query=query,
                parameters=parameters,
                **query_options
            ):
                items.append(item)
            
            return items
            
        except Exception as e:
            self.logger.error(f"Query error: {str(e)}")
            raise
    
    async def update_document(self, document_id: str, partition_key: str,
                            updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update an existing document.
        
        Args:
            document_id: Document identifier
            partition_key: Partition key value
            updates: Fields to update
            
        Returns:
            Updated document
        """
        try:
            # First get the existing document
            existing_doc = await self.get_document(document_id, partition_key)
            if not existing_doc:
                raise ValueError(f"Document {document_id} not found")
            
            # Apply updates
            existing_doc.update(updates)
            existing_doc['updated_at'] = datetime.now().isoformat()
            
            # Replace the document
            database = self.client.get_database_client(self.database_name)
            container = database.get_container_client(self.container_name)
            
            result = await container.replace_item(
                item=document_id,
                body=existing_doc
            )
            return result
            
        except Exception as e:
            self.logger.error(f"Document update error: {str(e)}")
            raise
    
    async def delete_document(self, document_id: str, partition_key: str) -> bool:
        """
        Delete a document.
        
        Args:
            document_id: Document identifier
            partition_key: Partition key value
            
        Returns:
            True if deleted successfully
        """
        try:
            database = self.client.get_database_client(self.database_name)
            container = database.get_container_client(self.container_name)
            
            await container.delete_item(
                item=document_id,
                partition_key=partition_key
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Document deletion error: {str(e)}")
            return False

class AzureCognitiveSearchClient:
    """
    Async client for Azure Cognitive Search operations.
    Handles vector search and hybrid search capabilities.
    """
    
    def __init__(self):
        self.search_endpoint = os.getenv('SEARCH_SERVICE_ENDPOINT')
        self.search_key = os.getenv('SEARCH_API_KEY')
        self.index_name = os.getenv('SEARCH_INDEX_NAME', 'incidents-index')
        
        self.client = SearchClient(
            endpoint=self.search_endpoint,
            index_name=self.index_name,
            credential=AzureKeyCredential(self.search_key)
        )
        self.logger = logging.getLogger("azure.search")
    
    async def vector_search(self, query_vector: List[float], top_k: int = 10,
                           filters: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            filters: Optional OData filter expression
            
        Returns:
            List of search results with similarity scores
        """
        try:
            vector_query = VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=top_k,
                fields="embedding"
            )
            
            results = await self.client.search(
                search_text="",
                vector_queries=[vector_query],
                filter=filters,
                top=top_k,
                include_total_count=True
            )
            
            search_results = []
            async for result in results:
                search_results.append({
                    'id': result.get('id'),
                    'content': result.get('content'),
                    'metadata': result.get('metadata', {}),
                    'score': result.get('@search.score', 0.0)
                })
            
            return search_results
            
        except Exception as e:
            self.logger.error(f"Vector search error: {str(e)}")
            raise
    
    async def hybrid_search(self, query_text: str, query_vector: List[float],
                           top_k: int = 10, filters: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining text and vector search.
        
        Args:
            query_text: Text query
            query_vector: Query embedding vector
            top_k: Number of results to return
            filters: Optional OData filter expression
            
        Returns:
            List of search results ranked by combined relevance
        """
        try:
            vector_query = VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=top_k,
                fields="embedding"
            )
            
            results = await self.client.search(
                search_text=query_text,
                vector_queries=[vector_query],
                filter=filters,
                top=top_k,
                include_total_count=True
            )
            
            search_results = []
            async for result in results:
                search_results.append({
                    'id': result.get('id'),
                    'content': result.get('content'),
                    'metadata': result.get('metadata', {}),
                    'score': result.get('@search.score', 0.0),
                    'highlights': result.get('@search.highlights', {})
                })
            
            return search_results
            
        except Exception as e:
            self.logger.error(f"Hybrid search error: {str(e)}")
            raise
    
    async def text_search(self, query_text: str, top_k: int = 10,
                         filters: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Perform text-only search.
        
        Args:
            query_text: Text query
            top_k: Number of results to return
            filters: Optional OData filter expression
            
        Returns:
            List of search results
        """
        try:
            results = await self.client.search(
                search_text=query_text,
                filter=filters,
                top=top_k,
                include_total_count=True
            )
            
            search_results = []
            async for result in results:
                search_results.append({
                    'id': result.get('id'),
                    'content': result.get('content'),
                    'metadata': result.get('metadata', {}),
                    'score': result.get('@search.score', 0.0),
                    'highlights': result.get('@search.highlights', {})
                })
            
            return search_results
            
        except Exception as e:
            self.logger.error(f"Text search error: {str(e)}")
            raise
    
    async def index_document(self, document: Dict[str, Any]) -> bool:
        """
        Index a single document.
        
        Args:
            document: Document to index
            
        Returns:
            True if indexed successfully
        """
        try:
            result = await self.client.upload_documents([document])
            return result[0].succeeded
            
        except Exception as e:
            self.logger.error(f"Document indexing error: {str(e)}")
            return False
    
    async def index_documents_batch(self, documents: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Index multiple documents in batch.
        
        Args:
            documents: List of documents to index
            
        Returns:
            Dictionary with success/failure counts
        """
        try:
            results = await self.client.upload_documents(documents)
            
            stats = {'succeeded': 0, 'failed': 0}
            for result in results:
                if result.succeeded:
                    stats['succeeded'] += 1
                else:
                    stats['failed'] += 1
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Batch indexing error: {str(e)}")
            return {'succeeded': 0, 'failed': len(documents)}

class AzureClientManager:
    """
    Unified manager for all Azure service clients.
    Provides a single interface for agents to access Azure services.
    """
    
    def __init__(self):
        self.openai_client = OpenAIClient()
        self.cosmos_client = AzureCosmosClient()
        self.search_client = AzureCognitiveSearchClient()
        self.logger = logging.getLogger("azure.manager")
    
    async def initialize(self):
        """Initialize all Azure clients"""
        try:
            # Initialize Cosmos DB
            await self.cosmos_client._initialize_database()
            self.logger.info("Azure clients initialized successfully")
        except Exception as e:
            self.logger.error(f"Azure client initialization error: {str(e)}")
            raise
    
    # OpenAI convenience methods
    async def get_chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Get chat completion using OpenAI"""
        return await self.openai_client.get_chat_completion(messages, **kwargs)
    
    def validate_azure_configuration() -> Dict[str, bool]:
        """Validate all Azure service configurations"""
        config = get_azure_config()
        return config.validate_configuration()
    
    async def get_embeddings(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Generate embeddings using OpenAI"""
        return await self.openai_client.get_embeddings(texts)
    
    # Cosmos DB convenience methods
    async def store_incident(self, incident_data: Dict[str, Any]) -> Dict[str, Any]:
        """Store incident data in Cosmos DB"""
        return await self.cosmos_client.create_document(
            incident_data, 
            incident_data.get('category', 'general')
        )
    
    async def get_incident(self, incident_id: str, category: str) -> Optional[Dict[str, Any]]:
        """Retrieve incident by ID"""
        return await self.cosmos_client.get_document(incident_id, category)
    
    async def query_incidents(self, query: str, parameters: Optional[List] = None) -> List[Dict[str, Any]]:
        """Query incidents using SQL syntax"""
        return await self.cosmos_client.query_documents(query, parameters)
    
    # Search convenience methods
    async def search_similar_incidents(self, query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for similar incidents using hybrid search"""
        # Generate embedding for the query
        query_embedding = await self.get_embeddings(query_text)
        
        # Perform hybrid search
        return await self.search_client.hybrid_search(
            query_text=query_text,
            query_vector=query_embedding,
            top_k=top_k
        )
    
    async def vector_search_incidents(self, query_vector: List[float], top_k: int = 10, 
                                    filters: Optional[str] = None) -> List[Dict[str, Any]]:
        """Perform vector search on incidents"""
        return await self.search_client.vector_search(query_vector, top_k, filters)
    
    async def index_incident_for_search(self, incident: Dict[str, Any]) -> bool:
        """Index an incident for search"""
        # Prepare document for search index
        search_doc = {
            'id': incident.get('id'),
            'content': f"{incident.get('summary', '')} {incident.get('description', '')}",
            'embedding': incident.get('embedding', []),
            'metadata': {
                'category': incident.get('category'),
                'severity': incident.get('severity'),
                'priority': incident.get('priority'),
                'date_submitted': incident.get('date_submitted'),
                'resolution': incident.get('resolution', '')
            }
        }
        
        return await self.search_client.index_document(search_doc)
    
    # Combined operations
    async def store_and_index_incident(self, incident_data: Dict[str, Any]) -> bool:
        """Store incident in Cosmos DB and index for search"""
        try:
            # Generate embedding if not present
            if 'embedding' not in incident_data:
                text_content = f"{incident_data.get('summary', '')} {incident_data.get('description', '')}"
                incident_data['embedding'] = await self.get_embeddings(text_content)
            
            # Store in Cosmos DB
            stored_incident = await self.store_incident(incident_data)
            
            # Index for search
            search_success = await self.index_incident_for_search(stored_incident)
            
            if search_success:
                self.logger.info(f"Incident {stored_incident['id']} stored and indexed successfully")
                return True
            else:
                self.logger.warning(f"Incident {stored_incident['id']} stored but indexing failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Error storing and indexing incident: {str(e)}")
            return False
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all Azure services"""
        status = {
            'openai': 'unknown',
            'cosmos': 'unknown',
            'search': 'unknown',
            'timestamp': datetime.now().isoformat()
        }
        
        # Test OpenAI
        try:
            await self.openai_client.get_embeddings("health check")
            status['openai'] = 'healthy'
        except Exception as e:
            status['openai'] = f'error: {str(e)}'
        
        # Test Cosmos DB
        try:
            await self.cosmos_client.query_documents("SELECT TOP 1 * FROM c")
            status['cosmos'] = 'healthy'
        except Exception as e:
            status['cosmos'] = f'error: {str(e)}'
        
        # Test Search
        try:
            await self.search_client.text_search("health check", top_k=1)
            status['search'] = 'healthy'
        except Exception as e:
            status['search'] = f'error: {str(e)}'
        
        return status


    async def search_with_filters(self, query_text: str, filters: Dict[str, Any], 
                                top_k: int = 10) -> List[Dict[str, Any]]:
        """Search with additional filters"""
        filter_string = self._build_odata_filters(filters)
        return await self.search_client.text_search(query_text, top_k, filter_string)
    
    def _build_odata_filters(self, filters: Dict[str, Any]) -> Optional[str]:
        """Build OData filter string from dictionary"""
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
    
    async def get_incident_metrics(self, time_range: Dict[str, Any]) -> Dict[str, Any]:
        """Get incident metrics for the specified time range"""
        try:
            # Calculate date range
            from datetime import datetime, timedelta
            end_date = datetime.now()
            if 'days' in time_range:
                start_date = end_date - timedelta(days=time_range['days'])
            else:
                start_date = end_date - timedelta(days=30)
            
            # Query for metrics
            query = """
            SELECT 
                COUNT(1) as total_count,
                c.category,
                c.severity,
                c.priority
            FROM c 
            WHERE c.date_submitted >= @start_date 
            AND c.date_submitted <= @end_date
            GROUP BY c.category, c.severity, c.priority
            """
            
            parameters = [
                {"name": "@start_date", "value": start_date.strftime('%d-%m-%Y')},
                {"name": "@end_date", "value": end_date.strftime('%d-%m-%Y')}
            ]
            
            results = await self.query_incidents(query, parameters)
            
            # Process results into metrics
            metrics = {
                'total_incidents': sum(r.get('total_count', 0) for r in results),
                'category_breakdown': {},
                'severity_breakdown': {},
                'time_range': time_range
            }
            
            for result in results:
                category = result.get('category', 'Unknown')
                severity = result.get('severity', 'Unknown')
                count = result.get('total_count', 0)
                
                if category not in metrics['category_breakdown']:
                    metrics['category_breakdown'][category] = 0
                metrics['category_breakdown'][category] += count
                
                if severity not in metrics['severity_breakdown']:
                    metrics['severity_breakdown'][severity] = 0
                metrics['severity_breakdown'][severity] += count
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error getting incident metrics: {str(e)}")
            return {'total_incidents': 0, 'category_breakdown': {}, 'severity_breakdown': {}}