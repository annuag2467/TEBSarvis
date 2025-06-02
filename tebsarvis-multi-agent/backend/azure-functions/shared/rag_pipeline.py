"""
RAG (Retrieval-Augmented Generation) Pipeline for TEBSarvis Multi-Agent System
Implements the complete RAG workflow: Retrieval -> Augmentation -> Generation
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
import json
import re
from dataclasses import dataclass, asdict
from enum import Enum

from azure_clients import AzureClientManager

class RetrievalStrategy(Enum):
    """Different retrieval strategies for the RAG pipeline"""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    VECTOR_ONLY = "vector_only"

class AugmentationLevel(Enum):
    """Levels of context augmentation"""
    MINIMAL = "minimal"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"

@dataclass
class RetrievalContext:
    """Context information for retrieval operations"""
    query: str
    strategy: RetrievalStrategy
    max_results: int = 10
    similarity_threshold: float = 0.3
    filters: Dict[str, Any] = None
    metadata_boost: bool = True
    rerank_results: bool = True

@dataclass
class RetrievedDocument:
    """Represents a retrieved document with metadata"""
    id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    source: str
    highlights: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class AugmentedContext:
    """Augmented context for generation"""
    original_query: str
    retrieved_documents: List[RetrievedDocument]
    augmented_prompt: str
    context_summary: str
    confidence_score: float
    retrieval_metadata: Dict[str, Any]

class RAGPipeline:
    """
    Complete RAG pipeline implementation for intelligent document retrieval and generation.
    Integrates with Azure OpenAI, Cognitive Search, and Cosmos DB.
    """
    
    def __init__(self):
        self.azure_manager = AzureClientManager()
        self.logger = logging.getLogger("rag_pipeline")
        
        # Pipeline configuration
        self.default_max_tokens = 2000
        self.default_temperature = 0.7
        self.context_window_size = 8000  # tokens
        self.max_context_documents = 15
        
        # Augmentation templates
        self.augmentation_templates = {
            "minimal": """Context: {context}
            
Question: {query}
Answer:""",
            
            "standard": """Based on the following relevant information:

{context}

Please answer this question: {query}

Provide a comprehensive answer based on the context above. If the context doesn't contain sufficient information, clearly state what information is missing.""",
            
            "comprehensive": """You are an expert IT support assistant with access to a comprehensive knowledge base.

Relevant Context:
{context}

Question: {query}

Instructions:
1. Analyze the provided context carefully
2. Provide a detailed, accurate answer based on the context
3. Include specific references to relevant information when possible
4. If multiple solutions exist, rank them by effectiveness
5. Indicate confidence level in your response
6. Suggest follow-up actions if appropriate

Answer:"""
        }
    
    async def initialize(self):
        """Initialize the RAG pipeline"""
        try:
            await self.azure_manager.initialize()
            self.logger.info("RAG Pipeline initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize RAG Pipeline: {str(e)}")
            raise
    
    async def process_query(self, query: str, 
                          retrieval_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID,
                          augmentation_level: AugmentationLevel = AugmentationLevel.STANDARD,
                          max_results: int = 10,
                          **kwargs) -> Dict[str, Any]:
        """
        Process a query through the complete RAG pipeline.
        
        Args:
            query: User query
            retrieval_strategy: Strategy for document retrieval
            augmentation_level: Level of context augmentation
            max_results: Maximum number of documents to retrieve
            **kwargs: Additional parameters
            
        Returns:
            Complete RAG pipeline results
        """
        try:
            pipeline_start = datetime.now()
            
            # Step 1: Retrieval
            retrieval_context = RetrievalContext(
                query=query,
                strategy=retrieval_strategy,
                max_results=max_results,
                similarity_threshold=kwargs.get('similarity_threshold', 0.3),
                filters=kwargs.get('filters'),
                metadata_boost=kwargs.get('metadata_boost', True),
                rerank_results=kwargs.get('rerank_results', True)
            )
            
            retrieved_docs = await self._retrieve_documents(retrieval_context)
            
            # Step 2: Augmentation
            augmented_context = await self._augment_context(
                query, retrieved_docs, augmentation_level
            )
            
            # Step 3: Generation
            generated_response = await self._generate_response(
                augmented_context, 
                temperature=kwargs.get('temperature', self.default_temperature),
                max_tokens=kwargs.get('max_tokens', self.default_max_tokens)
            )
            
            pipeline_end = datetime.now()
            processing_time = (pipeline_end - pipeline_start).total_seconds()
            
            return {
                'query': query,
                'response': generated_response['response'],
                'sources': [doc.to_dict() for doc in retrieved_docs],
                'augmented_context': augmented_context.context_summary,
                'confidence_score': augmented_context.confidence_score,
                'retrieval_metadata': {
                    'strategy': retrieval_strategy.value,
                    'documents_retrieved': len(retrieved_docs),
                    'augmentation_level': augmentation_level.value,
                    'processing_time_seconds': processing_time
                },
                'generation_metadata': generated_response.get('metadata', {}),
                'pipeline_metadata': {
                    'timestamp': pipeline_end.isoformat(),
                    'pipeline_version': '1.0.0'
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in RAG pipeline: {str(e)}")
            raise
    
    async def _retrieve_documents(self, context: RetrievalContext) -> List[RetrievedDocument]:
        """
        Retrieve relevant documents based on the retrieval strategy.
        
        Args:
            context: Retrieval context
            
        Returns:
            List of retrieved documents
        """
        try:
            documents = []
            
            if context.strategy == RetrievalStrategy.SEMANTIC:
                documents = await self._semantic_retrieval(context)
            elif context.strategy == RetrievalStrategy.KEYWORD:
                documents = await self._keyword_retrieval(context)
            elif context.strategy == RetrievalStrategy.HYBRID:
                documents = await self._hybrid_retrieval(context)
            elif context.strategy == RetrievalStrategy.VECTOR_ONLY:
                documents = await self._vector_retrieval(context)
            
            # Apply post-retrieval processing
            if context.rerank_results:
                documents = await self._rerank_documents(documents, context.query)
            
            if context.metadata_boost:
                documents = self._apply_metadata_boost(documents, context)
            
            # Filter by similarity threshold
            documents = [
                doc for doc in documents 
                if doc.score >= context.similarity_threshold
            ]
            
            # Limit results
            documents = documents[:context.max_results]
            
            self.logger.info(f"Retrieved {len(documents)} documents using {context.strategy.value} strategy")
            return documents
            
        except Exception as e:
            self.logger.error(f"Error in document retrieval: {str(e)}")
            raise
    
    async def _semantic_retrieval(self, context: RetrievalContext) -> List[RetrievedDocument]:
        """Perform semantic retrieval using embeddings"""
        try:
            # Generate query embedding
            query_embedding = await self.azure_manager.get_embeddings(context.query)
            
            # Perform vector search
            search_results = await self.azure_manager.search_client.vector_search(
                query_vector=query_embedding,
                top_k=context.max_results * 2,  # Get more to allow for filtering
                filters=self._build_odata_filter(context.filters)
            )
            
            return self._convert_search_results(search_results, "semantic")
            
        except Exception as e:
            self.logger.error(f"Error in semantic retrieval: {str(e)}")
            return []
    
    async def _keyword_retrieval(self, context: RetrievalContext) -> List[RetrievedDocument]:
        """Perform keyword-based retrieval"""
        try:
            search_results = await self.azure_manager.search_client.text_search(
                query_text=context.query,
                top_k=context.max_results * 2,
                filters=self._build_odata_filter(context.filters)
            )
            
            return self._convert_search_results(search_results, "keyword")
            
        except Exception as e:
            self.logger.error(f"Error in keyword retrieval: {str(e)}")
            return []
    
    async def _hybrid_retrieval(self, context: RetrievalContext) -> List[RetrievedDocument]:
        """Perform hybrid retrieval combining semantic and keyword search"""
        try:
            # Generate query embedding
            query_embedding = await self.azure_manager.get_embeddings(context.query)
            
            # Perform hybrid search
            search_results = await self.azure_manager.search_client.hybrid_search(
                query_text=context.query,
                query_vector=query_embedding,
                top_k=context.max_results * 2,
                filters=self._build_odata_filter(context.filters)
            )
            
            return self._convert_search_results(search_results, "hybrid")
            
        except Exception as e:
            self.logger.error(f"Error in hybrid retrieval: {str(e)}")
            return []
    
    async def _vector_retrieval(self, context: RetrievalContext) -> List[RetrievedDocument]:
        """Perform pure vector retrieval"""
        return await self._semantic_retrieval(context)
    
    def _convert_search_results(self, search_results: List[Dict[str, Any]], 
                               source: str) -> List[RetrievedDocument]:
        """Convert search results to RetrievedDocument objects"""
        documents = []
        
        for result in search_results:
            doc = RetrievedDocument(
                id=result.get('id', ''),
                content=result.get('content', ''),
                score=result.get('score', 0.0),
                metadata=result.get('metadata', {}),
                source=source,
                highlights=result.get('highlights', {})
            )
            documents.append(doc)
        
        return documents
    
    def _build_odata_filter(self, filters: Optional[Dict[str, Any]]) -> Optional[str]:
        """Build OData filter string from filter dictionary"""
        if not filters:
            return None
        
        filter_parts = []
        
        for key, value in filters.items():
            if key == 'category':
                filter_parts.append(f"metadata/category eq '{value}'")
            elif key == 'severity':
                filter_parts.append(f"metadata/severity eq '{value}'")
            elif key == 'date_range':
                if 'start' in value:
                    filter_parts.append(f"metadata/date_submitted ge '{value['start']}'")
                if 'end' in value:
                    filter_parts.append(f"metadata/date_submitted le '{value['end']}'")
            elif key == 'has_resolution':
                if value:
                    filter_parts.append("metadata/has_resolution eq true")
                else:
                    filter_parts.append("metadata/has_resolution eq false")
        
        return " and ".join(filter_parts) if filter_parts else None
    
    async def _rerank_documents(self, documents: List[RetrievedDocument], 
                               query: str) -> List[RetrievedDocument]:
        """Rerank documents using additional relevance signals"""
        try:
            # Simple reranking based on content quality and metadata
            for doc in documents:
                relevance_boost = 0.0
                
                # Boost documents with resolutions
                if doc.metadata.get('has_resolution'):
                    relevance_boost += 0.1
                
                # Boost based on content length (medium length preferred)
                content_length = len(doc.content)
                if 100 <= content_length <= 1000:
                    relevance_boost += 0.05
                
                # Boost recent documents slightly
                if doc.metadata.get('date_submitted'):
                    # This would need proper date parsing in production
                    relevance_boost += 0.02
                
                # Apply boost
                doc.score = min(doc.score + relevance_boost, 1.0)
            
            # Sort by updated scores
            documents.sort(key=lambda x: x.score, reverse=True)
            return documents
            
        except Exception as e:
            self.logger.error(f"Error in document reranking: {str(e)}")
            return documents
    
    def _apply_metadata_boost(self, documents: List[RetrievedDocument], 
                            context: RetrievalContext) -> List[RetrievedDocument]:
        """Apply metadata-based boosting to document scores"""
        try:
            query_lower = context.query.lower()
            
            for doc in documents:
                boost = 0.0
                
                # Category matching
                category = doc.metadata.get('category', '').lower()
                if category and any(word in category for word in query_lower.split()):
                    boost += 0.15
                
                # Priority/severity matching
                if 'urgent' in query_lower or 'critical' in query_lower:
                    severity = doc.metadata.get('severity', '').lower()
                    priority = doc.metadata.get('priority', '').lower()
                    if 'high' in severity or 'critical' in severity or 'high' in priority:
                        boost += 0.1
                
                # Apply boost
                doc.score = min(doc.score + boost, 1.0)
            
            return documents
            
        except Exception as e:
            self.logger.error(f"Error applying metadata boost: {str(e)}")
            return documents
    
    async def _augment_context(self, query: str, documents: List[RetrievedDocument],
                             augmentation_level: AugmentationLevel) -> AugmentedContext:
        """
        Augment the query with retrieved context.
        
        Args:
            query: Original query
            documents: Retrieved documents
            augmentation_level: Level of augmentation
            
        Returns:
            Augmented context for generation
        """
        try:
            # Prepare context from documents
            context_parts = []
            total_tokens = 0
            used_documents = []
            
            for doc in documents:
                # Estimate token count (rough approximation)
                doc_tokens = len(doc.content.split()) * 1.3
                
                if total_tokens + doc_tokens > self.context_window_size * 0.7:  # Leave room for prompt
                    break
                
                # Format document context
                doc_context = self._format_document_context(doc)
                context_parts.append(doc_context)
                total_tokens += doc_tokens
                used_documents.append(doc)
            
            # Combine context
            combined_context = "\n\n".join(context_parts)
            
            # Generate augmented prompt
            template = self.augmentation_templates.get(
                augmentation_level.value, 
                self.augmentation_templates["standard"]
            )
            
            augmented_prompt = template.format(
                context=combined_context,
                query=query
            )
            
            # Calculate confidence based on retrieval quality
            confidence_score = self._calculate_context_confidence(used_documents, query)
            
            # Create context summary
            context_summary = self._create_context_summary(used_documents)
            
            return AugmentedContext(
                original_query=query,
                retrieved_documents=used_documents,
                augmented_prompt=augmented_prompt,
                context_summary=context_summary,
                confidence_score=confidence_score,
                retrieval_metadata={
                    'documents_used': len(used_documents),
                    'total_documents_retrieved': len(documents),
                    'estimated_tokens': total_tokens,
                    'augmentation_level': augmentation_level.value
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in context augmentation: {str(e)}")
            raise
    
    def _format_document_context(self, doc: RetrievedDocument) -> str:
        """Format a document for inclusion in context"""
        context = f"Source: {doc.metadata.get('category', 'Unknown')}\n"
        
        if doc.metadata.get('date_submitted'):
            context += f"Date: {doc.metadata['date_submitted']}\n"
        
        context += f"Content: {doc.content}"
        
        if doc.metadata.get('resolution'):
            context += f"\nResolution: {doc.metadata['resolution']}"
        
        return context
    
    def _calculate_context_confidence(self, documents: List[RetrievedDocument], 
                                    query: str) -> float:
        """Calculate confidence score for the augmented context"""
        if not documents:
            return 0.0
        
        # Base confidence on average document scores
        avg_score = sum(doc.score for doc in documents) / len(documents)
        
        # Boost confidence for documents with resolutions
        resolution_boost = sum(
            0.1 for doc in documents 
            if doc.metadata.get('has_resolution')
        ) / len(documents)
        
        # Boost confidence for multiple relevant documents
        quantity_boost = min(len(documents) * 0.05, 0.2)
        
        confidence = min(avg_score + resolution_boost + quantity_boost, 1.0)
        return confidence
    
    def _create_context_summary(self, documents: List[RetrievedDocument]) -> str:
        """Create a summary of the retrieved context"""
        if not documents:
            return "No relevant documents found."
        
        categories = set()
        has_resolutions = 0
        
        for doc in documents:
            if doc.metadata.get('category'):
                categories.add(doc.metadata['category'])
            if doc.metadata.get('has_resolution'):
                has_resolutions += 1
        
        summary = f"Found {len(documents)} relevant documents"
        
        if categories:
            summary += f" from categories: {', '.join(categories)}"
        
        if has_resolutions:
            summary += f", {has_resolutions} with resolutions"
        
        return summary
    
    async def _generate_response(self, context: AugmentedContext, 
                               temperature: float = 0.7,
                               max_tokens: int = 2000) -> Dict[str, Any]:
        """
        Generate response using the augmented context.
        
        Args:
            context: Augmented context
            temperature: Generation temperature
            max_tokens: Maximum tokens in response
            
        Returns:
            Generated response with metadata
        """
        try:
            generation_start = datetime.now()
            
            # Prepare messages for chat completion
            messages = [
                {
                    "role": "user",
                    "content": context.augmented_prompt
                }
            ]
            
            # Generate response
            response = await self.azure_manager.get_chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            generation_end = datetime.now()
            generation_time = (generation_end - generation_start).total_seconds()
            
            return {
                'response': response,
                'metadata': {
                    'generation_time_seconds': generation_time,
                    'temperature': temperature,
                    'max_tokens': max_tokens,
                    'context_confidence': context.confidence_score,
                    'documents_used': len(context.retrieved_documents)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in response generation: {str(e)}")
            raise
    
    async def stream_response(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Stream response generation for real-time applications.
        
        Args:
            query: User query
            **kwargs: Additional parameters
            
        Returns:
            Streaming response generator and metadata
        """
        try:
            # Perform retrieval and augmentation
            retrieval_context = RetrievalContext(
                query=query,
                strategy=kwargs.get('strategy', RetrievalStrategy.HYBRID),
                max_results=kwargs.get('max_results', 10)
            )
            
            retrieved_docs = await self._retrieve_documents(retrieval_context)
            augmented_context = await self._augment_context(
                query, retrieved_docs, 
                kwargs.get('augmentation_level', AugmentationLevel.STANDARD)
            )
            
            # Prepare messages
            messages = [
                {
                    "role": "user", 
                    "content": augmented_context.augmented_prompt
                }
            ]
            
            # Return streaming generator and metadata
            return {
                'stream': self.azure_manager.openai_client.get_streaming_completion(
                    messages=messages,
                    temperature=kwargs.get('temperature', 0.7),
                    max_tokens=kwargs.get('max_tokens', 2000)
                ),
                'sources': [doc.to_dict() for doc in retrieved_docs],
                'context_summary': augmented_context.context_summary,
                'confidence_score': augmented_context.confidence_score
            }
            
        except Exception as e:
            self.logger.error(f"Error in streaming response: {str(e)}")
            raise
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and configuration"""
        return {
            'status': 'operational',
            'configuration': {
                'default_max_tokens': self.default_max_tokens,
                'default_temperature': self.default_temperature,
                'context_window_size': self.context_window_size,
                'max_context_documents': self.max_context_documents
            },
            'supported_strategies': [strategy.value for strategy in RetrievalStrategy],
            'supported_augmentation_levels': [level.value for level in AugmentationLevel],
            'timestamp': datetime.now().isoformat()
        }