"""
Knowledge Base Builder for TEBSarvis Multi-Agent System
Builds and manages the vector search index using Azure Cognitive Search and FAISS.
"""

import asyncio
import logging
import json
import os
import sys
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
import uuid
from pathlib import Path
import pickle
import time
from dataclasses import dataclass, asdict
from enum import Enum

# Add the backend path to sys.path to import shared utilities
backend_path = os.path.join(os.path.dirname(__file__), '..', 'azure-functions', 'shared')
sys.path.append(backend_path)

from azure_clients import AzureClientManager
from agent_utils import TextProcessor, CacheManager, MetricsCollector, batch_list

# Optional FAISS import (install with: pip install faiss-cpu or faiss-gpu)
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available. Install with: pip install faiss-cpu")

class IndexType(Enum):
    """Supported index types"""
    AZURE_SEARCH = "azure_search"
    FAISS_FLAT = "faiss_flat"
    FAISS_IVF = "faiss_ivf"
    FAISS_HNSW = "faiss_hnsw"
    HYBRID = "hybrid"  # Both Azure Search and FAISS

class IndexStrategy(Enum):
    """Index building strategies"""
    FULL_REBUILD = "full_rebuild"
    INCREMENTAL = "incremental"
    BATCH_UPDATE = "batch_update"

@dataclass
class IndexConfig:
    """Configuration for index building"""
    index_type: IndexType
    strategy: IndexStrategy
    index_name: str
    batch_size: int = 100
    vector_dimension: int = 1536
    similarity_metric: str = "cosine"  # cosine, euclidean, dot_product
    
    # FAISS-specific configurations
    faiss_nlist: int = 100  # Number of clusters for IVF
    faiss_nprobe: int = 10  # Number of clusters to search
    faiss_ef_construction: int = 200  # HNSW construction parameter
    faiss_ef_search: int = 100  # HNSW search parameter
    
    # Azure Search specific configurations
    azure_search_replicas: int = 1
    azure_search_partitions: int = 1
    azure_search_tier: str = "standard"

@dataclass
class IndexMetadata:
    """Metadata for the knowledge base index"""
    index_id: str
    index_type: str
    vector_dimension: int
    total_documents: int
    created_at: str
    updated_at: str
    version: str
    configuration: Dict[str, Any]
    statistics: Dict[str, Any]

class KnowledgeBaseBuilder:
    """
    Builds and manages knowledge base indexes for incident data.
    Supports both Azure Cognitive Search and FAISS for vector similarity search.
    """
    
    def __init__(self, config: IndexConfig):
        self.config = config
        self.logger = logging.getLogger("knowledge_base_builder")
        self.azure_manager = AzureClientManager()
        self.text_processor = TextProcessor()
        self.metrics = MetricsCollector()
        
        # Index storage
        self.faiss_index = None
        self.document_metadata = {}  # Maps vector index to document metadata
        self.id_to_index = {}  # Maps document ID to vector index
        
        # Processing statistics
        self.build_stats = {
            'total_documents': 0,
            'successfully_indexed': 0,
            'failed_to_index': 0,
            'index_size_bytes': 0,
            'build_time_seconds': 0.0,
            'index_type': config.index_type.value
        }
        
        # Index paths
        self.index_base_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'indexes')
        os.makedirs(self.index_base_path, exist_ok=True)
    
    async def initialize(self):
        """Initialize the knowledge base builder"""
        try:
            await self.azure_manager.initialize()
            
            if self.config.index_type in [IndexType.FAISS_FLAT, IndexType.FAISS_IVF, IndexType.FAISS_HNSW, IndexType.HYBRID]:
                if not FAISS_AVAILABLE:
                    raise ImportError("FAISS is required but not installed")
            
            self.logger.info(f"Knowledge Base Builder initialized with index type: {self.config.index_type.value}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Knowledge Base Builder: {str(e)}")
            raise
    
    async def build_index(self, embeddings_file: str, output_path: str = None) -> Dict[str, Any]:
        """
        Build the knowledge base index from embeddings file.
        
        Args:
            embeddings_file: Path to embeddings JSON file
            output_path: Optional path to save the index
            
        Returns:
            Index building results
        """
        try:
            start_time = time.time()
            self.logger.info(f"Starting index build from {embeddings_file}")
            
            # Load embeddings data
            embeddings_data = await self._load_embeddings_data(embeddings_file)
            
            # Validate embeddings
            validated_data = await self._validate_embeddings(embeddings_data)
            
            # Build index based on type
            if self.config.index_type == IndexType.AZURE_SEARCH:
                index_result = await self._build_azure_search_index(validated_data)
            elif self.config.index_type in [IndexType.FAISS_FLAT, IndexType.FAISS_IVF, IndexType.FAISS_HNSW]:
                index_result = await self._build_faiss_index(validated_data)
            elif self.config.index_type == IndexType.HYBRID:
                index_result = await self._build_hybrid_index(validated_data)
            else:
                raise ValueError(f"Unsupported index type: {self.config.index_type}")
            
            # Save index if output path provided
            if output_path:
                await self._save_index(output_path)
            
            # Calculate final statistics
            build_time = time.time() - start_time
            self.build_stats['build_time_seconds'] = build_time
            
            # Create index metadata
            metadata = IndexMetadata(
                index_id=str(uuid.uuid4()),
                index_type=self.config.index_type.value,
                vector_dimension=self.config.vector_dimension,
                total_documents=self.build_stats['successfully_indexed'],
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
                version="1.0.0",
                configuration=asdict(self.config),
                statistics=self.build_stats
            )
            
            self.logger.info(f"Index build completed in {build_time:.2f} seconds")
            
            return {
                'index_metadata': asdict(metadata),
                'build_statistics': self.build_stats,
                'index_result': index_result,
                'output_path': output_path
            }
            
        except Exception as e:
            self.logger.error(f"Error building index: {str(e)}")
            raise
    
    async def _load_embeddings_data(self, embeddings_file: str) -> List[Dict[str, Any]]:
        """Load embeddings data from file"""
        try:
            if not os.path.exists(embeddings_file):
                raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")
            
            # Determine file format and load accordingly
            if embeddings_file.endswith('.json'):
                with open(embeddings_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Handle different JSON structures
                if isinstance(data, dict) and 'embeddings' in data:
                    embeddings_data = data['embeddings']
                elif isinstance(data, list):
                    embeddings_data = data
                else:
                    raise ValueError("Invalid JSON structure for embeddings file")
                    
            elif embeddings_file.endswith(('.pkl', '.pickle')):
                with open(embeddings_file, 'rb') as f:
                    embeddings_data = pickle.load(f)
                    
            elif embeddings_file.endswith('.npz'):
                data = np.load(embeddings_file, allow_pickle=True)
                # Convert NumPy format back to list of dicts
                vectors = data['vectors']
                metadata = data['metadata']
                
                embeddings_data = []
                for i, vector in enumerate(vectors):
                    embedding_dict = {
                        'id': metadata[i]['id'],
                        'embeddings': vector.tolist(),
                        'embedding_dimension': len(vector),
                        'model_used': metadata[i].get('model_used', 'unknown'),
                        'created_at': metadata[i].get('created_at', datetime.now().isoformat())
                    }
                    embeddings_data.append(embedding_dict)
            else:
                raise ValueError(f"Unsupported file format: {embeddings_file}")
            
            self.build_stats['total_documents'] = len(embeddings_data)
            self.logger.info(f"Loaded {len(embeddings_data)} embeddings from {embeddings_file}")
            
            return embeddings_data
            
        except Exception as e:
            self.logger.error(f"Error loading embeddings data: {str(e)}")
            raise
    
    async def _validate_embeddings(self, embeddings_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and clean embeddings data"""
        try:
            validated_data = []
            
            for idx, embedding_dict in enumerate(embeddings_data):
                try:
                    # Check required fields
                    if 'id' not in embedding_dict:
                        embedding_dict['id'] = f"doc_{idx}"
                    
                    if 'embeddings' not in embedding_dict:
                        self.logger.warning(f"No embeddings found for document {embedding_dict['id']}")
                        self.build_stats['failed_to_index'] += 1
                        continue
                    
                    embeddings = embedding_dict['embeddings']
                    
                    # Handle different embedding formats
                    if isinstance(embeddings, dict):
                        # Multi-vector embeddings - use the first one or concatenate
                        if self.config.index_type == IndexType.AZURE_SEARCH:
                            # For Azure Search, use the first vector
                            first_field = list(embeddings.keys())[0]
                            embeddings = embeddings[first_field]
                        else:
                            # For FAISS, concatenate all vectors
                            embeddings = np.concatenate(list(embeddings.values())).tolist()
                    
                    # Validate vector dimension
                    if len(embeddings) != self.config.vector_dimension:
                        self.logger.warning(f"Dimension mismatch for document {embedding_dict['id']}: expected {self.config.vector_dimension}, got {len(embeddings)}")
                        # Try to pad or truncate
                        if len(embeddings) < self.config.vector_dimension:
                            embeddings.extend([0.0] * (self.config.vector_dimension - len(embeddings)))
                        else:
                            embeddings = embeddings[:self.config.vector_dimension]
                    
                    # Check for invalid values
                    embeddings_array = np.array(embeddings)
                    if not np.all(np.isfinite(embeddings_array)):
                        self.logger.warning(f"Invalid values in embeddings for document {embedding_dict['id']}")
                        self.build_stats['failed_to_index'] += 1
                        continue
                    
                    # Update the embedding in the dict
                    embedding_dict['embeddings'] = embeddings
                    validated_data.append(embedding_dict)
                    
                except Exception as e:
                    self.logger.error(f"Error validating embedding {idx}: {str(e)}")
                    self.build_stats['failed_to_index'] += 1
                    continue
            
            self.logger.info(f"Validated {len(validated_data)} embeddings")
            return validated_data
            
        except Exception as e:
            self.logger.error(f"Error validating embeddings: {str(e)}")
            raise
    
    async def _build_azure_search_index(self, embeddings_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build Azure Cognitive Search index"""
        try:
            self.logger.info("Building Azure Cognitive Search index")
            
            # Create search index schema if it doesn't exist
            await self._create_azure_search_schema()
            
            # Prepare documents for indexing
            search_documents = []
            
            for embedding_dict in embeddings_data:
                try:
                    # Create search document
                    search_doc = {
                        'id': embedding_dict['id'],
                        'embedding': embedding_dict['embeddings'],
                        'content': embedding_dict.get('text_content', ''),
                        'metadata': {
                            'model_used': embedding_dict.get('model_used', 'unknown'),
                            'created_at': embedding_dict.get('created_at', datetime.now().isoformat()),
                            'embedding_dimension': len(embedding_dict['embeddings'])
                        }
                    }
                    
                    # Add original incident data if available
                    if 'original_incident' in embedding_dict:
                        incident = embedding_dict['original_incident']
                        search_doc['metadata'].update({
                            'category': incident.get('category'),
                            'severity': incident.get('severity'),
                            'priority': incident.get('priority'),
                            'date_submitted': incident.get('date_submitted'),
                            'has_resolution': bool(incident.get('resolution'))
                        })
                        
                        # Add searchable text content
                        text_parts = []
                        for field in ['summary', 'description', 'resolution']:
                            if incident.get(field):
                                text_parts.append(str(incident[field]))
                        search_doc['content'] = ' '.join(text_parts)
                    
                    search_documents.append(search_doc)
                    
                except Exception as e:
                    self.logger.error(f"Error preparing document {embedding_dict['id']} for Azure Search: {str(e)}")
                    self.build_stats['failed_to_index'] += 1
                    continue
            
            # Index documents in batches
            indexed_count = 0
            batches = batch_list(search_documents, self.config.batch_size)
            
            for batch_idx, batch in enumerate(batches):
                try:
                    self.logger.debug(f"Indexing batch {batch_idx + 1}/{len(batches)}")
                    
                    result = await self.azure_manager.search_client.index_documents_batch(batch)
                    indexed_count += result['succeeded']
                    
                    if result['failed'] > 0:
                        self.logger.warning(f"Failed to index {result['failed']} documents in batch {batch_idx}")
                        self.build_stats['failed_to_index'] += result['failed']
                    
                except Exception as e:
                    self.logger.error(f"Error indexing batch {batch_idx}: {str(e)}")
                    self.build_stats['failed_to_index'] += len(batch)
                    continue
            
            self.build_stats['successfully_indexed'] = indexed_count
            
            return {
                'index_type': 'azure_search',
                'index_name': self.config.index_name,
                'documents_indexed': indexed_count,
                'batches_processed': len(batches)
            }
            
        except Exception as e:
            self.logger.error(f"Error building Azure Search index: {str(e)}")
            raise
    
    async def _create_azure_search_schema(self):
        """Create Azure Search index schema"""
        try:
            # This would typically involve calling Azure Search management API
            # For now, we assume the index schema exists
            # In production, you would create the index with proper schema definition
            self.logger.info(f"Using existing Azure Search index: {self.config.index_name}")
            
        except Exception as e:
            self.logger.error(f"Error creating Azure Search schema: {str(e)}")
            raise
    
    async def _build_faiss_index(self, embeddings_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build FAISS vector index"""
        try:
            self.logger.info(f"Building FAISS index of type: {self.config.index_type.value}")
            
            # Extract vectors and metadata
            vectors = []
            metadata = []
            
            for embedding_dict in embeddings_data:
                try:
                    vectors.append(embedding_dict['embeddings'])
                    
                    # Store metadata
                    doc_metadata = {
                        'id': embedding_dict['id'],
                        'model_used': embedding_dict.get('model_used', 'unknown'),
                        'created_at': embedding_dict.get('created_at', datetime.now().isoformat())
                    }
                    
                    # Add original incident data if available
                    if 'original_incident' in embedding_dict:
                        doc_metadata['original_incident'] = embedding_dict['original_incident']
                    
                    metadata.append(doc_metadata)
                    
                except Exception as e:
                    self.logger.error(f"Error preparing vector for document {embedding_dict['id']}: {str(e)}")
                    self.build_stats['failed_to_index'] += 1
                    continue
            
            if not vectors:
                raise ValueError("No valid vectors found for indexing")
            
            # Convert to numpy array
            vectors_array = np.array(vectors, dtype=np.float32)
            
            # Create FAISS index based on type
            if self.config.index_type == IndexType.FAISS_FLAT:
                self.faiss_index = self._create_faiss_flat_index(vectors_array)
            elif self.config.index_type == IndexType.FAISS_IVF:
                self.faiss_index = self._create_faiss_ivf_index(vectors_array)
            elif self.config.index_type == IndexType.FAISS_HNSW:
                self.faiss_index = self._create_faiss_hnsw_index(vectors_array)
            
            # Store metadata
            self.document_metadata = {i: metadata[i] for i in range(len(metadata))}
            self.id_to_index = {metadata[i]['id']: i for i in range(len(metadata))}
            
            self.build_stats['successfully_indexed'] = len(vectors)
            
            return {
                'index_type': 'faiss',
                'faiss_index_type': self.config.index_type.value,
                'vectors_indexed': len(vectors),
                'index_size': self.faiss_index.ntotal,
                'dimension': self.config.vector_dimension
            }
            
        except Exception as e:
            self.logger.error(f"Error building FAISS index: {str(e)}")
            raise
    
    def _create_faiss_flat_index(self, vectors: np.ndarray) -> faiss.Index:
        """Create FAISS Flat (brute force) index"""
        try:
            if self.config.similarity_metric == "cosine":
                # Normalize vectors for cosine similarity
                faiss.normalize_L2(vectors)
                index = faiss.IndexFlatIP(self.config.vector_dimension)  # Inner product for normalized vectors
            elif self.config.similarity_metric == "euclidean":
                index = faiss.IndexFlatL2(self.config.vector_dimension)
            else:
                index = faiss.IndexFlatIP(self.config.vector_dimension)  # Default to inner product
            
            index.add(vectors)
            self.logger.info(f"Created FAISS Flat index with {index.ntotal} vectors")
            
            return index
            
        except Exception as e:
            self.logger.error(f"Error creating FAISS Flat index: {str(e)}")
            raise
    
    def _create_faiss_ivf_index(self, vectors: np.ndarray) -> faiss.Index:
        """Create FAISS IVF (Inverted File) index"""
        try:
            # Create quantizer
            if self.config.similarity_metric == "cosine":
                faiss.normalize_L2(vectors)
                quantizer = faiss.IndexFlatIP(self.config.vector_dimension)
                index = faiss.IndexIVFFlat(quantizer, self.config.vector_dimension, 
                                         self.config.faiss_nlist, faiss.METRIC_INNER_PRODUCT)
            else:
                quantizer = faiss.IndexFlatL2(self.config.vector_dimension)
                index = faiss.IndexIVFFlat(quantizer, self.config.vector_dimension, 
                                         self.config.faiss_nlist, faiss.METRIC_L2)
            
            # Train the index
            index.train(vectors)
            index.add(vectors)
            
            # Set search parameters
            index.nprobe = self.config.faiss_nprobe
            
            self.logger.info(f"Created FAISS IVF index with {index.ntotal} vectors, {self.config.faiss_nlist} clusters")
            
            return index
            
        except Exception as e:
            self.logger.error(f"Error creating FAISS IVF index: {str(e)}")
            raise
    
    def _create_faiss_hnsw_index(self, vectors: np.ndarray) -> faiss.Index:
        """Create FAISS HNSW (Hierarchical Navigable Small World) index"""
        try:
            # HNSW index
            index = faiss.IndexHNSWFlat(self.config.vector_dimension, 32)  # 32 is M parameter
            
            # Set construction parameters
            index.hnsw.efConstruction = self.config.faiss_ef_construction
            index.hnsw.efSearch = self.config.faiss_ef_search
            
            if self.config.similarity_metric == "cosine":
                faiss.normalize_L2(vectors)
            
            index.add(vectors)
            
            self.logger.info(f"Created FAISS HNSW index with {index.ntotal} vectors")
            
            return index
            
        except Exception as e:
            self.logger.error(f"Error creating FAISS HNSW index: {str(e)}")
            raise
    
    async def _build_hybrid_index(self, embeddings_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build hybrid index using both Azure Search and FAISS"""
        try:
            self.logger.info("Building hybrid index (Azure Search + FAISS)")
            
            # Build both indexes
            azure_result = await self._build_azure_search_index(embeddings_data)
            faiss_result = await self._build_faiss_index(embeddings_data)
            
            return {
                'index_type': 'hybrid',
                'azure_search': azure_result,
                'faiss': faiss_result,
                'total_documents': self.build_stats['successfully_indexed']
            }
            
        except Exception as e:
            self.logger.error(f"Error building hybrid index: {str(e)}")
            raise
    
    async def _save_index(self, output_path: str):
        """Save the built index to disk"""
        try:
            os.makedirs(output_path, exist_ok=True)
            
            # Save based on index type
            if self.config.index_type in [IndexType.FAISS_FLAT, IndexType.FAISS_IVF, IndexType.FAISS_HNSW]:
                await self._save_faiss_index(output_path)
            elif self.config.index_type == IndexType.HYBRID:
                await self._save_faiss_index(output_path)
                # Azure Search index is already persisted in the cloud
            
            # Save configuration and metadata
            config_path = os.path.join(output_path, 'config.json')
            with open(config_path, 'w') as f:
                json.dump(asdict(self.config), f, indent=2, default=str)
            
            stats_path = os.path.join(output_path, 'build_stats.json')
            with open(stats_path, 'w') as f:
                json.dump(self.build_stats, f, indent=2)
            
            self.logger.info(f"Index saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving index: {str(e)}")
            raise
    
    async def _save_faiss_index(self, output_path: str):
        """Save FAISS index and metadata"""
        try:
            if self.faiss_index is None:
                raise ValueError("No FAISS index to save")
            
            # Save FAISS index
            index_path = os.path.join(output_path, 'faiss_index.bin')
            faiss.write_index(self.faiss_index, index_path)
            
            # Save metadata
            metadata_path = os.path.join(output_path, 'document_metadata.pkl')
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.document_metadata, f)
            
            # Save ID mapping
            id_mapping_path = os.path.join(output_path, 'id_to_index.pkl')
            with open(id_mapping_path, 'wb') as f:
                pickle.dump(self.id_to_index, f)
            
            self.logger.info(f"FAISS index saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving FAISS index: {str(e)}")
            raise
    
    async def load_index(self, index_path: str) -> Dict[str, Any]:
        """Load a previously built index"""
        try:
            self.logger.info(f"Loading index from {index_path}")
            
            # Load configuration
            config_path = os.path.join(index_path, 'config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                    # Update current config
                    for key, value in config_data.items():
                        if hasattr(self.config, key):
                            setattr(self.config, key, value)
            
            # Load based on index type
            if self.config.index_type in [IndexType.FAISS_FLAT, IndexType.FAISS_IVF, IndexType.FAISS_HNSW, IndexType.HYBRID]:
                await self._load_faiss_index(index_path)
            
            # Load build statistics
            stats_path = os.path.join(index_path, 'build_stats.json')
            if os.path.exists(stats_path):
                with open(stats_path, 'r') as f:
                    self.build_stats = json.load(f)
            
            self.logger.info("Index loaded successfully")
            
            return {
                'status': 'loaded',
                'index_type': self.config.index_type.value,
                'total_documents': self.build_stats.get('successfully_indexed', 0)
            }
            
        except Exception as e:
            self.logger.error(f"Error loading index: {str(e)}")
            raise
    
    async def _load_faiss_index(self, index_path: str):
        """Load FAISS index and metadata"""
        try:
            # Load FAISS index
            index_file = os.path.join(index_path, 'faiss_index.bin')
            if os.path.exists(index_file):
                self.faiss_index = faiss.read_index(index_file)
                self.logger.info(f"Loaded FAISS index with {self.faiss_index.ntotal} vectors")
            
            # Load metadata
            metadata_file = os.path.join(index_path, 'document_metadata.pkl')
            if os.path.exists(metadata_file):
                with open(metadata_file, 'rb') as f:
                    self.document_metadata = pickle.load(f)
            
            # Load ID mapping
            id_mapping_file = os.path.join(index_path, 'id_to_index.pkl')
            if os.path.exists(id_mapping_file):
                with open(id_mapping_file, 'rb') as f:
                    self.id_to_index = pickle.load(f)
            
        except Exception as e:
            self.logger.error(f"Error loading FAISS index: {str(e)}")
            raise
    
    async def search_index(self, query_vector: List[float], top_k: int = 10, 
                          index_type: str = None) -> List[Dict[str, Any]]:
        """
        Search the built index.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            index_type: Specific index to search ('azure' or 'faiss')
            
        Returns:
            List of search results
        """
        try:
            if index_type == 'azure' or (index_type is None and self.config.index_type == IndexType.AZURE_SEARCH):
                return await self._search_azure_index(query_vector, top_k)
            elif index_type == 'faiss' or (index_type is None and self.config.index_type in [IndexType.FAISS_FLAT, IndexType.FAISS_IVF, IndexType.FAISS_HNSW]):
                return await self._search_faiss_index(query_vector, top_k)
            elif self.config.index_type == IndexType.HYBRID:
                # Search both and combine results
                azure_results = await self._search_azure_index(query_vector, top_k)
                faiss_results = await self._search_faiss_index(query_vector, top_k)
                return self._combine_search_results(azure_results, faiss_results, top_k)
            else:
                raise ValueError(f"No index available for search")
                
        except Exception as e:
            self.logger.error(f"Error searching index: {str(e)}")
            raise
    
    async def _search_azure_index(self, query_vector: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Search Azure Cognitive Search index"""
        try:
            # Use Azure Search vector search
            search_results = await self.azure_manager.vector_search(
                index_name=self.config.index_name,
                query_vector=query_vector,
                top_k=top_k
            )
            
            results = []
            for result in search_results:
                results.append({
                    'id': result['id'],
                    'score': result['score'],
                    'content': result.get('content', ''),
                    'metadata': result.get('metadata', {}),
                    'source': 'azure_search'
                })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching Azure index: {str(e)}")
            return []
    
    async def _search_faiss_index(self, query_vector: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Search FAISS index"""
        try:
            if self.faiss_index is None:
                raise ValueError("FAISS index not loaded")
            
            # Convert query to numpy array
            query_array = np.array([query_vector], dtype=np.float32)
            
            # Normalize if using cosine similarity
            if self.config.similarity_metric == "cosine":
                faiss.normalize_L2(query_array)
            
            # Search
            scores, indices = self.faiss_index.search(query_array, top_k)
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx >= 0:  # Valid result
                    metadata = self.document_metadata.get(idx, {})
                    results.append({
                        'id': metadata.get('id', f'doc_{idx}'),
                        'score': float(score),
                        'index': int(idx),
                        'metadata': metadata,
                        'source': 'faiss'
                    })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching FAISS index: {str(e)}")
            return []
    
    def _combine_search_results(self, azure_results: List[Dict[str, Any]], 
                               faiss_results: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """Combine results from Azure Search and FAISS"""
        try:
            # Simple combination strategy: interleave results
            combined = []
            azure_idx = 0
            faiss_idx = 0
            
            while len(combined) < top_k and (azure_idx < len(azure_results) or faiss_idx < len(faiss_results)):
                # Alternate between sources
                if azure_idx < len(azure_results) and (len(combined) % 2 == 0 or faiss_idx >= len(faiss_results)):
                    result = azure_results[azure_idx].copy()
                    result['combined_rank'] = len(combined) + 1
                    combined.append(result)
                    azure_idx += 1
                elif faiss_idx < len(faiss_results):
                    result = faiss_results[faiss_idx].copy()
                    result['combined_rank'] = len(combined) + 1
                    combined.append(result)
                    faiss_idx += 1
            
            return combined
            
        except Exception as e:
            self.logger.error(f"Error combining search results: {str(e)}")
            return azure_results + faiss_results
    
    async def add_documents(self, new_embeddings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Add new documents to existing index (incremental update)"""
        try:
            self.logger.info(f"Adding {len(new_embeddings)} new documents to index")
            
            if self.config.index_type == IndexType.AZURE_SEARCH:
                return await self._add_documents_azure(new_embeddings)
            elif self.config.index_type in [IndexType.FAISS_FLAT, IndexType.FAISS_IVF, IndexType.FAISS_HNSW]:
                return await self._add_documents_faiss(new_embeddings)
            elif self.config.index_type == IndexType.HYBRID:
                azure_result = await self._add_documents_azure(new_embeddings)
                faiss_result = await self._add_documents_faiss(new_embeddings)
                return {'azure': azure_result, 'faiss': faiss_result}
            
        except Exception as e:
            self.logger.error(f"Error adding documents: {str(e)}")
            raise
    
    async def _add_documents_azure(self, new_embeddings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Add documents to Azure Search index"""
        try:
            # Prepare documents similar to initial indexing
            search_documents = []
            
            for embedding_dict in new_embeddings:
                search_doc = {
                    'id': embedding_dict['id'],
                    'embedding': embedding_dict['embeddings'],
                    'content': embedding_dict.get('text_content', ''),
                    'metadata': {
                        'model_used': embedding_dict.get('model_used', 'unknown'),
                        'created_at': embedding_dict.get('created_at', datetime.now().isoformat()),
                        'embedding_dimension': len(embedding_dict['embeddings'])
                    }
                }
                
                if 'original_incident' in embedding_dict:
                    incident = embedding_dict['original_incident']
                    search_doc['metadata'].update({
                        'category': incident.get('category'),
                        'severity': incident.get('severity'),
                        'priority': incident.get('priority')
                    })
                
                search_documents.append(search_doc)
            
            # Index documents
            result = await self.azure_manager.search_client.index_documents_batch(search_documents)
            
            return {
                'documents_added': result['succeeded'],
                'failed_documents': result['failed']
            }
            
        except Exception as e:
            self.logger.error(f"Error adding documents to Azure Search: {str(e)}")
            raise
    
    async def _add_documents_faiss(self, new_embeddings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Add documents to FAISS index"""
        try:
            if self.faiss_index is None:
                raise ValueError("FAISS index not loaded")
            
            # Extract vectors and metadata
            vectors = []
            new_metadata = []
            
            for embedding_dict in new_embeddings:
                vectors.append(embedding_dict['embeddings'])
                
                doc_metadata = {
                    'id': embedding_dict['id'],
                    'model_used': embedding_dict.get('model_used', 'unknown'),
                    'created_at': embedding_dict.get('created_at', datetime.now().isoformat())
                }
                
                if 'original_incident' in embedding_dict:
                    doc_metadata['original_incident'] = embedding_dict['original_incident']
                
                new_metadata.append(doc_metadata)
            
            # Convert to numpy array
            vectors_array = np.array(vectors, dtype=np.float32)
            
            # Normalize if using cosine similarity
            if self.config.similarity_metric == "cosine":
                faiss.normalize_L2(vectors_array)
            
            # Add to index
            start_idx = self.faiss_index.ntotal
            self.faiss_index.add(vectors_array)
            
            # Update metadata
            for i, metadata in enumerate(new_metadata):
                idx = start_idx + i
                self.document_metadata[idx] = metadata
                self.id_to_index[metadata['id']] = idx
            
            return {
                'documents_added': len(vectors),
                'total_documents': self.faiss_index.ntotal
            }
            
        except Exception as e:
            self.logger.error(f"Error adding documents to FAISS: {str(e)}")
            raise
    
    async def remove_documents(self, document_ids: List[str]) -> Dict[str, Any]:
        """Remove documents from index"""
        try:
            self.logger.info(f"Removing {len(document_ids)} documents from index")
            
            if self.config.index_type == IndexType.AZURE_SEARCH:
                return await self._remove_documents_azure(document_ids)
            elif self.config.index_type in [IndexType.FAISS_FLAT, IndexType.FAISS_IVF, IndexType.FAISS_HNSW]:
                return await self._remove_documents_faiss(document_ids)
            elif self.config.index_type == IndexType.HYBRID:
                azure_result = await self._remove_documents_azure(document_ids)
                faiss_result = await self._remove_documents_faiss(document_ids)
                return {'azure': azure_result, 'faiss': faiss_result}
            
        except Exception as e:
            self.logger.error(f"Error removing documents: {str(e)}")
            raise
    
    async def _remove_documents_azure(self, document_ids: List[str]) -> Dict[str, Any]:
        """Remove documents from Azure Search index"""
        try:
            # Azure Search supports deletion by ID
            delete_actions = [{'@search.action': 'delete', 'id': doc_id} for doc_id in document_ids]
            result = await self.azure_manager.search_client.index_documents_batch(delete_actions)
            
            return {
                'documents_removed': result['succeeded'],
                'failed_removals': result['failed']
            }
            
        except Exception as e:
            self.logger.error(f"Error removing documents from Azure Search: {str(e)}")
            raise
    
    async def _remove_documents_faiss(self, document_ids: List[str]) -> Dict[str, Any]:
        """Remove documents from FAISS index"""
        try:
            # FAISS doesn't support deletion directly
            # We need to rebuild the index without the specified documents
            self.logger.warning("FAISS doesn't support direct deletion. Consider rebuilding the index.")
            
            # Remove from metadata mappings
            removed_count = 0
            for doc_id in document_ids:
                if doc_id in self.id_to_index:
                    idx = self.id_to_index[doc_id]
                    del self.id_to_index[doc_id]
                    if idx in self.document_metadata:
                        del self.document_metadata[idx]
                    removed_count += 1
            
            return {
                'documents_removed': removed_count,
                'note': 'Metadata removed. FAISS index rebuild recommended for complete removal.'
            }
            
        except Exception as e:
            self.logger.error(f"Error removing documents from FAISS: {str(e)}")
            raise
    
    def get_index_statistics(self) -> Dict[str, Any]:
        """Get current index statistics"""
        try:
            stats = {
                'index_type': self.config.index_type.value,
                'total_documents': 0,
                'build_statistics': self.build_stats
            }
            
            if self.faiss_index is not None:
                stats['faiss_total_vectors'] = self.faiss_index.ntotal
                stats['faiss_dimension'] = self.faiss_index.d
                stats['total_documents'] = self.faiss_index.ntotal
            
            if self.config.index_type in [IndexType.AZURE_SEARCH, IndexType.HYBRID]:
                # Would query Azure Search for document count
                stats['azure_search_index'] = self.config.index_name
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting index statistics: {str(e)}")
            return {'error': str(e)}

# Utility functions
async def build_knowledge_base_from_file(embeddings_file: str, output_path: str, 
                                       config: IndexConfig = None) -> Dict[str, Any]:
    """Build knowledge base from embeddings file"""
    try:
        if config is None:
            config = IndexConfig(
                index_type=IndexType.FAISS_FLAT,
                strategy=IndexStrategy.FULL_REBUILD,
                index_name="tebsarvis_incidents"
            )
        
        builder = KnowledgeBaseBuilder(config)
        await builder.initialize()
        
        result = await builder.build_index(embeddings_file, output_path)
        return result
        
    except Exception as e:
        logging.error(f"Error building knowledge base: {str(e)}")
        raise

async def batch_build_indexes(embeddings_files: List[str], output_dir: str,
                            configs: List[IndexConfig] = None) -> Dict[str, Any]:
    """Build multiple indexes from different embedding files"""
    results = []
    
    if configs is None:
        configs = [IndexConfig(
            index_type=IndexType.FAISS_FLAT,
            strategy=IndexStrategy.FULL_REBUILD,
            index_name=f"index_{i}"
        ) for i in range(len(embeddings_files))]
    
    for i, (embeddings_file, config) in enumerate(zip(embeddings_files, configs)):
        try:
            output_path = os.path.join(output_dir, f"index_{i}")
            result = await build_knowledge_base_from_file(embeddings_file, output_path, config)
            results.append({
                'embeddings_file': embeddings_file,
                'output_path': output_path,
                'status': 'success',
                'result': result
            })
            
        except Exception as e:
            results.append({
                'embeddings_file': embeddings_file,
                'status': 'error',
                'error': str(e)
            })
    
    return {
        'batch_results': results,
        'total_indexes': len(embeddings_files),
        'successful_builds': len([r for r in results if r['status'] == 'success']),
        'failed_builds': len([r for r in results if r['status'] == 'error'])
    }

# Command-line interface
async def main():
    """Main function for CLI usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Build knowledge base index from embeddings')
    parser.add_argument('embeddings_file', help='Path to embeddings file')
    parser.add_argument('--output', '-o', help='Output directory for index')
    parser.add_argument('--index-type', choices=['azure_search', 'faiss_flat', 'faiss_ivf', 'faiss_hnsw', 'hybrid'],
                       default='faiss_flat', help='Type of index to build')
    parser.add_argument('--index-name', default='tebsarvis_incidents', help='Name for the index')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size for processing')
    parser.add_argument('--vector-dimension', type=int, default=1536, help='Vector dimension')
    parser.add_argument('--similarity-metric', choices=['cosine', 'euclidean', 'dot_product'],
                       default='cosine', help='Similarity metric to use')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Create configuration
        config = IndexConfig(
            index_type=IndexType(args.index_type),
            strategy=IndexStrategy.FULL_REBUILD,
            index_name=args.index_name,
            batch_size=args.batch_size,
            vector_dimension=args.vector_dimension,
            similarity_metric=args.similarity_metric
        )
        
        output_path = args.output or f"index_{args.index_type}"
        
        result = await build_knowledge_base_from_file(args.embeddings_file, output_path, config)
        
        print(f"Knowledge base build completed successfully!")
        print(f"Index type: {result['index_metadata']['index_type']}")
        print(f"Total documents indexed: {result['index_metadata']['total_documents']}")
        print(f"Build time: {result['build_statistics']['build_time_seconds']:.2f} seconds")
        print(f"Output saved to: {output_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))