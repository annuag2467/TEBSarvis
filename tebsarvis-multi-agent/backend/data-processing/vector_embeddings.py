"""
Vector Embeddings Generator for TEBSarvis Multi-Agent System
Generates embeddings for incident data using Azure OpenAI and other embedding models.
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

class EmbeddingModel(Enum):
    """Supported embedding models"""
    AZURE_OPENAI_ADA = "azure_openai_ada"
    AZURE_OPENAI_ADA_002 = "azure_openai_ada_002"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    BGE_LARGE = "bge_large"

class EmbeddingStrategy(Enum):
    """Embedding generation strategies"""
    SINGLE_FIELD = "single_field"  # Embed only one field (e.g., summary)
    CONCATENATED = "concatenated"  # Concatenate multiple fields
    WEIGHTED_CONCATENATED = "weighted_concatenated"  # Weight fields by importance
    MULTI_VECTOR = "multi_vector"  # Generate separate embeddings for each field

@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation"""
    model: EmbeddingModel
    strategy: EmbeddingStrategy
    batch_size: int = 50
    max_tokens: int = 8000
    retry_attempts: int = 3
    cache_embeddings: bool = True
    normalize_embeddings: bool = True
    fields_to_embed: List[str] = None
    field_weights: Dict[str, float] = None

@dataclass
class EmbeddingResult:
    """Result of embedding generation"""
    id: str
    embeddings: Union[List[float], Dict[str, List[float]]]
    model_used: str
    strategy_used: str
    embedding_dimension: int
    created_at: str
    metadata: Dict[str, Any] = None

class VectorEmbeddingGenerator:
    """
    Generates vector embeddings for incident data using various embedding models.
    Supports batch processing, caching, and multiple embedding strategies.
    """
    
    def __init__(self, config: EmbeddingConfig = None):
        self.config = config or EmbeddingConfig(
            model=EmbeddingModel.AZURE_OPENAI_ADA_002,
            strategy=EmbeddingStrategy.CONCATENATED,
            fields_to_embed=['summary', 'description', 'category']
        )
        
        self.logger = logging.getLogger("vector_embeddings")
        self.azure_manager = AzureClientManager()
        self.text_processor = TextProcessor()
        self.cache_manager = CacheManager(default_ttl=3600)  # 1 hour cache
        self.metrics = MetricsCollector()
        
        # Model-specific configurations
        self.model_configs = {
            EmbeddingModel.AZURE_OPENAI_ADA_002: {
                'max_tokens': 8191,
                'dimensions': 1536,
                'rate_limit_rpm': 240000,
                'rate_limit_tpm': 350000
            },
            EmbeddingModel.AZURE_OPENAI_ADA: {
                'max_tokens': 2048,
                'dimensions': 1024,
                'rate_limit_rpm': 240000,
                'rate_limit_tpm': 350000
            }
        }
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        
        # Processing statistics
        self.processing_stats = {
            'total_items': 0,
            'successful_embeddings': 0,
            'failed_embeddings': 0,
            'cached_embeddings': 0,
            'total_tokens_used': 0,
            'total_processing_time': 0.0
        }
    
    async def initialize(self):
        """Initialize the embedding generator"""
        try:
            await self.azure_manager.initialize()
            self.logger.info(f"Vector Embedding Generator initialized with model: {self.config.model.value}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Vector Embedding Generator: {str(e)}")
            raise
    
    async def generate_embeddings_for_incidents(self, incidents: List[Dict[str, Any]], 
                                              output_path: str = None) -> Dict[str, Any]:
        """
        Generate embeddings for a list of incidents.
        
        Args:
            incidents: List of incident dictionaries
            output_path: Optional path to save embeddings
            
        Returns:
            Dictionary with embeddings and processing results
        """
        try:
            start_time = time.time()
            self.logger.info(f"Starting embedding generation for {len(incidents)} incidents")
            
            # Validate and prepare incidents
            prepared_incidents = await self._prepare_incidents_for_embedding(incidents)
            
            # Generate embeddings in batches
            embedding_results = await self._generate_embeddings_batch(prepared_incidents)
            
            # Post-process results
            processed_results = await self._post_process_embeddings(embedding_results)
            
            # Save results if output path provided
            if output_path:
                await self._save_embeddings(processed_results, output_path)
            
            # Calculate final statistics
            total_time = time.time() - start_time
            self.processing_stats['total_processing_time'] = total_time
            
            self.logger.info(f"Embedding generation completed in {total_time:.2f} seconds")
            
            return {
                'embeddings': processed_results,
                'statistics': self.processing_stats,
                'configuration': asdict(self.config),
                'processing_metadata': {
                    'total_incidents': len(incidents),
                    'successful_embeddings': len(processed_results),
                    'processing_time_seconds': total_time,
                    'timestamp': datetime.now().isoformat(),
                    'generator_version': '1.0.0'
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    async def _prepare_incidents_for_embedding(self, incidents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare incidents for embedding generation"""
        try:
            prepared_incidents = []
            
            for incident in incidents:
                try:
                    # Extract text content based on strategy
                    text_content = await self._extract_text_for_embedding(incident)
                    
                    # Validate text content
                    if not text_content or not text_content.strip():
                        self.logger.warning(f"Empty text content for incident {incident.get('id', 'unknown')}")
                        continue
                    
                    # Truncate if too long
                    max_tokens = self.model_configs[self.config.model]['max_tokens']
                    text_content = self._truncate_text(text_content, max_tokens)
                    
                    prepared_incident = {
                        'id': incident.get('id', str(uuid.uuid4())),
                        'text_content': text_content,
                        'original_incident': incident,
                        'fields_used': self.config.fields_to_embed
                    }
                    
                    prepared_incidents.append(prepared_incident)
                    
                except Exception as e:
                    self.logger.error(f"Error preparing incident {incident.get('id', 'unknown')}: {str(e)}")
                    continue
            
            self.processing_stats['total_items'] = len(prepared_incidents)
            self.logger.info(f"Prepared {len(prepared_incidents)} incidents for embedding")
            
            return prepared_incidents
            
        except Exception as e:
            self.logger.error(f"Error preparing incidents: {str(e)}")
            raise
    
    async def _extract_text_for_embedding(self, incident: Dict[str, Any]) -> str:
        """Extract text content based on the configured strategy"""
        try:
            if self.config.strategy == EmbeddingStrategy.SINGLE_FIELD:
                # Use only the first field from fields_to_embed
                field = self.config.fields_to_embed[0] if self.config.fields_to_embed else 'summary'
                return self.text_processor.clean_text(incident.get(field, ''))
            
            elif self.config.strategy == EmbeddingStrategy.CONCATENATED:
                # Concatenate all specified fields
                text_parts = []
                for field in self.config.fields_to_embed or ['summary', 'description']:
                    if field in incident and incident[field]:
                        text_parts.append(self.text_processor.clean_text(str(incident[field])))
                
                return ' '.join(text_parts)
            
            elif self.config.strategy == EmbeddingStrategy.WEIGHTED_CONCATENATED:
                # Concatenate fields with weights (repeat important fields)
                text_parts = []
                weights = self.config.field_weights or {'summary': 2, 'description': 1, 'category': 1}
                
                for field in self.config.fields_to_embed or ['summary', 'description', 'category']:
                    if field in incident and incident[field]:
                        clean_text = self.text_processor.clean_text(str(incident[field]))
                        weight = weights.get(field, 1)
                        
                        # Repeat text based on weight
                        for _ in range(int(weight)):
                            text_parts.append(clean_text)
                
                return ' '.join(text_parts)
            
            elif self.config.strategy == EmbeddingStrategy.MULTI_VECTOR:
                # For multi-vector, we'll return a structured format
                text_dict = {}
                for field in self.config.fields_to_embed or ['summary', 'description', 'category']:
                    if field in incident and incident[field]:
                        text_dict[field] = self.text_processor.clean_text(str(incident[field]))
                
                return json.dumps(text_dict)  # Will be handled specially in embedding generation
            
            else:
                # Default to concatenated
                return f"{incident.get('summary', '')} {incident.get('description', '')}"
                
        except Exception as e:
            self.logger.error(f"Error extracting text for embedding: {str(e)}")
            return ""
    
    def _truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limits"""
        try:
            # Rough approximation: 1 token ≈ 0.75 words
            max_words = int(max_tokens * 0.75)
            words = text.split()
            
            if len(words) <= max_words:
                return text
            
            # Truncate and add ellipsis
            truncated_words = words[:max_words-1]
            return ' '.join(truncated_words) + '...'
            
        except Exception:
            return text[:max_tokens * 4]  # Fallback: rough character limit
    
    async def _generate_embeddings_batch(self, prepared_incidents: List[Dict[str, Any]]) -> List[EmbeddingResult]:
        """Generate embeddings in batches"""
        try:
            all_results = []
            batches = batch_list(prepared_incidents, self.config.batch_size)
            
            self.logger.info(f"Processing {len(batches)} batches of size {self.config.batch_size}")
            
            for batch_idx, batch in enumerate(batches):
                try:
                    self.logger.debug(f"Processing batch {batch_idx + 1}/{len(batches)}")
                    
                    # Check cache first
                    batch_results = []
                    uncached_items = []
                    
                    for item in batch:
                        cache_key = self._generate_cache_key(item)
                        cached_embedding = self.cache_manager.get(cache_key)
                        
                        if cached_embedding and self.config.cache_embeddings:
                            batch_results.append(cached_embedding)
                            self.processing_stats['cached_embeddings'] += 1
                        else:
                            uncached_items.append((item, cache_key))
                    
                    # Generate embeddings for uncached items
                    if uncached_items:
                        new_embeddings = await self._generate_batch_embeddings(
                            [item[0] for item in uncached_items]
                        )
                        
                        # Cache new embeddings
                        for (item, cache_key), embedding_result in zip(uncached_items, new_embeddings):
                            if self.config.cache_embeddings:
                                self.cache_manager.set(cache_key, embedding_result)
                            batch_results.append(embedding_result)
                    
                    all_results.extend(batch_results)
                    
                    # Rate limiting
                    await self._apply_rate_limiting()
                    
                except Exception as e:
                    self.logger.error(f"Error processing batch {batch_idx}: {str(e)}")
                    # Continue with next batch
                    continue
            
            self.logger.info(f"Generated embeddings for {len(all_results)} incidents")
            return all_results
            
        except Exception as e:
            self.logger.error(f"Error in batch embedding generation: {str(e)}")
            raise
    
    async def _generate_batch_embeddings(self, batch_items: List[Dict[str, Any]]) -> List[EmbeddingResult]:
        """Generate embeddings for a single batch"""
        try:
            results = []
            
            if self.config.model in [EmbeddingModel.AZURE_OPENAI_ADA, EmbeddingModel.AZURE_OPENAI_ADA_002]:
                results = await self._generate_azure_openai_embeddings(batch_items)
            else:
                # For other models, generate one by one for now
                for item in batch_items:
                    result = await self._generate_single_embedding(item)
                    if result:
                        results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error generating batch embeddings: {str(e)}")
            return []
    
    async def _generate_azure_openai_embeddings(self, batch_items: List[Dict[str, Any]]) -> List[EmbeddingResult]:
        """Generate embeddings using Azure OpenAI"""
        try:
            results = []
            
            # Prepare texts for embedding
            texts = []
            for item in batch_items:
                if self.config.strategy == EmbeddingStrategy.MULTI_VECTOR:
                    # Handle multi-vector strategy
                    text_dict = json.loads(item['text_content'])
                    texts.append(list(text_dict.values()))  # Will handle separately
                else:
                    texts.append(item['text_content'])
            
            if self.config.strategy == EmbeddingStrategy.MULTI_VECTOR:
                # Generate separate embeddings for each field
                for idx, item in enumerate(batch_items):
                    text_dict = json.loads(item['text_content'])
                    field_embeddings = {}
                    
                    for field, text in text_dict.items():
                        if text.strip():
                            embedding = await self.azure_manager.get_embeddings(text)
                            if self.config.normalize_embeddings:
                                embedding = self._normalize_embedding(embedding)
                            field_embeddings[field] = embedding
                    
                    result = EmbeddingResult(
                        id=item['id'],
                        embeddings=field_embeddings,
                        model_used=self.config.model.value,
                        strategy_used=self.config.strategy.value,
                        embedding_dimension=len(list(field_embeddings.values())[0]) if field_embeddings else 0,
                        created_at=datetime.now().isoformat(),
                        metadata={
                            'fields_embedded': list(field_embeddings.keys()),
                            'total_fields': len(text_dict)
                        }
                    )
                    results.append(result)
                    self.processing_stats['successful_embeddings'] += 1
            
            else:
                # Generate single embedding for concatenated text
                embeddings = await self.azure_manager.get_embeddings(texts)
                
                for idx, item in enumerate(batch_items):
                    try:
                        embedding = embeddings[idx] if isinstance(embeddings[0], list) else embeddings
                        
                        if self.config.normalize_embeddings:
                            embedding = self._normalize_embedding(embedding)
                        
                        result = EmbeddingResult(
                            id=item['id'],
                            embeddings=embedding,
                            model_used=self.config.model.value,
                            strategy_used=self.config.strategy.value,
                            embedding_dimension=len(embedding),
                            created_at=datetime.now().isoformat(),
                            metadata={
                                'text_length': len(item['text_content']),
                                'fields_used': item['fields_used']
                            }
                        )
                        results.append(result)
                        self.processing_stats['successful_embeddings'] += 1
                        
                    except Exception as e:
                        self.logger.error(f"Error processing embedding for item {item['id']}: {str(e)}")
                        self.processing_stats['failed_embeddings'] += 1
                        continue
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error generating Azure OpenAI embeddings: {str(e)}")
            return []
    
    async def _generate_single_embedding(self, item: Dict[str, Any]) -> Optional[EmbeddingResult]:
        """Generate embedding for a single item (fallback method)"""
        try:
            text_content = item['text_content']
            
            if self.config.model in [EmbeddingModel.AZURE_OPENAI_ADA, EmbeddingModel.AZURE_OPENAI_ADA_002]:
                embedding = await self.azure_manager.get_embeddings(text_content)
                
                if self.config.normalize_embeddings:
                    embedding = self._normalize_embedding(embedding)
                
                result = EmbeddingResult(
                    id=item['id'],
                    embeddings=embedding,
                    model_used=self.config.model.value,
                    strategy_used=self.config.strategy.value,
                    embedding_dimension=len(embedding),
                    created_at=datetime.now().isoformat(),
                    metadata={
                        'text_length': len(text_content),
                        'fields_used': item['fields_used']
                    }
                )
                
                self.processing_stats['successful_embeddings'] += 1
                return result
            
        except Exception as e:
            self.logger.error(f"Error generating single embedding for {item['id']}: {str(e)}")
            self.processing_stats['failed_embeddings'] += 1
            return None
    
    def _normalize_embedding(self, embedding: List[float]) -> List[float]:
        """Normalize embedding vector to unit length"""
        try:
            embedding_array = np.array(embedding)
            norm = np.linalg.norm(embedding_array)
            
            if norm == 0:
                return embedding
            
            normalized = embedding_array / norm
            return normalized.tolist()
            
        except Exception:
            return embedding
    
    def _generate_cache_key(self, item: Dict[str, Any]) -> str:
        """Generate cache key for an item"""
        text_hash = hash(item['text_content'])
        model_key = self.config.model.value
        strategy_key = self.config.strategy.value
        
        return f"embedding_{model_key}_{strategy_key}_{text_hash}"
    
    async def _apply_rate_limiting(self):
        """Apply rate limiting between requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            await asyncio.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    async def _post_process_embeddings(self, embedding_results: List[EmbeddingResult]) -> List[Dict[str, Any]]:
        """Post-process embedding results"""
        try:
            processed_results = []
            
            for result in embedding_results:
                processed_result = asdict(result)
                
                # Add quality metrics
                processed_result['quality_metrics'] = self._calculate_embedding_quality(result)
                
                # Add similarity metadata if needed
                if self.config.strategy == EmbeddingStrategy.MULTI_VECTOR:
                    processed_result['vector_count'] = len(result.embeddings)
                else:
                    processed_result['vector_count'] = 1
                
                processed_results.append(processed_result)
            
            return processed_results
            
        except Exception as e:
            self.logger.error(f"Error post-processing embeddings: {str(e)}")
            return [asdict(result) for result in embedding_results]
    
    def _calculate_embedding_quality(self, result: EmbeddingResult) -> Dict[str, Any]:
        """Calculate quality metrics for an embedding"""
        try:
            metrics = {
                'dimension_consistency': True,
                'has_valid_values': True,
                'normalized': self.config.normalize_embeddings
            }
            
            if isinstance(result.embeddings, list):
                # Single vector
                embedding_array = np.array(result.embeddings)
                
                # Check for NaN or infinite values
                metrics['has_valid_values'] = np.all(np.isfinite(embedding_array))
                
                # Check if normalized (approximately unit length)
                if self.config.normalize_embeddings:
                    norm = np.linalg.norm(embedding_array)
                    metrics['is_normalized'] = abs(norm - 1.0) < 0.01
                
                # Calculate variance (diversity of values)
                metrics['variance'] = float(np.var(embedding_array))
                
            elif isinstance(result.embeddings, dict):
                # Multi-vector
                all_valid = True
                dimensions = []
                
                for field, embedding in result.embeddings.items():
                    embedding_array = np.array(embedding)
                    if not np.all(np.isfinite(embedding_array)):
                        all_valid = False
                    dimensions.append(len(embedding))
                
                metrics['has_valid_values'] = all_valid
                metrics['dimension_consistency'] = len(set(dimensions)) == 1
                metrics['field_count'] = len(result.embeddings)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating embedding quality: {str(e)}")
            return {'error': str(e)}
    
    async def _save_embeddings(self, embeddings: List[Dict[str, Any]], output_path: str):
        """Save embeddings to file"""
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Determine format based on file extension
            if output_path.endswith('.json'):
                await self._save_embeddings_json(embeddings, output_path)
            elif output_path.endswith('.pkl') or output_path.endswith('.pickle'):
                await self._save_embeddings_pickle(embeddings, output_path)
            elif output_path.endswith('.npz'):
                await self._save_embeddings_numpy(embeddings, output_path)
            else:
                # Default to JSON
                await self._save_embeddings_json(embeddings, output_path)
            
            self.logger.info(f"Saved {len(embeddings)} embeddings to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving embeddings: {str(e)}")
            raise
    
    async def _save_embeddings_json(self, embeddings: List[Dict[str, Any]], output_path: str):
        """Save embeddings in JSON format"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(embeddings, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            self.logger.error(f"Error saving JSON embeddings: {str(e)}")
            raise
    
    async def _save_embeddings_pickle(self, embeddings: List[Dict[str, Any]], output_path: str):
        """Save embeddings in Pickle format"""
        try:
            with open(output_path, 'wb') as f:
                pickle.dump(embeddings, f)
        except Exception as e:
            self.logger.error(f"Error saving Pickle embeddings: {str(e)}")
            raise
    
    async def _save_embeddings_numpy(self, embeddings: List[Dict[str, Any]], output_path: str):
        """Save embeddings in NumPy format"""
        try:
            # Extract vectors and metadata separately
            vectors = []
            metadata = []
            
            for embedding in embeddings:
                if isinstance(embedding['embeddings'], list):
                    vectors.append(embedding['embeddings'])
                else:
                    # For multi-vector, save the first field or concatenate
                    if isinstance(embedding['embeddings'], dict):
                        first_vector = list(embedding['embeddings'].values())[0]
                        vectors.append(first_vector)
                
                metadata.append({
                    'id': embedding['id'],
                    'model_used': embedding['model_used'],
                    'created_at': embedding['created_at']
                })
            
            # Save vectors and metadata
            np.savez_compressed(
                output_path,
                vectors=np.array(vectors),
                metadata=metadata
            )
        except Exception as e:
            self.logger.error(f"Error saving NumPy embeddings: {str(e)}")
            raise
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        return self.processing_stats.copy()
    
    async def load_embeddings(self, file_path: str) -> List[Dict[str, Any]]:
        """Load embeddings from file"""
        try:
            if file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            elif file_path.endswith('.pkl') or file_path.endswith('.pickle'):
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
            elif file_path.endswith('.npz'):
                data = np.load(file_path, allow_pickle=True)
                vectors = data['vectors']
                metadata = data['metadata']
                
                embeddings = []
                for i, vector in enumerate(vectors):
                    embedding = {
                        'embeddings': vector.tolist(),
                        'embedding_dimension': len(vector),
                        **metadata[i]
                    }
                    embeddings.append(embedding)
                
                return embeddings
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
                
        except Exception as e:
            self.logger.error(f"Error loading embeddings: {str(e)}")
            raise

# Utility functions
async def generate_embeddings_for_file(input_file: str, output_file: str, 
                                     config: EmbeddingConfig = None) -> Dict[str, Any]:
    """Generate embeddings for incidents from a JSON file"""
    try:
        # Load incidents
        with open(input_file, 'r', encoding='utf-8') as f:
            incidents = json.load(f)
        
        # Generate embeddings
        generator = VectorEmbeddingGenerator(config)
        await generator.initialize()
        
        result = await generator.generate_embeddings_for_incidents(incidents, output_file)
        return result
        
    except Exception as e:
        logging.error(f"Error generating embeddings for file: {str(e)}")
        raise

async def batch_generate_embeddings(input_files: List[str], output_dir: str,
                                  config: EmbeddingConfig = None) -> Dict[str, Any]:
    """Generate embeddings for multiple input files"""
    results = []
    generator = VectorEmbeddingGenerator(config)
    await generator.initialize()
    
    for input_file in input_files:
        try:
            output_file = os.path.join(output_dir, f"{Path(input_file).stem}_embeddings.json")
            
            with open(input_file, 'r', encoding='utf-8') as f:
                incidents = json.load(f)
            
            result = await generator.generate_embeddings_for_incidents(incidents, output_file)
            results.append({
                'input_file': input_file,
                'output_file': output_file,
                'status': 'success',
                'result': result
            })
            
        except Exception as e:
            results.append({
                'input_file': input_file,
                'status': 'error',
                'error': str(e)
            })
    
    return {
        'batch_results': results,
        'total_files': len(input_files),
        'successful_files': len([r for r in results if r['status'] == 'success']),
        'failed_files': len([r for r in results if r['status'] == 'error'])
    }

# Command-line interface
async def main():
    """Main function for CLI usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate vector embeddings for incident data')
    parser.add_argument('input_file', help='Path to input JSON file with incidents')
    parser.add_argument('--output', '-o', help='Output file path for embeddings')
    parser.add_argument('--model', choices=['azure_openai_ada_002', 'azure_openai_ada'], 
                       default='azure_openai_ada_002', help='Embedding model to use')
    parser.add_argument('--strategy', choices=['single_field', 'concatenated', 'weighted_concatenated', 'multi_vector'],
                       default='concatenated', help='Embedding strategy')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size for processing')
    parser.add_argument('--fields', nargs='+', default=['summary', 'description', 'category'],
                       help='Fields to include in embeddings')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Create configuration
        config = EmbeddingConfig(
            model=EmbeddingModel(args.model),
            strategy=EmbeddingStrategy(args.strategy),
            batch_size=args.batch_size,
            fields_to_embed=args.fields
        )
        
        output_path = args.output or f"{Path(args.input_file).stem}_embeddings.json"
        
        result = await generate_embeddings_for_file(args.input_file, output_path, config)
        
        print(f"Embedding generation completed successfully!")
        print(f"Generated embeddings for {result['processing_metadata']['successful_embeddings']} incidents")
        print(f"Processing time: {result['processing_metadata']['processing_time_seconds']:.2f} seconds")
        print(f"Output saved to: {output_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))