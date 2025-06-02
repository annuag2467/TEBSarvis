"""
Data Ingestion Pipeline for TEBSarvis Multi-Agent System
Processes Excel data, generates embeddings, and indexes for search.
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import re
import uuid
import os

from .azure_clients import AzureClientManager

class DataIngestionPipeline:
    """
    Pipeline for processing Excel incident data and preparing it for the multi-agent system.
    Handles data cleaning, embedding generation, and search index creation.
    """
    
    def __init__(self):
        self.azure_manager = AzureClientManager()
        self.logger = logging.getLogger("data_ingestion")
        
        # Processing statistics
        self.stats = {
            'total_records': 0,
            'processed_records': 0,
            'failed_records': 0,
            'embeddings_generated': 0,
            'records_indexed': 0
        }
    
    async def process_excel_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process the Excel file containing incident data.
        
        Args:
            file_path: Path to the Excel file
            
        Returns:
            Processing results and statistics
        """
        try:
            self.logger.info(f"Starting processing of Excel file: {file_path}")
            
            # Read Excel file
            df = pd.read_excel(file_path)
            self.stats['total_records'] = len(df)
            
            self.logger.info(f"Loaded {len(df)} records from Excel file")
            
            # Clean and validate data
            cleaned_df = await self._clean_data(df)
            
            # Process each record
            processed_records = []
            failed_records = []
            
            for index, row in cleaned_df.iterrows():
                try:
                    processed_record = await self._process_incident_record(row, index)
                    if processed_record:
                        processed_records.append(processed_record)
                        self.stats['processed_records'] += 1
                    else:
                        failed_records.append({'index': index, 'error': 'Processing failed'})
                        self.stats['failed_records'] += 1
                        
                except Exception as e:
                    self.logger.error(f"Error processing record {index}: {str(e)}")
                    failed_records.append({'index': index, 'error': str(e)})
                    self.stats['failed_records'] += 1
            
            # Store processed records in Cosmos DB and index for search
            storage_results = await self._store_and_index_records(processed_records)
            
            results = {
                'processing_stats': self.stats,
                'processed_records': len(processed_records),
                'failed_records': len(failed_records),
                'storage_results': storage_results,
                'sample_records': processed_records[:5] if processed_records else [],
                'processing_metadata': {
                    'file_path': file_path,
                    'timestamp': datetime.now().isoformat(),
                    'pipeline_version': '1.0.0'
                }
            }
            
            self.logger.info(f"Processing completed. Processed: {len(processed_records)}, Failed: {len(failed_records)}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing Excel file: {str(e)}")
            raise
    
    async def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate the incident data"""
        try:
            self.logger.info("Cleaning and validating data...")
            
            # Make a copy to avoid modifying original
            cleaned_df = df.copy()
            
            # Standard column name mapping (adjust based on your Excel structure)
            column_mapping = {
                'Incident ID': 'incident_id',
                'Summary': 'summary',
                'Description': 'description',
                'Category': 'category',
                'Priority': 'priority',
                'Severity': 'severity',
                'Date Submitted': 'date_submitted',
                'Reporter': 'reporter',
                'Resolution': 'resolution',
                'Resolution Date': 'resolution_date',
                'Status': 'status'
            }
            
            # Rename columns to standard format
            cleaned_df = cleaned_df.rename(columns=column_mapping)
            
            # Fill missing values
            cleaned_df['summary'] = cleaned_df['summary'].fillna('No summary provided')
            cleaned_df['description'] = cleaned_df['description'].fillna('No description provided')
            cleaned_df['category'] = cleaned_df['category'].fillna('General')
            cleaned_df['priority'] = cleaned_df['priority'].fillna('Medium')
            cleaned_df['severity'] = cleaned_df['severity'].fillna('3 - Medium')
            cleaned_df['resolution'] = cleaned_df['resolution'].fillna('')
            
            # Clean text fields
            text_columns = ['summary', 'description', 'resolution']
            for col in text_columns:
                if col in cleaned_df.columns:
                    cleaned_df[col] = cleaned_df[col].astype(str)
                    cleaned_df[col] = cleaned_df[col].apply(self._clean_text)
            
            # Standardize date format
            if 'date_submitted' in cleaned_df.columns:
                cleaned_df['date_submitted'] = pd.to_datetime(
                    cleaned_df['date_submitted'], 
                    errors='coerce'
                ).dt.strftime('%d-%m-%Y %H:%M')
            
            if 'resolution_date' in cleaned_df.columns:
                cleaned_df['resolution_date'] = pd.to_datetime(
                    cleaned_df['resolution_date'], 
                    errors='coerce'
                ).dt.strftime('%d-%m-%Y %H:%M')
            
            # Remove completely empty rows
            cleaned_df = cleaned_df.dropna(subset=['summary', 'description'], how='all')
            
            # Generate incident IDs if missing
            if 'incident_id' not in cleaned_df.columns or cleaned_df['incident_id'].isna().any():
                cleaned_df['incident_id'] = cleaned_df.apply(
                    lambda x: f"INC{str(x.name + 1).zfill(6)}", axis=1
                )
            
            self.logger.info(f"Data cleaning completed. {len(cleaned_df)} valid records remain")
            return cleaned_df
            
        except Exception as e:
            self.logger.error(f"Error cleaning data: {str(e)}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if pd.isna(text) or text == 'nan':
            return ''
        
        text = str(text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', '', text)
        
        return text
    
    async def _process_incident_record(self, row: pd.Series, index: int) -> Optional[Dict[str, Any]]:
        """Process a single incident record"""
        try:
            # Extract data from row
            incident_data = {
                'id': str(row.get('incident_id', f"INC{str(index + 1).zfill(6)}")),
                'summary': str(row.get('summary', '')),
                'description': str(row.get('description', '')),
                'category': str(row.get('category', 'General')),
                'priority': str(row.get('priority', 'Medium')),
                'severity': str(row.get('severity', '3 - Medium')),
                'date_submitted': str(row.get('date_submitted', '')),
                'reporter': str(row.get('reporter', '')),
                'resolution': str(row.get('resolution', '')),
                'resolution_date': str(row.get('resolution_date', '')),
                'status': str(row.get('status', 'Open')),
                'created_at': datetime.now().isoformat(),
                'data_source': 'excel_import'
            }
            
            # Generate embedding for search
            content_for_embedding = f"{incident_data['summary']} {incident_data['description']}"
            if incident_data['resolution']:
                content_for_embedding += f" {incident_data['resolution']}"
            
            # Generate embedding
            embedding = await self.azure_manager.get_embeddings(content_for_embedding)
            incident_data['embedding'] = embedding
            self.stats['embeddings_generated'] += 1
            
            # Add metadata for better categorization
            incident_data['metadata'] = {
                'word_count': len(content_for_embedding.split()),
                'has_resolution': bool(incident_data['resolution'].strip()),
                'text_length': len(content_for_embedding),
                'processing_timestamp': datetime.now().isoformat()
            }
            
            return incident_data
            
        except Exception as e:
            self.logger.error(f"Error processing record at index {index}: {str(e)}")
            return None
    
    async def _store_and_index_records(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Store records in Cosmos DB and index for search"""
        try:
            self.logger.info(f"Storing {len(records)} records in Cosmos DB and search index...")
            
            cosmos_results = {'stored': 0, 'failed': 0}
            search_results = {'indexed': 0, 'failed': 0}
            
            # Process in batches
            batch_size = 10
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                
                # Store in Cosmos DB
                for record in batch:
                    try:
                        await self.azure_manager.store_incident(record)
                        cosmos_results['stored'] += 1
                    except Exception as e:
                        self.logger.error(f"Failed to store record {record['id']}: {str(e)}")
                        cosmos_results['failed'] += 1
                
                # Index for search
                search_documents = []
                for record in batch:
                    try:
                        search_doc = self._prepare_search_document(record)
                        search_documents.append(search_doc)
                    except Exception as e:
                        self.logger.error(f"Failed to prepare search doc for {record['id']}: {str(e)}")
                        search_results['failed'] += 1
                
                # Batch index
                if search_documents:
                    try:
                        batch_stats = await self.azure_manager.search_client.index_documents_batch(search_documents)
                        search_results['indexed'] += batch_stats.get('succeeded', 0)
                        search_results['failed'] += batch_stats.get('failed', 0)
                    except Exception as e:
                        self.logger.error(f"Failed to index batch: {str(e)}")
                        search_results['failed'] += len(search_documents)
                
                # Small delay between batches
                await asyncio.sleep(0.1)
            
            self.stats['records_indexed'] = search_results['indexed']
            
            return {
                'cosmos_db': cosmos_results,
                'search_index': search_results,
                'total_processed': len(records)
            }
            
        except Exception as e:
            self.logger.error(f"Error storing and indexing records: {str(e)}")
            raise
    
    def _prepare_search_document(self, incident: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare incident data for search indexing"""
        return {
            'id': incident['id'],
            'content': f"{incident['summary']} {incident['description']} {incident['resolution']}",
            'embedding': incident['embedding'],
            'metadata': {
                'category': incident['category'],
                'priority': incident['priority'],
                'severity': incident['severity'],
                'date_submitted': incident['date_submitted'],
                'resolution': incident['resolution'],
                'status': incident['status'],
                'has_resolution': incident['metadata']['has_resolution']
            }
        }
    
    async def generate_knowledge_base(self, file_path: str) -> Dict[str, Any]:
        """
        Generate a comprehensive knowledge base from the Excel data.
        
        Args:
            file_path: Path to the Excel file
            
        Returns:
            Knowledge base generation results
        """
        try:
            # Process the Excel file first
            processing_results = await self.process_excel_file(file_path)
            
            # Generate additional knowledge artifacts
            knowledge_base = {
                'categories': await self._extract_categories(),
                'common_issues': await self._identify_common_issues(),
                'resolution_patterns': await self._extract_resolution_patterns(),
                'statistics': await self._generate_statistics(),
                'processing_results': processing_results
            }
            
            # Store knowledge base metadata
            kb_metadata = {
                'id': 'knowledge_base_v1',
                'created_at': datetime.now().isoformat(),
                'source_file': file_path,
                'total_incidents': processing_results['processed_records'],
                'knowledge_base': knowledge_base
            }
            
            # Store in Cosmos DB
            await self.azure_manager.store_incident(kb_metadata)
            
            return knowledge_base
            
        except Exception as e:
            self.logger.error(f"Error generating knowledge base: {str(e)}")
            raise
    
    async def _extract_categories(self) -> List[Dict[str, Any]]:
        """Extract incident categories and their frequency"""
        try:
            query = """
            SELECT c.category, COUNT(1) as count 
            FROM c 
            WHERE c.data_source = 'excel_import'
            GROUP BY c.category 
            ORDER BY COUNT(1) DESC
            """
            
            results = await self.azure_manager.query_incidents(query)
            return results
            
        except Exception as e:
            self.logger.error(f"Error extracting categories: {str(e)}")
            return []
    
    async def _identify_common_issues(self) -> List[Dict[str, Any]]:
        """Identify common issues based on summary patterns"""
        try:
            # Get all summaries
            query = """
            SELECT c.summary, c.category, COUNT(1) as frequency
            FROM c 
            WHERE c.data_source = 'excel_import'
            GROUP BY c.summary, c.category
            HAVING COUNT(1) > 1
            ORDER BY COUNT(1) DESC
            """
            
            results = await self.azure_manager.query_incidents(query)
            return results[:20]  # Top 20 common issues
            
        except Exception as e:
            self.logger.error(f"Error identifying common issues: {str(e)}")
            return []
    
    async def _extract_resolution_patterns(self) -> List[Dict[str, Any]]:
        """Extract successful resolution patterns"""
        try:
            query = """
            SELECT c.category, c.resolution, COUNT(1) as frequency
            FROM c 
            WHERE c.data_source = 'excel_import'
            AND LENGTH(c.resolution) > 10
            GROUP BY c.category, c.resolution
            ORDER BY COUNT(1) DESC
            """
            
            results = await self.azure_manager.query_incidents(query)
            return results[:15]  # Top 15 resolution patterns
            
        except Exception as e:
            self.logger.error(f"Error extracting resolution patterns: {str(e)}")
            return []
    
    async def _generate_statistics(self) -> Dict[str, Any]:
        """Generate overall statistics from the data"""
        try:
            stats = {}
            
            # Total incidents
            total_query = "SELECT COUNT(1) as total FROM c WHERE c.data_source = 'excel_import'"
            total_result = await self.azure_manager.query_incidents(total_query)
            stats['total_incidents'] = total_result[0]['total'] if total_result else 0
            
            # Resolved incidents
            resolved_query = """
            SELECT COUNT(1) as resolved FROM c 
            WHERE c.data_source = 'excel_import' 
            AND LENGTH(c.resolution) > 0
            """
            resolved_result = await self.azure_manager.query_incidents(resolved_query)
            stats['resolved_incidents'] = resolved_result[0]['resolved'] if resolved_result else 0
            
            # Resolution rate
            if stats['total_incidents'] > 0:
                stats['resolution_rate'] = (stats['resolved_incidents'] / stats['total_incidents']) * 100
            else:
                stats['resolution_rate'] = 0
            
            # Category distribution
            category_query = """
            SELECT c.category, COUNT(1) as count 
            FROM c 
            WHERE c.data_source = 'excel_import'
            GROUP BY c.category
            """
            stats['category_distribution'] = await self.azure_manager.query_incidents(category_query)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error generating statistics: {str(e)}")
            return {}
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        return {
            'stats': self.stats.copy(),
            'timestamp': datetime.now().isoformat()
        }