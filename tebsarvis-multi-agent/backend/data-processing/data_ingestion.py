"""
Data Ingestion Module for TEBSarvis Multi-Agent System
Converts Excel incident data to structured JSON format for processing.
"""

import pandas as pd
import json
import logging
import asyncio
import os
import sys
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import uuid
import re
from pathlib import Path
from dotenv import load_dotenv

# Define paths for environment variable loading
env_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
local_settings_path = os.path.join(os.path.dirname(__file__), '..', 'azure_functions', 'local.settings.json')

# Load environment variables
if os.path.exists(env_path):
    load_dotenv(env_path)

# Load from local.settings.json if available
if os.path.exists(local_settings_path):
    try:
        with open(local_settings_path) as f:
            settings = json.load(f)
            if 'Values' in settings:
                for key, value in settings['Values'].items():
                    if not os.getenv(key):  # Only set if not already set
                        os.environ[key] = str(value)
    except Exception as e:
        print(f"Warning: Could not load local.settings.json: {e}")

# Add the backend path to sys.path to import shared utilities
backend_path = os.path.join(os.path.dirname(__file__), '..')
if backend_path not in sys.path:
    sys.path.append(backend_path)

from azure_functions.shared.azure_clients import AzureClientManager
from azure_functions.shared.agent_utils import TextProcessor, DataTransformer, ValidationError

class DataIngestionProcessor:
    """
    Processes raw incident data from Excel files and converts to structured JSON format.
    Handles data cleaning, validation, and enrichment.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("data_ingestion")
        self.azure_manager = AzureClientManager()
        self.text_processor = TextProcessor()
        self.data_transformer = DataTransformer()
        
        # Data processing configuration
        self.required_columns = [
            'Case ID', 'Summary', 'Description', 'Category', 
            'Severity', 'Priority', 'Date Submitted', 'Reporter'
        ]
        
        # Column mapping for normalization
        self.column_mapping = {
            'Case ID': 'id',
            'Summary': 'summary',
            'Description': 'description',
            'Category': 'category',
            'Severity': 'severity',
            'Priority': 'priority',
            'Date Submitted': 'date_submitted',
            'Reporter': 'reporter',
            'Resolution': 'resolution',
            'Resolution Date': 'resolution_date',
            'Status': 'status',
            'Assigned To': 'assigned_to',
            'Tags': 'tags'
        }
        
        # Data validation rules
        self.validation_rules = {
            'summary': {'min_length': 10, 'max_length': 500},
            'description': {'min_length': 20, 'max_length': 2000},
            'category': {'required': True},
            'severity': {'valid_values': ['Low', 'Medium', 'High', 'Critical']},
            'priority': {'valid_values': ['Low', 'Medium', 'High', 'Critical']}
        }
        
        # Statistics tracking
        self.processing_stats = {
            'total_records': 0,
            'processed_records': 0,
            'failed_records': 0,
            'validation_errors': 0,
            'data_quality_score': 0.0
        }
    
    async def initialize(self):
        """Initialize the data ingestion processor"""
        try:
            await self.azure_manager.initialize()
            self.logger.info("Data Ingestion Processor initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Data Ingestion Processor: {str(e)}")
            raise
    
    async def process_excel_file(self, file_path: str, output_path: str = None) -> Dict[str, Any]:
        """
        Process Excel file and convert to structured JSON.
        
        Args:
            file_path: Path to the Excel file
            output_path: Optional output path for JSON file
            
        Returns:
            Processing results with statistics
        """
        try:
            self.logger.info(f"Starting to process Excel file: {file_path}")
            
            # Validate input file
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Excel file not found: {file_path}")
            
            # Read Excel file
            df = await self._read_excel_file(file_path)
            
            # Validate and clean data
            cleaned_df = await self._clean_and_validate_data(df)
            
            # Convert to structured format
            structured_data = await self._convert_to_structured_format(cleaned_df)
            
            # Enrich data with additional processing
            enriched_data = await self._enrich_incident_data(structured_data)
            
            # Save to JSON if output path provided
            if output_path:
                await self._save_to_json(enriched_data, output_path)
            
            # Calculate final statistics
            self._calculate_processing_statistics(len(df), len(enriched_data))
            
            self.logger.info(f"Successfully processed {len(enriched_data)} incidents")
            
            return {
                'incidents': enriched_data,
                'statistics': self.processing_stats,
                'processing_metadata': {
                    'source_file': file_path,
                    'output_file': output_path,
                    'processed_at': datetime.now().isoformat(),
                    'processor_version': '1.0.0'
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error processing Excel file: {str(e)}")
            raise
    
    async def _read_excel_file(self, file_path: str) -> pd.DataFrame:
        """Read and parse Excel file"""
        try:
            # Try different sheet names if the default doesn't work
            sheet_names = [None, 'Sheet1', 'Data', 'Incidents', 'Cases']
            
            df = None
            for sheet_name in sheet_names:
                try:
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    self.logger.info(f"Successfully read Excel file with sheet: {sheet_name}")
                    break
                except Exception as e:
                    if sheet_name is None:
                        self.logger.warning(f"Could not read default sheet: {str(e)}")
                    continue
            
            if df is None:
                raise ValueError("Could not read any sheet from the Excel file")
            
            # Basic validation
            if df.empty:
                raise ValueError("Excel file is empty")
            
            self.processing_stats['total_records'] = len(df)
            self.logger.info(f"Read {len(df)} records from Excel file")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error reading Excel file: {str(e)}")
            raise
    
    async def _clean_and_validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate the dataframe"""
        try:
            self.logger.info("Starting data cleaning and validation")
            
            # Normalize column names
            df = self._normalize_column_names(df)
            
            # Check for required columns
            missing_columns = [col for col in self.required_columns if col not in df.columns]
            if missing_columns:
                self.logger.warning(f"Missing columns: {missing_columns}")
            
            # Remove completely empty rows
            initial_count = len(df)
            df = df.dropna(how='all')
            removed_count = initial_count - len(df)
            if removed_count > 0:
                self.logger.info(f"Removed {removed_count} completely empty rows")
            
            # Clean text fields
            text_columns = ['Summary', 'Description', 'Category', 'Reporter']
            for col in text_columns:
                if col in df.columns:
                    df[col] = df[col].astype(str).apply(self.text_processor.clean_text)
            
            # Validate data quality
            df = await self._validate_data_quality(df)
            
            self.logger.info(f"Data cleaning completed. {len(df)} records remaining")
            return df
            
        except Exception as e:
            self.logger.error(f"Error in data cleaning: {str(e)}")
            raise
    
    def _normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names to handle variations"""
        try:
            # Create a mapping of found columns to standard names
            column_map = {}
            
            for col in df.columns:
                col_lower = col.lower().strip()
                
                # Map common variations
                if 'case' in col_lower and 'id' in col_lower:
                    column_map[col] = 'Case ID'
                elif 'summary' in col_lower or 'title' in col_lower:
                    column_map[col] = 'Summary'
                elif 'description' in col_lower or 'detail' in col_lower:
                    column_map[col] = 'Description'
                elif 'category' in col_lower or 'type' in col_lower:
                    column_map[col] = 'Category'
                elif 'severity' in col_lower:
                    column_map[col] = 'Severity'
                elif 'priority' in col_lower:
                    column_map[col] = 'Priority'
                elif 'date' in col_lower and ('submit' in col_lower or 'created' in col_lower):
                    column_map[col] = 'Date Submitted'
                elif 'reporter' in col_lower or 'requester' in col_lower:
                    column_map[col] = 'Reporter'
                elif 'resolution' in col_lower and 'date' not in col_lower:
                    column_map[col] = 'Resolution'
                elif 'resolution' in col_lower and 'date' in col_lower:
                    column_map[col] = 'Resolution Date'
                elif 'status' in col_lower:
                    column_map[col] = 'Status'
                elif 'assign' in col_lower:
                    column_map[col] = 'Assigned To'
                elif 'tag' in col_lower:
                    column_map[col] = 'Tags'
            
            # Rename columns
            df = df.rename(columns=column_map)
            
            self.logger.info(f"Normalized {len(column_map)} column names")
            return df
            
        except Exception as e:
            self.logger.error(f"Error normalizing column names: {str(e)}")
            return df
    
    async def _validate_data_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate data quality and mark issues"""
        try:
            validation_issues = []
            
            for index, row in df.iterrows():
                row_issues = []
                
                # Validate each field according to rules
                for field, rules in self.validation_rules.items():
                    if field in df.columns:
                        value = row[field]
                        
                        # Required field validation
                        if rules.get('required', False) and (pd.isna(value) or str(value).strip() == ''):
                            row_issues.append(f"{field}_missing")
                        
                        # Length validation
                        if not pd.isna(value):
                            value_str = str(value).strip()
                            if 'min_length' in rules and len(value_str) < rules['min_length']:
                                row_issues.append(f"{field}_too_short")
                            if 'max_length' in rules and len(value_str) > rules['max_length']:
                                row_issues.append(f"{field}_too_long")
                        
                        # Valid values validation
                        if 'valid_values' in rules and not pd.isna(value):
                            if str(value).strip() not in rules['valid_values']:
                                row_issues.append(f"{field}_invalid_value")
                
                if row_issues:
                    validation_issues.append({
                        'index': index,
                        'issues': row_issues
                    })
                    self.processing_stats['validation_errors'] += 1
            
            # Log validation issues
            if validation_issues:
                self.logger.warning(f"Found {len(validation_issues)} rows with validation issues")
                
                # For now, we'll keep all rows but log the issues
                # In production, you might want to filter out severely invalid rows
                
            return df
            
        except Exception as e:
            self.logger.error(f"Error in data validation: {str(e)}")
            return df
    
    async def _convert_to_structured_format(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Convert DataFrame to structured incident format"""
        try:
            structured_incidents = []
            
            for index, row in df.iterrows():
                try:
                    # Create base incident structure
                    incident = {}
                    
                    # Map columns to standardized fields
                    for excel_col, json_field in self.column_mapping.items():
                        if excel_col in df.columns:
                            value = row[excel_col]
                            
                            # Handle different data types
                            if pd.isna(value):
                                incident[json_field] = None
                            elif json_field in ['date_submitted', 'resolution_date']:
                                # Handle date fields
                                incident[json_field] = self._parse_date(value)
                            elif json_field in ['tags']:
                                # Handle list fields
                                incident[json_field] = self._parse_tags(value)
                            else:
                                # Handle string fields
                                incident[json_field] = str(value).strip() if value else None
                    
                    # Generate ID if missing
                    if not incident.get('id'):
                        incident['id'] = f"INC_{str(uuid.uuid4())[:8]}"
                    
                    # Add processing metadata
                    incident['processing_metadata'] = {
                        'source_row': index,
                        'processed_at': datetime.now().isoformat(),
                        'data_version': '1.0.0'
                    }
                    
                    # Add derived fields
                    incident['text_content'] = self._create_text_content(incident)
                    incident['keywords'] = self.text_processor.extract_keywords(incident['text_content'])
                    
                    structured_incidents.append(incident)
                    self.processing_stats['processed_records'] += 1
                    
                except Exception as e:
                    self.logger.error(f"Error processing row {index}: {str(e)}")
                    self.processing_stats['failed_records'] += 1
                    continue
            
            self.logger.info(f"Converted {len(structured_incidents)} incidents to structured format")
            return structured_incidents
            
        except Exception as e:
            self.logger.error(f"Error converting to structured format: {str(e)}")
            raise
    
    async def _enrich_incident_data(self, incidents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enrich incident data with additional processing"""
        try:
            self.logger.info("Starting data enrichment process")
            
            enriched_incidents = []
            
            for incident in incidents:
                try:
                    # Add data quality score
                    incident['data_quality_score'] = self._calculate_data_quality_score(incident)
                    
                    # Add complexity analysis
                    incident['complexity_analysis'] = self._analyze_incident_complexity(incident)
                    
                    # Add category insights
                    incident['category_insights'] = self._extract_category_insights(incident)
                    
                    # Add urgency indicators
                    incident['urgency_indicators'] = self._extract_urgency_indicators(incident)
                    
                    # Prepare for embedding generation (will be done separately)
                    incident['ready_for_embedding'] = True
                    
                    enriched_incidents.append(incident)
                    
                except Exception as e:
                    self.logger.error(f"Error enriching incident {incident.get('id', 'unknown')}: {str(e)}")
                    # Keep the original incident without enrichment
                    enriched_incidents.append(incident)
            
            self.logger.info(f"Enriched {len(enriched_incidents)} incidents")
            return enriched_incidents
            
        except Exception as e:
            self.logger.error(f"Error in data enrichment: {str(e)}")
            return incidents
    
    def _parse_date(self, date_value) -> Optional[str]:
        """Parse date value to ISO format string"""
        try:
            if pd.isna(date_value):
                return None
            
            # If it's already a datetime object
            if isinstance(date_value, pd.Timestamp):
                return date_value.isoformat()
            
            # If it's a string, try to parse it
            if isinstance(date_value, str):
                # Try common date formats
                date_formats = [
                    '%Y-%m-%d',
                    '%d-%m-%Y',
                    '%m/%d/%Y',
                    '%d/%m/%Y',
                    '%Y-%m-%d %H:%M:%S',
                    '%d-%m-%Y %H:%M'
                ]
                
                for fmt in date_formats:
                    try:
                        parsed_date = datetime.strptime(date_value.strip(), fmt)
                        return parsed_date.isoformat()
                    except ValueError:
                        continue
            
            # If all parsing attempts fail, return the original value as string
            return str(date_value)
            
        except Exception:
            return None
    
    def _parse_tags(self, tags_value) -> List[str]:
        """Parse tags from various formats"""
        try:
            if pd.isna(tags_value) or not tags_value:
                return []
            
            tags_str = str(tags_value).strip()
            
            # Split by common delimiters
            delimiters = [',', ';', '|', '\n']
            for delimiter in delimiters:
                if delimiter in tags_str:
                    tags = [tag.strip() for tag in tags_str.split(delimiter)]
                    return [tag for tag in tags if tag]  # Remove empty tags
            
            # If no delimiters found, return as single tag
            return [tags_str] if tags_str else []
            
        except Exception:
            return []
    
    def _create_text_content(self, incident: Dict[str, Any]) -> str:
        """Create combined text content for embedding generation"""
        text_parts = []
        
        # Add summary
        if incident.get('summary'):
            text_parts.append(incident['summary'])
        
        # Add description
        if incident.get('description'):
            text_parts.append(incident['description'])
        
        # Add category
        if incident.get('category'):
            text_parts.append(f"Category: {incident['category']}")
        
        # Add resolution if available
        if incident.get('resolution'):
            text_parts.append(f"Resolution: {incident['resolution']}")
        
        return " ".join(text_parts)
    
    def _calculate_data_quality_score(self, incident: Dict[str, Any]) -> float:
        """Calculate data quality score for an incident"""
        try:
            score = 0.0
            max_score = 0.0
            
            # Required fields (higher weight)
            required_checks = [
                ('id', 0.1),
                ('summary', 0.2),
                ('description', 0.2),
                ('category', 0.15)
            ]
            
            for field, weight in required_checks:
                max_score += weight
                if incident.get(field) and str(incident[field]).strip():
                    score += weight
            
            # Optional fields (lower weight)
            optional_checks = [
                ('severity', 0.1),
                ('priority', 0.1),
                ('date_submitted', 0.05),
                ('reporter', 0.05),
                ('resolution', 0.05)
            ]
            
            for field, weight in optional_checks:
                max_score += weight
                if incident.get(field) and str(incident[field]).strip():
                    score += weight
            
            # Content quality checks
            summary_length = len(incident.get('summary', ''))
            description_length = len(incident.get('description', ''))
            
            # Summary quality (good length range)
            if 20 <= summary_length <= 200:
                score += 0.05
            max_score += 0.05
            
            # Description quality (good length range)
            if 50 <= description_length <= 1000:
                score += 0.05
            max_score += 0.05
            
            return min(score / max_score, 1.0) if max_score > 0 else 0.0
            
        except Exception:
            return 0.5  # Default score if calculation fails
    
    def _analyze_incident_complexity(self, incident: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze incident complexity"""
        try:
            complexity = {
                'score': 0.0,
                'factors': [],
                'level': 'low'
            }
            
            text_content = incident.get('text_content', '')
            
            # Length-based complexity
            if len(text_content) > 500:
                complexity['score'] += 0.3
                complexity['factors'].append('lengthy_description')
            
            # Technical terms detection
            technical_terms = ['error', 'exception', 'timeout', 'connection', 'server', 'database', 'network']
            tech_count = sum(1 for term in technical_terms if term.lower() in text_content.lower())
            
            if tech_count >= 3:
                complexity['score'] += 0.4
                complexity['factors'].append('multiple_technical_terms')
            elif tech_count >= 1:
                complexity['score'] += 0.2
                complexity['factors'].append('technical_terms')
            
            # Priority/severity impact
            severity = incident.get('severity', '').lower()
            priority = incident.get('priority', '').lower()
            
            if 'high' in severity or 'critical' in severity:
                complexity['score'] += 0.2
                complexity['factors'].append('high_severity')
            
            if 'high' in priority or 'critical' in priority:
                complexity['score'] += 0.1
                complexity['factors'].append('high_priority')
            
            # Determine complexity level
            if complexity['score'] >= 0.7:
                complexity['level'] = 'high'
            elif complexity['score'] >= 0.4:
                complexity['level'] = 'medium'
            else:
                complexity['level'] = 'low'
            
            return complexity
            
        except Exception:
            return {'score': 0.5, 'factors': [], 'level': 'medium'}
    
    def _extract_category_insights(self, incident: Dict[str, Any]) -> Dict[str, Any]:
        """Extract insights about the incident category"""
        try:
            category = incident.get('category', '').lower()
            insights = {
                'primary_system': 'unknown',
                'service_type': 'unknown',
                'typical_resolution_time': 'unknown'
            }
            
            # System identification
            if 'lms' in category or 'learning' in category:
                insights['primary_system'] = 'Learning Management System'
                insights['service_type'] = 'educational_platform'
                insights['typical_resolution_time'] = '2-4 hours'
            elif 'email' in category or 'exchange' in category:
                insights['primary_system'] = 'Email System'
                insights['service_type'] = 'communication'
                insights['typical_resolution_time'] = '1-2 hours'
            elif 'network' in category or 'connectivity' in category:
                insights['primary_system'] = 'Network Infrastructure'
                insights['service_type'] = 'infrastructure'
                insights['typical_resolution_time'] = '1-6 hours'
            elif 'database' in category or 'sql' in category:
                insights['primary_system'] = 'Database System'
                insights['service_type'] = 'data_management'
                insights['typical_resolution_time'] = '2-8 hours'
            
            return insights
            
        except Exception:
            return {'primary_system': 'unknown', 'service_type': 'unknown', 'typical_resolution_time': 'unknown'}
    
    def _extract_urgency_indicators(self, incident: Dict[str, Any]) -> List[str]:
        """Extract urgency indicators from incident text"""
        try:
            text_content = incident.get('text_content', '').lower()
            urgency_patterns = [
                r'\b(urgent|emergency|critical|asap|immediately)\b',
                r'\b(down|outage|failure|broken|not working)\b',
                r'\b(production|live|critical system)\b',
                r'\b(all users|multiple users|everyone)\b',
                r'\b(cannot|unable|failed to)\b'
            ]
            
            indicators = []
            for pattern in urgency_patterns:
                matches = re.findall(pattern, text_content)
                indicators.extend(matches)
            
            return list(set(indicators))  # Remove duplicates
            
        except Exception:
            return []
    
    async def _save_to_json(self, data: List[Dict[str, Any]], output_path: str):
        """Save processed data to JSON file"""
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save with proper formatting
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"Saved {len(data)} incidents to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving to JSON: {str(e)}")
            raise
    
    def _calculate_processing_statistics(self, total_input: int, total_output: int):
        """Calculate final processing statistics"""
        try:
            success_rate = (self.processing_stats['processed_records'] / total_input) * 100 if total_input > 0 else 0
            
            # Calculate overall data quality score
            quality_scores = []
            # This would ideally be calculated from actual incident data
            # For now, we'll estimate based on processing success
            self.processing_stats['data_quality_score'] = min(success_rate / 100, 1.0)
            
            self.processing_stats.update({
                'success_rate': success_rate,
                'total_input_records': total_input,
                'total_output_records': total_output,
                'processing_timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            self.logger.error(f"Error calculating statistics: {str(e)}")
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        return self.processing_stats.copy()

# Async utility functions for batch processing
async def process_multiple_files(file_paths: List[str], output_dir: str) -> Dict[str, Any]:
    """Process multiple Excel files in parallel"""
    processor = DataIngestionProcessor()
    await processor.initialize()
    
    results = []
    for file_path in file_paths:
        try:
            output_path = os.path.join(output_dir, f"{Path(file_path).stem}_processed.json")
            result = await processor.process_excel_file(file_path, output_path)
            results.append({
                'file': file_path,
                'status': 'success',
                'result': result
            })
        except Exception as e:
            results.append({
                'file': file_path,
                'status': 'error',
                'error': str(e)
            })
    
    return {
        'batch_results': results,
        'total_files': len(file_paths),
        'successful_files': len([r for r in results if r['status'] == 'success']),
        'failed_files': len([r for r in results if r['status'] == 'error'])
    }

# Command-line interface for standalone usage
async def main():
    """Main function for CLI usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process Excel incident data to JSON format')
    parser.add_argument('input_file', help='Path to input Excel file')
    parser.add_argument('--output', '-o', help='Output JSON file path')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        processor = DataIngestionProcessor()
        await processor.initialize()
        
        output_path = args.output or f"{Path(args.input_file).stem}_processed.json"
        result = await processor.process_excel_file(args.input_file, output_path)
        
        print(f"Processing completed successfully!")
        print(f"Processed {result['statistics']['processed_records']} incidents")
        print(f"Data quality score: {result['statistics']['data_quality_score']:.2f}")
        print(f"Output saved to: {output_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))