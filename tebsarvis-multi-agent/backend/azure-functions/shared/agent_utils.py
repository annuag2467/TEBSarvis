"""
Agent Utility Functions for TEBSarvis Multi-Agent System
Shared utilities and helper functions for agents and Azure Functions.
"""

import asyncio
import logging
import json
import uuid
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime, timedelta
from functools import wraps
import traceback
import time
import re
from dataclasses import asdict

class AgentError(Exception):
    """Custom exception for agent-related errors"""
    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}

class ValidationError(AgentError):
    """Exception for data validation errors"""
    pass

class TimeoutError(AgentError):
    """Exception for timeout errors"""
    pass

def async_retry(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0, 
                exceptions: tuple = (Exception,)):
    """
    Decorator for retrying async functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Backoff multiplier for delay
        exceptions: Tuple of exceptions to catch and retry
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        break
                    
                    logger = logging.getLogger(func.__module__)
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {str(e)}")
                    
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
            
            raise last_exception
        return wrapper
    return decorator

def timeout(seconds: int):
    """
    Decorator to add timeout to async functions.
    
    Args:
        seconds: Timeout in seconds
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")
        return wrapper
    return decorator

def log_performance(logger: Optional[logging.Logger] = None):
    """
    Decorator to log function performance metrics.
    
    Args:
        logger: Logger instance to use
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            func_logger = logger or logging.getLogger(func.__module__)
            
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                func_logger.info(f"{func.__name__} completed in {execution_time:.3f}s")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                func_logger.error(f"{func.__name__} failed after {execution_time:.3f}s: {str(e)}")
                raise
        return wrapper
    return decorator

class TaskValidator:
    """Validator for agent task data"""
    
    @staticmethod
    def validate_task_data(task_data: Dict[str, Any], required_fields: List[str] = None) -> Dict[str, Any]:
        """
        Validate task data structure.
        
        Args:
            task_data: Task data to validate
            required_fields: List of required field names
            
        Returns:
            Validated task data
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(task_data, dict):
            raise ValidationError("Task data must be a dictionary")
        
        required_fields = required_fields or ['type']
        
        for field in required_fields:
            if field not in task_data:
                raise ValidationError(f"Required field '{field}' missing from task data")
        
        # Validate task type
        if 'type' in task_data and not isinstance(task_data['type'], str):
            raise ValidationError("Task type must be a string")
        
        return task_data
    
    @staticmethod
    def validate_incident_data(incident_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate incident data structure.
        
        Args:
            incident_data: Incident data to validate
            
        Returns:
            Validated incident data
        """
        required_fields = ['summary', 'category']
        
        if not isinstance(incident_data, dict):
            raise ValidationError("Incident data must be a dictionary")
        
        for field in required_fields:
            if field not in incident_data or not incident_data[field]:
                raise ValidationError(f"Required field '{field}' missing or empty in incident data")
        
        # Validate field types
        string_fields = ['summary', 'description', 'category', 'priority', 'severity']
        for field in string_fields:
            if field in incident_data and not isinstance(incident_data[field], str):
                raise ValidationError(f"Field '{field}' must be a string")
        
        return incident_data

class ResponseFormatter:
    """Formatter for agent responses"""
    
    @staticmethod
    def format_success_response(data: Any, metadata: Dict[str, Any] = None, 
                              message: str = "Success") -> Dict[str, Any]:
        """
        Format a successful response.
        
        Args:
            data: Response data
            metadata: Optional metadata
            message: Success message
            
        Returns:
            Formatted response dictionary
        """
        response = {
            "success": True,
            "message": message,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        
        if metadata:
            response["metadata"] = metadata
            
        return response
    
    @staticmethod
    def format_error_response(error: Union[str, Exception], 
                            error_code: str = None,
                            details: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Format an error response.
        
        Args:
            error: Error message or exception
            error_code: Optional error code
            details: Optional error details
            
        Returns:
            Formatted error response dictionary
        """
        if isinstance(error, Exception):
            error_message = str(error)
            error_type = type(error).__name__
        else:
            error_message = error
            error_type = "Error"
        
        response = {
            "success": False,
            "error": {
                "message": error_message,
                "type": error_type,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        if error_code:
            response["error"]["code"] = error_code
            
        if details:
            response["error"]["details"] = details
            
        return response

class TextProcessor:
    """Utility class for text processing operations"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', '', text)
        
        return text
    
    @staticmethod
    def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
        """
        Extract keywords from text.
        
        Args:
            text: Text to extract keywords from
            max_keywords: Maximum number of keywords to return
            
        Returns:
            List of keywords
        """
        if not text:
            return []
        
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter out common stop words
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 
            'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 
            'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 
            'two', 'who', 'boy', 'did', 'she', 'use', 'way', 'will', 'with'
        }
        
        keywords = [word for word in words if word not in stop_words and len(word) > 3]
        
        # Count frequency and return most common
        from collections import Counter
        word_counts = Counter(keywords)
        return [word for word, count in word_counts.most_common(max_keywords)]
    
    @staticmethod
    def calculate_similarity(text1: str, text2: str) -> float:
        """
        Calculate simple text similarity using Jaccard similarity.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        if not text1 or not text2:
            return 0.0
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)

class DataTransformer:
    """Utility class for data transformation operations"""
    
    @staticmethod
    def incident_to_search_doc(incident: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform incident data to search document format.
        
        Args:
            incident: Incident data
            
        Returns:
            Search document
        """
        return {
            'id': incident.get('id', str(uuid.uuid4())),
            'content': f"{incident.get('summary', '')} {incident.get('description', '')}",
            'metadata': {
                'category': incident.get('category'),
                'severity': incident.get('severity'),
                'priority': incident.get('priority'),
                'date_submitted': incident.get('date_submitted'),
                'resolution': incident.get('resolution', ''),
                'status': incident.get('status', 'Open')
            }
        }
    
    @staticmethod
    def flatten_nested_dict(data: Dict[str, Any], prefix: str = '') -> Dict[str, Any]:
        """
        Flatten a nested dictionary.
        
        Args:
            data: Nested dictionary to flatten
            prefix: Prefix for keys
            
        Returns:
            Flattened dictionary
        """
        flattened = {}
        
        for key, value in data.items():
            new_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                flattened.update(DataTransformer.flatten_nested_dict(value, new_key))
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        flattened.update(DataTransformer.flatten_nested_dict(item, f"{new_key}[{i}]"))
                    else:
                        flattened[f"{new_key}[{i}]"] = item
            else:
                flattened[new_key] = value
        
        return flattened

class CacheManager:
    """Simple in-memory cache manager"""
    
    def __init__(self, default_ttl: int = 300):
        self.cache = {}
        self.default_ttl = default_ttl
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key in self.cache:
            item = self.cache[key]
            if item['expires_at'] > datetime.now():
                return item['value']
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache"""
        ttl = ttl or self.default_ttl
        expires_at = datetime.now() + timedelta(seconds=ttl)
        
        self.cache[key] = {
            'value': value,
            'expires_at': expires_at
        }
    
    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        if key in self.cache:
            del self.cache[key]
            return True
        return False
    
    def clear(self) -> None:
        """Clear all cache"""
        self.cache.clear()
    
    def cleanup_expired(self) -> int:
        """Remove expired items and return count removed"""
        now = datetime.now()
        expired_keys = [
            key for key, item in self.cache.items()
            if item['expires_at'] <= now
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        return len(expired_keys)

class MetricsCollector:
    """Utility for collecting and managing metrics"""
    
    def __init__(self):
        self.metrics = {}
        self.start_time = datetime.now()
    
    def increment(self, metric_name: str, value: int = 1) -> None:
        """Increment a counter metric"""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = 0
        self.metrics[metric_name] += value
    
    def set_gauge(self, metric_name: str, value: float) -> None:
        """Set a gauge metric value"""
        self.metrics[metric_name] = value
    
    def record_timing(self, metric_name: str, duration: float) -> None:
        """Record a timing metric"""
        timing_key = f"{metric_name}_timing"
        if timing_key not in self.metrics:
            self.metrics[timing_key] = []
        self.metrics[timing_key].append(duration)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics"""
        return {
            'metrics': self.metrics.copy(),
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds(),
            'timestamp': datetime.now().isoformat()
        }
    
    def reset(self) -> None:
        """Reset all metrics"""
        self.metrics.clear()
        self.start_time = datetime.now()

# Utility functions
def generate_correlation_id() -> str:
    """Generate a correlation ID for tracking requests"""
    return str(uuid.uuid4())

def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """Safely parse JSON string"""
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default

def safe_json_dumps(obj: Any, default: str = "{}") -> str:
    """Safely serialize object to JSON"""
    try:
        return json.dumps(obj, default=str, indent=2)
    except (TypeError, ValueError):
        return default

def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate string to maximum length"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

def batch_list(items: List[Any], batch_size: int) -> List[List[Any]]:
    """Split list into batches"""
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple dictionaries"""
    result = {}
    for d in dicts:
        if d:
            result.update(d)
    return result

def get_nested_value(data: Dict[str, Any], path: str, default: Any = None) -> Any:
    """Get value from nested dictionary using dot notation"""
    try:
        keys = path.split('.')
        current = data
        for key in keys:
            current = current[key]
        return current
    except (KeyError, TypeError):
        return default

def set_nested_value(data: Dict[str, Any], path: str, value: Any) -> None:
    """Set value in nested dictionary using dot notation"""
    keys = path.split('.')
    current = data
    
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    current[keys[-1]] = value

# Global instances
cache_manager = CacheManager()
metrics_collector = MetricsCollector()