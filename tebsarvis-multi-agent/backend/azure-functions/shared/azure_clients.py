"""
Azure Service Clients for TEBSarvis Multi-Agent System
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
from openai import AsyncAzureOpenAI  # Using Azure OpenAI
import numpy as np
from ...config.azure_config import get_azure_config

class AzureOpenAIClient:
    """
    Async client for Azure OpenAI services.
    Handles chat completions, embeddings, and other AI operations.
    """
    
    def __init__(self):
        config = get_azure_config()
        openai_config = config.openai
        
        self.client = AsyncAzureOpenAI(
            api_key=openai_config.api_key,
            api_version=openai_config.api_version,
            azure_endpoint=openai_config.endpoint
        )
        self.embedding_model = openai_config.embedding_model
        self.chat_model = openai_config.chat_model
        self.logger = logging.getLogger("azure.openai")

    async def get_chat_completion(self, messages: List[Dict[str, str]], 
                                temperature: float = 0.7, max_tokens: int = 1000,
                                system_prompt: Optional[str] = None) -> str:
        """
        Get chat completion from OpenAI.
        """
        try:
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
        Get embeddings from OpenAI.
        """
        try:
            if isinstance(texts, str):
                texts = [texts]
                
            response = await self.client.embeddings.create(
                model=self.embedding_model,
                input=texts
            )
            
            embeddings = [data.embedding for data in response.data]
            return embeddings[0] if len(embeddings) == 1 else embeddings
            
        except Exception as e:
            self.logger.error(f"Embedding error: {str(e)}")
            raise

    async def get_streaming_completion(self, messages: List[Dict[str, str]], 
                                    temperature: float = 0.7, max_tokens: int = 1000):
        """
        Get streaming chat completion from OpenAI.
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