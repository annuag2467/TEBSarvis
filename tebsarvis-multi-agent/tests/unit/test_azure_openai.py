"""
Test Azure OpenAI configuration and connectivity
"""

import os
import asyncio
import sys
from pathlib import Path

# Add the backend path to sys.path
backend_path = Path(__file__).resolve().parent.parent.parent / 'tebsarvis-multi-agent' / 'backend'
sys.path.append(str(backend_path))

from azure_functions.shared.azure_clients import AzureOpenAIClient

async def test_azure_openai_connection():
    """Test the Azure OpenAI connection by making a simple API call"""
    try:
        client = AzureOpenAIClient()
        messages = [{"role": "user", "content": "Hello, this is a test message"}]
        
        print("Testing Azure OpenAI connection...")
        response = await client.get_chat_completion(messages)
        
        print("Connection successful!")
        print(f"Response: {response}")
        return True
        
    except Exception as e:
        print(f"Error testing Azure OpenAI connection: {str(e)}")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_azure_openai_connection())
    sys.exit(0 if result else 1)
