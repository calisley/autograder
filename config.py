#!/usr/bin/env python3
"""
Centralized configuration for the autograder system.
"""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

@dataclass
class AzureConfig:
    """Azure OpenAI configuration"""
    endpoint_gpt: str
    api_key_gpt: str
    endpoint_doc_intel: str
    api_key_doc_intel: str
    api_version: str = "2024-12-01-preview"

@dataclass
class RateLimits:
    """Rate limiting configuration"""
    requests_per_minute: int = 100
    tokens_per_minute: int = 100_000
    embedding_concurrent: int = 3
    question_concurrent: int = 3

@dataclass
class ModelConfig:
    """Model configuration"""
    default_model: str = "gpt-5-mini"
    embedding_model: str = "text-embedding-3-large"
    encoder_model: str = "o200k_base"
    max_tokens: int = 8192

@dataclass
class ProcessingConfig:
    """Processing configuration"""
    batch_size: int = 10
    sample_size: int = 10
    top_k_pages: int = 3
    max_dimension: int = 15000
    dpi: int = 200
    quality: int = 90

class Config:
    """Main configuration class"""
    
    def __init__(self):
        self.azure = self._load_azure_config()
        self.rate_limits = RateLimits()
        self.models = ModelConfig()
        self.processing = ProcessingConfig()
    
    def _load_azure_config(self) -> AzureConfig:
        """Load Azure configuration from environment variables"""
        # Load environment variables from .env file
        load_dotenv()
        
        return AzureConfig(
            endpoint_gpt=os.getenv("AZURE_ENDPOINT_GPT"),
            api_key_gpt=os.getenv("AZURE_API_KEY_GPT"),
            endpoint_doc_intel=os.getenv("AZURE_ENDPOINT"),
            api_key_doc_intel=os.getenv("AZURE_API_KEY")
        )
    
    def validate(self) -> bool:
        """Validate that all required configuration is present"""
        if not self.azure.endpoint_gpt or not self.azure.api_key_gpt:
            raise ValueError("Azure GPT credentials (AZURE_ENDPOINT_GPT, AZURE_API_KEY_GPT) are required")
        
        if not self.azure.endpoint_doc_intel or not self.azure.api_key_doc_intel:
            raise ValueError("Azure Document Intelligence credentials (AZURE_ENDPOINT, AZURE_API_KEY) are required")
        
        return True

# Global configuration instance
config = Config()

# Validate configuration on import
config.validate()
