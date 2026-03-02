from dataclasses import dataclass
from typing import Dict, Optional, Type, Union
from enum import Enum
from .query_models import QueryModel, EmbeddingModel


@dataclass
class ModelConfig:
    """Configuration for a model including its display name, API identifier, and type."""
    display_name: str
    model_id: str
    model_type: str = "query"  
    temperature: float = 0.0
    seed: int = 42


class ModelType(Enum):
    """Model types available in the system."""
    QUERY = "query"
    EMBEDDING = "embedding"


class ModelRegistry:
    """Central registry for all models in the application."""
    
    _model_configs: Dict[str, ModelConfig] = {
        "2.0-flash": ModelConfig(
            display_name="Gemini 2.0 Flash",
            model_id="gemini/gemini-2.0-flash",
            model_type="query"
        ),
        "2.5-flash": ModelConfig(
            display_name="Gemini 2.5 Flash",
            model_id="gemini/gemini-2.5-flash",
            model_type="query"
        ),
        "llama-3.3-70b": ModelConfig(
            display_name="Groq Llama 3.3 70B Versatile", 
            model_id="groq/llama-3.3-70b-versatile",
            model_type="query"
        ),
        "2.0-flash-lite": ModelConfig(
            display_name="Gemini 2.0 Flash Lite Safety",
            model_id="gemini/gemini-2.0-flash-lite",
            model_type="query"
        ),
        "gpt-5-mini": ModelConfig(
            display_name="GPT-5 Mini",
            model_id="azure/gpt-5-mini",
            model_type="query"
        ),
        "llama-4-scout-17b": ModelConfig(
            display_name="Llama 4 Scout 17B",
            model_id="azure/llama-4-scout-17b",
            model_type="query"
        ),
        "deepseek-v3.2": ModelConfig(
            display_name="DeepSeek V3.2",
            model_id="azure/deepseek-v3.2",
            model_type="query"
        ),
        "mistral-large-3": ModelConfig(
            display_name="Mistral Large 3",
            model_id="azure/mistral-large-3",
            model_type="query"
        ),
        # Embedding Models
        "voyage-v3": ModelConfig(
            display_name="Voyage AI v3",
            model_id="voyage-3",
            model_type="embedding"
        ),
    }
    
    # Cache for created model instances
    _model_cache: Dict[str, Union[QueryModel, EmbeddingModel]] = {}
    
    @classmethod
    def get_model(cls, model_name: str) -> Union[QueryModel, EmbeddingModel]:
        """
        Get a model instance by name. Creates and caches instances automatically.
        
        Args:
            model_name: The model identifier (e.g., "gemini", "groq", "voyage")
            
        Returns:
            QueryModel or EmbeddingModel instance
            
        Raises:
            KeyError: If model_name is not registered
            ValueError: If model type is not supported
        """
        if model_name not in cls._model_configs:
            available = ", ".join(cls._model_configs.keys())
            raise KeyError(f"Model '{model_name}' not found. Available: {available}")
        
        # Return cached instance if exists
        if model_name in cls._model_cache:
            return cls._model_cache[model_name]
        
        config = cls._model_configs[model_name]
        
        # Create appropriate model instance based on type
        if config.model_type == ModelType.QUERY.value:
            model = QueryModel(config.display_name, config.model_id)
            model.temperature = config.temperature
            model.seed = config.seed
        elif config.model_type == ModelType.EMBEDDING.value:
            model = EmbeddingModel(config.display_name, config.model_id)
        else:
            raise ValueError(f"Unsupported model type: {config.model_type}")
        
        # Cache and return
        cls._model_cache[model_name] = model
        return model
    
    @classmethod
    def get_available_models(cls, model_type: Optional[str] = None) -> Dict[str, ModelConfig]:
        """
        Get all available models, optionally filtered by type.
        
        Args:
            model_type: Filter by "query" or "embedding", or None for all
            
        Returns:
            Dictionary of model_name -> ModelConfig
        """
        if model_type is None:
            return cls._model_configs.copy()
        
        return {
            name: config for name, config in cls._model_configs.items()
            if config.model_type == model_type
        }
    
    @classmethod 
    def get_model_info(cls, model_name: str) -> ModelConfig:
        """Get configuration info for a model without creating instance."""
        if model_name not in cls._model_configs:
            available = ", ".join(cls._model_configs.keys())
            raise KeyError(f"Model '{model_name}' not found. Available: {available}")
        return cls._model_configs[model_name]


def get_query_model(model_name: str) -> QueryModel:
    """Get a query model instance. Validates it's actually a query model."""
    model = ModelRegistry.get_model(model_name)
    if not isinstance(model, QueryModel):
        raise ValueError(f"Model '{model_name}' is not a query model")
    return model


def get_embedding_model(model_name: str) -> EmbeddingModel:
    """Get an embedding model instance. Validates it's actually an embedding model.""" 
    model = ModelRegistry.get_model(model_name)
    if not isinstance(model, EmbeddingModel):
        raise ValueError(f"Model '{model_name}' is not an embedding model")
    return model


def list_query_models() -> list[str]:
    """Get list of available query model names."""
    return list(ModelRegistry.get_available_models("query").keys())


def list_embedding_models() -> list[str]:
    """Get list of available embedding model names."""
    return list(ModelRegistry.get_available_models("embedding").keys())