"""Agent model management and reward functions for multi-agent debate training."""

import re
import os
import math
import warnings
import torch
import numpy as np
from tqdm import tqdm
from peft import LoraConfig, PeftModel, PeftConfig, prepare_model_for_kbit_training
from typing import List, Dict, Any, Optional, Callable, Union, Tuple
from utils import set_random_seed, WandbLoggingCallback
from parser import parse_answer, grade_answer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import (
    SFTTrainer, SFTConfig,
    GRPOConfig, GRPOTrainer,
    KTOConfig, KTOTrainer,
    DPOConfig, DPOTrainer,
)
import asyncio
import time
import logging
from utils import pd
import shutil

logger = logging.getLogger(__name__)

class BaseModelManager:
    """Manager for handling multiple base models with adapter swapping capability."""
    
    # Class-level storage for tracking base models and a condition for synchronization
    _base_models = {}
    _condition = asyncio.Condition()
    
    @classmethod
    async def acquire_model(cls) -> Optional[str]:
        """
        Acquire an available base model from the pool using a condition variable.
        This method will wait efficiently until a model is free, mark it as in-use, and return its key.
        """
        async with cls._condition:
            # Wait until a model is available. The lambda function is re-evaluated whenever notified.
            await cls._condition.wait_for(lambda: any(not info['in_use'] for info in cls._base_models.values()))
            
            # Now that we are sure a model is free, find it and acquire it.
            for model_key, info in cls._base_models.items():
                if not info['in_use']:
                    info['in_use'] = True
                    info['last_used'] = time.time()
                    return model_key
            # This part should not be reachable due to the wait_for logic.

    @classmethod
    async def release_model(cls, model_key: str) -> None:
        """Release a model and notify one waiting coroutine."""
        async with cls._condition:
            if model_key in cls._base_models:
                if cls._base_models[model_key]['in_use']:
                    cls._base_models[model_key]['in_use'] = False
                    # Notify one waiting coroutine that a model has become available.
                    cls._condition.notify()
                else:
                    pd(f"Warning: Attempted to release model {model_key} that was not marked as in-use.")
            else:
                pd(f"Warning: Tried to release a model key that does not exist: {model_key}")

    @classmethod
    def get_or_create_base_model(
        cls,
        model_id: str,
        quantization_config=None,
        **model_kwargs
    ) -> Tuple[AutoModelForCausalLM, str]:
        """
        Get or create a base model with the specified configuration.
        
        Args:
            model_id: The model identifier
            quantization_config: Optional quantization configuration
            **model_kwargs: Additional model configuration parameters, can include:
                - _model_key_override: A specific key to use for this model instance.
                - _gpu_ids: A list of GPU IDs this model is assigned to.
            
        Returns:
            Tuple of (model, model_key)
        """
        # Pop custom args from model_kwargs so they aren't passed to from_pretrained
        model_key_override = model_kwargs.pop('_model_key_override', None)
        gpu_ids = model_kwargs.pop('_gpu_ids', None)

        device_map = model_kwargs.get('device_map', 'auto')
        
        # Use override if provided, otherwise generate key
        if model_key_override:
            model_key = model_key_override
        else:
            model_key = f"{model_id}_{device_map}"
        
        if model_key not in cls._base_models:
            pd(f"Creating new base model: {model_id} with device_map={device_map}")
            
            # Create the model
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                **model_kwargs
            )
            
            # Verify GPU placement
            try:
                param_device = next(model.parameters()).device
                pd(f"Verified model {model_key} is on {param_device}")
                if str(param_device) != device_map and device_map != 'auto':
                    print(f"WARNING: Model {model_key} was requested on {device_map} but is actually on {param_device}")
            except Exception as e:
                print(f"Error verifying model placement: {e}")
            
            # Store model info
            cls._base_models[model_key] = {
                "model": model,
                "in_use": False,
                "last_used": time.time(),
                "device": device_map,
                "gpu_ids": gpu_ids, # Store the assigned GPU IDs
                "adapters": {},  # Maps adapter_name -> (adapter_path, adapter_config)
                "active_adapter": None,
            }
            
            pd(f"Loaded base model {model_key} on {device_map}")
            
        return cls._base_models[model_key]["model"], model_key

    @classmethod
    def add_adapter(
        cls,
        model_key: str,
        adapter_path: str,
        adapter_name: str = None
    ) -> None:
        """
        Add an adapter to a base model if it doesn't exist.
        Uses caching to avoid reloading the same adapter.
        """
        if model_key not in cls._base_models:
            raise ValueError(f"Model {model_key} not found")
            
        model_info = cls._base_models[model_key]
        model = model_info["model"]
        
        # Generate a consistent adapter name if not provided
        if adapter_name is None:
            adapter_name = f"adapter_{os.path.basename(adapter_path)}"
            
        # Check if adapter is already loaded
        if adapter_name in model_info["adapters"]:
            pd(f"Adapter {adapter_name} already loaded on model {model_key}")
            return
            
        pd(f"Loading adapter from {adapter_path} onto model {model_key}")
        
        # If the model is not yet a PeftModel, wrap it while loading the first adapter.
        # Otherwise, if it's already a PeftModel, just load the new adapter.
        if not isinstance(model, PeftModel):
            pd(f"Model {model_key} is not a PeftModel. Wrapping with first adapter.")
            model = PeftModel.from_pretrained(model, adapter_path, adapter_name=adapter_name)
            model_info["model"] = model  # Update the stored model to the new PeftModel
        else:
            pd(f"Model {model_key} is already a PeftModel. Loading additional adapter.")
            model.load_adapter(adapter_path, adapter_name=adapter_name)

        # Cache the adapter info
        adapter_config = model.peft_config[adapter_name]
        model_info["adapters"][adapter_name] = (adapter_path, adapter_config)
        pd(f"Successfully loaded adapter {adapter_name}")

    @classmethod
    def set_active_adapter(
        cls,
        model_key: str,
        adapter_path_or_name: str
    ) -> None:
        """
        Set the active adapter for a model.
        If the adapter isn't loaded, it will be loaded first.
        """
        if model_key not in cls._base_models:
            raise ValueError(f"Model {model_key} not found")
            
        model_info = cls._base_models[model_key]
        model = model_info["model"]
        
        # If adapter_path_or_name is a path, load it first
        if os.path.exists(adapter_path_or_name):
            adapter_name = f"adapter_{os.path.basename(adapter_path_or_name)}"
            cls.add_adapter(model_key, adapter_path_or_name, adapter_name)
        else:
            adapter_name = adapter_path_or_name
            
        # Verify adapter exists (may have been added just now)
        model_info = cls._base_models[model_key]  # refresh in case add_adapter replaced model
        model = model_info["model"]

        if adapter_name not in model_info["adapters"]:
            raise ValueError(f"Adapter {adapter_name} not found on model {model_key}")

        # Set as active adapter
        model.set_adapter(adapter_name)
        model_info["active_adapter"] = adapter_name
        pd(f"Successfully activated adapter {adapter_name}")

    @classmethod
    def remove_adapter(
        cls,
        model_key: str,
        adapter_name: str
    ) -> None:
        """
        Remove an adapter from a model.
        Only removes if it's not the active adapter.
        """
        if model_key not in cls._base_models:
            raise ValueError(f"Model {model_key} not found")
            
        model_info = cls._base_models[model_key]
        model = model_info["model"]
        
        # Don't remove if it's the active adapter
        if model_info["active_adapter"] == adapter_name:
            pd(f"Not removing active adapter {adapter_name}")
            return
            
        if adapter_name in model_info["adapters"]:
            pd(f"Removing adapter {adapter_name}")
            # Check if model has the 'delete_adapter' method
            if hasattr(model, 'delete_adapter'):
                model.delete_adapter(adapter_name)
            del model_info["adapters"][adapter_name]
            pd(f"Successfully removed adapter {adapter_name}")

    @classmethod
    def delete_base_model(cls, model_key: str) -> None:
        """Delete a base model and clean up its resources entirely."""
        if model_key not in cls._base_models:
            return
            
        model_info = cls._base_models.pop(model_key, None)
        if not model_info:
            return

        model = model_info["model"]
        
        # Clean up all adapters
        if isinstance(model, PeftModel):
            for adapter_name in list(model.peft_config.keys()):
                try:
                    model.delete_adapter(adapter_name)
                except Exception as e:
                    print(f"Warning: Error removing adapter {adapter_name} during cleanup: {e}")
            
        # Clear model from memory
        del model
        
        # Force cleanup
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class Agent:
    """Language model agent for generation, fine-tuning, and RL training."""
    
    def __init__(
        self,
        model_name: str,
        checkpoint_path: Optional[str] = None,
        agent_id: int = 0,
        role: str = 'agent',
        device_map: str = 'auto',
        special_tokens: Optional[List[str]] = None,
        wandb_run: Optional[Any] = None,
        use_cache: bool = True,
        attn_implementation: str = 'flash_attention_2',
        devices: Optional[List[str]] = None,
        quantization: bool = False,
        adapter_mode: bool = False,
        seed: int = 0,
        task: str = 'gsm8k',
        for_training: bool = False
    ) -> None:
        """
        Initialize an agent with a specified model and configuration.
        
        Args:
            model_name: Name of the model to load (e.g., 'mistral7b', 'llama3b')
            checkpoint_path: Path to a saved checkpoint to load
            agent_id: Unique identifier for this agent
            role: Role descriptor for the agent
            device_map: Device mapping strategy for model loading
            special_tokens: Additional special tokens to add to the tokenizer
            wandb_run: Weights & Biases run object for logging
            use_cache: Whether to use KV cache in model for faster generation
            attn_implementation: Attention implementation to use
            devices: List of available devices
            quantization: Whether to use quantization for model loading
            adapter_mode: Whether to use adapter swapping (requires quantization)
            seed: Random seed for reproducibility
            task: Task type for parsing answers
            for_training: Whether this agent is being initialized for training (vs debate/inference)
        """
        # device_map = 'cpu'
        self.seed = seed + agent_id
        set_random_seed(self.seed)

        self.task = task
        self.adapter_mode = adapter_mode
        self.quantization = quantization  # Store quantization state
        
        self._model_id_map = {
            "mistral7b": "mistralai/Mistral-7B-Instruct-v0.3",
            "llama1b": "meta-llama/Llama-3.2-1B-Instruct",
            "llama3b": "meta-llama/Llama-3.2-3B-Instruct", 
            "llama8b": "meta-llama/Llama-3.1-8B-Instruct",
            "phi4b": "microsoft/Phi-3-mini-128k-instruct",
            "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
            "qwen2b": "Qwen/Qwen2.5-1.5B-Instruct",
            "gemma4b": "google/gemma-3-4b-it"
        }

        # For QLoRA training, use a specific device rather than auto distribution
        if quantization and device_map == 'auto' and for_training:
            if devices and len(devices) > 0:
                device_index = agent_id % len(devices)
                specific_device = devices[device_index]
                device_map = specific_device
            else:
                current_device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
                device_map = f'cuda:{current_device}' if torch.cuda.is_available() else 'cpu'
            pd(f"Agent {agent_id} - Using specific device mapping for QLoRA training: {device_map}")
            
        # Create proper BitsAndBytesConfig for quantization
        bnb_config = None
        if quantization:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.float16,  # Use float16 for better performance
                bnb_4bit_use_double_quant=True,
            )
            pd(f"Agent {agent_id} - QLoRA 4-bit quantization ENABLED with nf4 and double quantization")
        else:
            pd(f"Agent {agent_id} - Quantization DISABLED")

        self.model_id = self._model_id_map.get(model_name)
        if not self.model_id:
            raise ValueError(f"Invalid model name: {model_name}. Available models: {list(self._model_id_map.keys())}")
          
        self.role = role
        self.id = agent_id
        self.wandb_run = wandb_run
        self.device_map = device_map
        cache_dir='./checkpoints'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, cache_dir=cache_dir, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.base_model_key = None
        
        # Model loading with proper quantization config
        if adapter_mode and quantization:
            # In adapter mode, we get a shared base model and apply adapters
            self.model, self.base_model_key = BaseModelManager.get_or_create_base_model(
                self.model_id, 
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
                device_map=device_map,
                cache_dir=cache_dir,
                low_cpu_mem_usage=True,
                attn_implementation=attn_implementation
            )

            def _find_adapter_dir(root: str) -> Optional[str]:
                """Return a directory that contains adapter_config.json starting from *root*.

                Looks first at *root* itself, then in the newest checkpoint-* sub-dir.
                """
                if not root or not os.path.isdir(root):
                    return None

                cfg = os.path.join(root, "adapter_config.json")
                if os.path.isfile(cfg):
                    return root

                subs = [d for d in os.listdir(root) if d.startswith("checkpoint-")]
                if not subs:
                    return None
                subs.sort(key=lambda d: int(d.split("-")[1]))
                candidate = os.path.join(root, subs[-1])
                return candidate if os.path.isfile(os.path.join(candidate, "adapter_config.json")) else None

            if checkpoint_path and os.path.exists(checkpoint_path):
                adapter_dir = _find_adapter_dir(checkpoint_path)
                if adapter_dir:
                    BaseModelManager.set_active_adapter(self.base_model_key, adapter_dir)
                else:
                    pd(f"No adapter found under {checkpoint_path}; proceeding without loading")
        else:
            # Traditional path - direct model loading
            if checkpoint_path and os.path.exists(checkpoint_path):
                if quantization:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        checkpoint_path,
                        quantization_config=bnb_config,
                        torch_dtype=torch.float16,
                        device_map=device_map,
                        low_cpu_mem_usage=True,
                        attn_implementation=attn_implementation
                    )
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        checkpoint_path,
                        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                        device_map=device_map,
                        low_cpu_mem_usage=True,
                        attn_implementation=attn_implementation
                    )
            else:
                if quantization:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_id,
                        quantization_config=bnb_config,
                        torch_dtype=torch.float16,
                        device_map=device_map,
                        cache_dir=cache_dir,
                        low_cpu_mem_usage=True,
                        attn_implementation=attn_implementation
                    )
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_id,
                        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                        device_map=device_map,
                        cache_dir=cache_dir,
                        low_cpu_mem_usage=True,
                        attn_implementation=attn_implementation
                    )
        
        # Enable gradient checkpointing for quantized models during training
        if quantization and for_training:
            # Prepare model for PEFT training when using quantization
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=True,
            )
            pd(f"Agent {agent_id} - Prepared model for k-bit training (gradient checkpointing & input grad hook already enabled by PEFT)")
            
            # Disable caching during training when checkpointing is on to avoid runtime errors
            if hasattr(self.model, 'config'):
                self.model.config.use_cache = False

        # LLama is missing the chat template in Huggingface
        if model_name == 'llama':
            chat_template = open('chat_templates/llama-3-instruct.jinja').read()
            chat_template = chat_template.replace('    ', '').replace('\n', '')
            self.tokenizer.chat_template = chat_template

        if special_tokens:
            self.tokenizer.add_special_tokens({"additional_special_tokens": ["<user>", "<assistant>"]})
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        pd(f"Padding side: {self.tokenizer.padding_side}")
        
            
        # For Phi model, ensure loss computation happens on the same device as the forward pass
        if model_name == 'phi4b':
            # Create a custom loss function that ensures device consistency
            original_loss_fn = self.model.loss_function
            
            def custom_loss_function(logits, labels, **kwargs):
                # Move labels to the same device as logits
                if labels.device != logits.device:
                    labels = labels.to(logits.device)
                # Move any other tensors in kwargs to the same device
                for k, v in kwargs.items():
                    if isinstance(v, torch.Tensor) and v.device != logits.device:
                        kwargs[k] = v.to(logits.device)
                return original_loss_fn(logits=logits, labels=labels, **kwargs)
            
            # Replace the model's loss function
            self.model.loss_function = custom_loss_function
    
    def cleanup(self) -> None:
        """
        Properly clean up agent resources. Call this explicitly instead of relying on __del__.
        """
        # If using adapter mode, we don't release the model here anymore,
        # as it's managed by the central scheduler/orchestrator.
        self.base_model_key = None
        
        # Clear references to model components
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            self.tokenizer = None
        
        # Force garbage collection
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def __del__(self):
        """
        Fallback cleanup - should not be relied upon. Use cleanup() explicitly.
        """
        # The base model is managed centrally, so we don't release it from the agent's destructor.
        self.base_model_key = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()

    def get_device(self) -> torch.device:
        """
        Get the device where the model is located.
        
        Returns:
            Device of the model's parameters
        """
        return next(self.model.parameters()).device
    
    def save_checkpoint(
        self,
        checkpoint_path: str
    ) -> None:
        """
        Save model checkpoint, creating directories if needed.
        
        Args:
            checkpoint_path: Directory path to save the model
        """
        os.makedirs(checkpoint_path, exist_ok=True)
        
        if self.adapter_mode:
            # In adapter mode, save just the adapter weights
            if hasattr(self.model, "save_pretrained") and callable(self.model.save_pretrained):
                # If it's a PeftModel
                self.model.save_pretrained(checkpoint_path)
            else:
                print(f"Warning: Cannot save adapter weights - model doesn't have save_pretrained method")
        else:
            # Traditional save - save the whole model
            self.model.save_pretrained(checkpoint_path)

    def get_safe_max_length(self) -> int:
        """
        Get safe max context length based on model type and available GPU memory.
        
        Returns:
            Maximum safe sequence length for the current model and hardware
        """
        base_length = self.tokenizer.model_max_length
        
        # Model-specific safe lengths
        safe_lengths = {
            "meta-llama/Meta-Llama-3-8B": 4096,
            "meta-llama/Llama-3.2-3B-Instruct": 4096,
            "meta-llama/Llama-3.2-1B-Instruct": 4096,
            "mistralai/Mistral-7B-Instruct-v0.3": 8192,
            "microsoft/Phi-3-mini-128k-instruct": 2048,
            "Qwen/Qwen2.5-7B-Instruct": 8192,
            "Qwen/Qwen2.5-1.5B-Instruct": 8192,
            "google/gemma-3-4b-it": 8192
        }
        
        # Get safe length for current model
        safe_length = safe_lengths.get(self.model_id, 2048)  # Default to 2048 if unknown
        
        # Additional safety for GPU memory
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory
            # Reduce length if less than 24GB GPU memory
            if gpu_mem < 24 * (1024**3):  
                safe_length = min(safe_length, 4096)
        
        return min(safe_length, base_length)

    @torch.inference_mode()
    async def batch_generate(
        self,
        contexts_list: List[List[Dict[str, str]]],
        device: torch.device,
        top_p: float = 0.9,
        temperature: float = 1.0
    ) -> List[Dict]:
        """
        Generate responses for multiple contexts in parallel.
        
        Args:
            contexts_list: List of conversation contexts, each a list of message dicts
            device: Device to run generation on
            top_p: Nucleus sampling probability threshold
            temperature: Sampling temperature
            
        Returns:
            List of completion dictionaries with generated responses
        """
        self.model.eval()
        max_length = self.get_safe_max_length()
        
        # Apply chat template to each context
        input_texts = [
            self.tokenizer.apply_chat_template(
                context, 
                tokenize=False, 
                add_generation_prompt=True
            )
            for context in contexts_list
        ]
        
        # Batch encode all prompts
        encoded = self.tokenizer.batch_encode_plus(
                input_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt',
                return_attention_mask=True,
            )
        
        # Move tensors to device
        encoded = encoded.to(device)
        
        # Report input shape statistics
        input_shape = encoded.input_ids.shape
        max_tokens = encoded.input_ids.size(1)
        total_tokens = torch.sum(encoded.attention_mask).item()
        avg_tokens = total_tokens / len(contexts_list)
        
        # Base generation parameters
        generate_kwargs = {
            'input_ids': encoded.input_ids,
            'attention_mask': encoded.attention_mask,
            'max_new_tokens': 256,
            'return_dict_in_generate': True,
            'pad_token_id': self.tokenizer.eos_token_id,
            'output_scores': True,
        }
        
        # Handle greedy vs sampling decoding
        if temperature == 0.0:
            # Greedy decoding
            generate_kwargs['do_sample'] = False
        else:
            # Sampling
            generate_kwargs['do_sample'] = True
            generate_kwargs['top_p'] = top_p
            generate_kwargs['temperature'] = temperature
        
        # Add use_cache for non-Gemma models
        if not 'gemma' in self.model_id.lower():
            generate_kwargs['use_cache'] = True
        
        # Run generation in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        
        # Time the actual model generation
        model_inference_start = time.time()
        outputs = await loop.run_in_executor(
            None,
            lambda: self.model.generate(**generate_kwargs)
        )
        model_inference_time = time.time() - model_inference_start
        
        # Process outputs
        completions = []
    
        # Calculate output statistics
        output_length = outputs.sequences.size(1) - encoded.input_ids.size(1)
        
        # Decode each sequence
        for i, generated_ids in enumerate(outputs.sequences):
            # Skip the input tokens
            seq_decode_start = time.time()
            new_tokens = generated_ids[encoded.input_ids.size(1):]
            completion = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            completions.append({"choices": [{"message": {"role": "assistant", "content": completion}}]})

        total_time = time.time() - start_time
        return completions

    async def generate(
        self,
        context: List[Dict[str, str]],
        device: torch.device,
        top_p: float = 0.9,
        temperature: float = 1.0,
        other_responses: Optional[List[str]] = None,
        diversity_prompt: bool = False
    ) -> str:
        """
        Generate a response for a single context - wrapper around batch_generate.
        
        Args:
            context: Conversation context as a list of message dicts
            device: Device to run generation on
            top_p: Nucleus sampling probability threshold
            temperature: Sampling temperature
            other_responses: Optional list of other agents' responses
            diversity_prompt: Whether to include a diversity prompt
            
        Returns:
            Generated response string
        """
        # If we have other responses and summarize is enabled, create a summary
        if other_responses:
            summary = "Here are solutions from other agents:\n"
            for i, resp in enumerate(other_responses):
                summary += f"\nAgent {i}'s response: {resp}"
            if diversity_prompt:
                summary += "\nYour solution should arrive at the correct answer using a different method compared to other agents."
            context.append({"role": "user", "content": summary})
        
        completions = await self.batch_generate([context], device, top_p, temperature)
        return completions[0]["choices"][0]["message"]["content"]

    def _has_valid_sft_training_checkpoint(
        self,
        experiment_dir: str,
        agent_id: int
    ) -> bool:
        """
        Check if valid SFT training state checkpoints exist for resuming training.
        
        Args:
            experiment_dir: The experiment directory path
            agent_id: The agent ID
            
        Returns:
            bool: True if valid SFT training checkpoints exist, False otherwise
        """
        sft_checkpoint_dir = os.path.join(experiment_dir, "logs", "sft", str(agent_id))
        
        if not os.path.exists(sft_checkpoint_dir):
            return False
            
        # Find the latest checkpoint directory
        checkpoint_dirs = [d for d in os.listdir(sft_checkpoint_dir) if d.startswith('checkpoint-')]
        if not checkpoint_dirs:
            return False
            
        # Sort by checkpoint number and get the latest
        latest_checkpoint = sorted(checkpoint_dirs, key=lambda x: int(x.split('-')[1]))[-1]
        checkpoint_path = os.path.join(sft_checkpoint_dir, latest_checkpoint)
        
        # Check for essential SFT training files
        required_files = [
            "trainer_state.json",  # Training state
            "adapter_model.safetensors",  # Adapter weights
            "adapter_config.json",  # Adapter configuration
            "optimizer.pt",  # Optimizer state
            "scheduler.pt"  # Learning rate scheduler state
        ]
        
        # All required files must exist
        return all(os.path.exists(os.path.join(checkpoint_path, f)) for f in required_files)

    def finetune(
        self,
        train_data: List[Dict[str, Any]],
        config: Dict[str, Any],
        round: int,
        checkpoint_path: str
    ) -> None:
        """
        Fine-tune the model using supervised learning on provided data.
        
        Args:
            train_data: Training data for fine-tuning
            config: Configuration dictionary with training parameters
            round: Current training round number
            checkpoint_path: Path to save fine-tuned model
        """
        sft_config = SFTConfig(
            max_seq_length=self.get_safe_max_length(),
            num_train_epochs=config['epoch_sft'],
            learning_rate=config['lr_sft'],
            weight_decay=config['weight_decay_sft'],
            per_device_train_batch_size=config['batch_size_sft'],
            per_device_eval_batch_size=config['batch_size_sft'],
            gradient_accumulation_steps=2,
            optim='adamw_torch',
            logging_strategy="steps",
            logging_steps=10,
            warmup_ratio=0.1, 
            lr_scheduler_type='cosine',
            max_grad_norm=1.0,
            output_dir=f"{config['experiment_dir']}/logs/sft/{self.id}",
            logging_dir=f"{config['experiment_dir']}/logs/sft/{self.id}",
            report_to=None,
            run_name=f'SFT_Agent{self.id}_Round{round}',
            disable_tqdm=False,
            seed=self.seed,
            data_seed=self.seed,
            packing=False,
            torch_empty_cache_steps=10,
            bf16=True,
            save_strategy="epoch",
        )
        from datasets import Dataset

        data = Dataset.from_list([{"text": self.tokenizer.apply_chat_template(
                ctx['messages'], 
                tokenize=False, 
                add_generation_prompt=True
            )} for ctx in train_data])
        
        def tokenize(sample):
            model_inps =  self.tokenizer(sample["text"], padding=True, truncation=True, max_length=self.get_safe_max_length())
            return model_inps
        
        tokenized_data = data.map(tokenize, batched=True, desc="Tokenizing data", remove_columns=data.column_names)
        
        # Configure LoRA parameters for adapter mode
        peft_config = None
        if self.adapter_mode:
            print(f"Using LoRA configuration for adapter-based fine-tuning")
            peft_config = LoraConfig(
                r=config.get('lora_r', 8),
                lora_alpha=config.get('lora_alpha', 32),
                target_modules=config.get('lora_target_modules', ["q_proj", "k_proj", "v_proj", "o_proj"]),
                task_type="CAUSAL_LM",
                lora_dropout=config.get('lora_dropout', 0.1),
            )

        trainer = SFTTrainer(
            model=self.model,
            train_dataset=tokenized_data,
            args=sft_config,
            peft_config=peft_config if self.adapter_mode else None,
        )

        trainer.add_callback(WandbLoggingCallback(agent_id=self.id, round=round, wandb_run=self.wandb_run, phase='SFT', log_interval=sft_config.logging_steps))
        
        # Check if we should resume from checkpoint
        sft_checkpoint_exists = self._has_valid_sft_training_checkpoint(config['experiment_dir'], self.id)
        if sft_checkpoint_exists:
            print(f"Resuming SFT training from existing checkpoint in {sft_config.output_dir}")
            trainer.train(resume_from_checkpoint=True)
        else:
            print(f"Starting fresh SFT training")
            trainer.train()
            
        trainer.save_model(checkpoint_path)
        print(f"SFT Checkpoint is saved to {checkpoint_path}")

    
    def compute_log_probs(
        self,
        prompts: List[str],
        completions: List[str]
    ):
        device = self.get_device()

        # 1. Tokenise prompt+completion together so we score P(comp | prompt)
        joint_texts   = [p + c for p, c in zip(prompts, completions)]
        enc = self.tokenizer.batch_encode_plus(
            joint_texts,
            padding=True,
            truncation=True,
            max_length=self.get_safe_max_length(),
            return_tensors='pt',
            return_attention_mask=True,
            padding_side="left"
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self.model(**enc).logits        # [B, L, V]
            logp_all = torch.log_softmax(logits, -1)

            # Gather log‑probs for each next token (includes final token)
            next_ids = enc["input_ids"][:, 1:].unsqueeze(-1)           # [B, L‑1, 1]
            tok_logp = logp_all[:, :-1, :].gather(-1, next_ids).squeeze(-1)  # [B, L‑1]

        # 2. Mask out prompt tokens (properly handling left padding)
        prompt_lens = torch.tensor(
            [len(self.tokenizer(p, add_special_tokens=False)["input_ids"])
            for p in prompts],
            device=device
        )
        
        # Use attention mask's cumulative sum to get true token positions
        position_ids = enc["attention_mask"].cumsum(dim=1) - 1  # [B, L]
        pos_ids_sliced = position_ids[:, 1:]                   # [B, L‑1]
        comp_mask = pos_ids_sliced >= prompt_lens.unsqueeze(1)  # [B, L‑1]

        # 3. Build ragged list of completion log‑probs and token IDs
        result = []
        for i in range(len(completions)):
            mask = comp_mask[i]
            log_probs = tok_logp[i, mask]
            token_ids = enc["input_ids"][i, 1:][mask]  # Get completion token IDs
            result.append((log_probs, token_ids))
            
        # Verify log_probs and token_ids have matching lengths
        for i, (log_probs, token_ids) in enumerate(result):
            if len(log_probs) != len(token_ids):
                print(f"Warning: Mismatch in lengths for item {i}")
                print(f"log_probs length: {len(log_probs)}, token_ids length: {len(token_ids)}")
                raise ValueError(f"Length mismatch: log_probs ({len(log_probs)}) != token_ids ({len(token_ids)})")
        return result

    def _compute_confidence_score(
        self,
        response: str,
        log_probs_and_ids: Tuple[torch.Tensor, torch.Tensor]
    ) -> float:
        """
        Compute confidence score from log probabilities.
        
        Args:
            response: The generated response
            log_probs_and_ids: Tuple of (log probabilities, token ids)
            
        Returns:
            Mean log probability of all tokens (more negative = less confident)
        """
        try:
            log_probs, _ = log_probs_and_ids
            if log_probs is None or len(log_probs) == 0:
                return -1.0  # Default to low confidence
        
            if torch.isnan(log_probs).any() or torch.isinf(log_probs).any():
                return -1.0
        
            # Take mean of log probabilities directly
            mean_log_prob = log_probs.mean().item()
            return mean_log_prob
                
        except Exception as e:
            print(f"Error computing confidence score: {e}")
            return -1.0

    def _has_valid_rl_training_checkpoint(
        self,
        experiment_dir: str,
        agent_id: int
    ) -> bool:
        """
        Check if valid RL training state checkpoints exist for resuming training.
        
        Args:
            experiment_dir: The experiment directory path
            agent_id: The agent ID
            
        Returns:
            bool: True if valid RL training checkpoints exist, False otherwise
        """
        rl_checkpoint_dir = os.path.join(experiment_dir, "logs", "rl", str(agent_id))
        
        if not os.path.exists(rl_checkpoint_dir):
            return False
            
        # Find the latest checkpoint directory
        checkpoint_dirs = [d for d in os.listdir(rl_checkpoint_dir) if d.startswith('checkpoint-')]
        if not checkpoint_dirs:
            return False
            
        # Sort by checkpoint number and get the latest
        latest_checkpoint = sorted(checkpoint_dirs, key=lambda x: int(x.split('-')[1]))[-1]
        checkpoint_path = os.path.join(rl_checkpoint_dir, latest_checkpoint)
        
        # Check for essential RL training files
        required_files = [
            "trainer_state.json",  # Training state
            "adapter_model.safetensors",  # Adapter weights
            "adapter_config.json",  # Adapter configuration
            "optimizer.pt",  # Optimizer state
            "scheduler.pt"  # Learning rate scheduler state
        ]
        
        # All required files must exist
        return all(os.path.exists(os.path.join(checkpoint_path, f)) for f in required_files)

    def post_train(
        self,
        train_data: Any,
        config: Dict[str, Any],
        round: int,
        checkpoint_path: str,
        precomputed_rewards: Optional[List[float]] = None
    ) -> None:
        """
        RL post-training phase to optimize for consensus and diversity.
        
        Args:
            train_data: Training data for RL
            config: Configuration dictionary with training parameters
            round: Current training round number
            checkpoint_path: Path to save RL-trained model
            precomputed_rewards: List of precomputed rewards to use
        """
        print(f"Train data size: {len(train_data)}")
        
        # Format reward function
        def format_reward_func(completions: List[str], **kwargs) -> List[float]:
            """Reward proper formatting with bullet points and boxed answers."""
            scores = []
            for completion in completions:
                score = 0.0
                
                # Split by newlines and filter out empty lines
                lines = [line.strip() for line in completion.split("\n") if line.strip()]
                
                # Check boxed answer positioning and count
                boxed_count = completion.count("\\boxed{")
                boxed_in_last_line = lines and "\\boxed{" in lines[-1]
                
                # Count thought steps
                thought_steps = sum(1 for line in lines if "\\boxed{" not in line and len(line) > 10)
                
                # Reward having multiple distinct thought steps
                if thought_steps >= 2:
                    score += 0.2
                    # Extra reward for more thorough reasoning (up to 0.6 total)
                    score += min(0.2, (thought_steps - 2) * 0.04)
                
                # Reward for having exactly one boxed answer
                if boxed_count == 1:
                    score += 0.3
                    # Additional reward if it's in the last line
                    if boxed_in_last_line:
                        score += 0.2
                    else:
                        # Penalize if not in last line
                        score -= 0.1
                else:
                    # Penalize multiple or missing boxed answers
                    score -= 0.2
                
                scores.append(max(0.0, min(1.0, score)))  # Clamp between 0 and 1
            
            # Log metrics if W&B is configured
            if hasattr(self, 'wandb_run') and self.wandb_run:
                self.wandb_run.log({
                    f"Agent_{self.id}/RL_{round}/format_reward": torch.tensor(scores).mean().item(),
                }, step=self.trainer.state.global_step if hasattr(self, 'trainer') else None)
            return scores

        def correctness_reward_func(completions: List[str], **kwargs) -> List[float]:
            """Reward correct answers with +2.0, incorrect with 0.0 or -1.0."""
            # Apply verifiable_reward setting to determine incorrect answer score
            if config.get('use_majority_vote', True):
                labels = kwargs["majority_answer"]
            else:
                labels = kwargs["ground_truth"]
                
            incorrect_score = 0.0 if config.get('verifiable_reward', False) else -1.0
            rewards = [2.0 if grade_answer(parse_answer(response, self.task), answer) else incorrect_score 
                   for (response, answer) in zip(completions, labels)]
                   
            # Log metrics if W&B is configured
            if hasattr(self, 'wandb_run') and self.wandb_run:
                self.wandb_run.log({
                    f"Agent_{self.id}/RL_{round}/correctness_reward": torch.tensor(rewards).mean().item(),
                }, step=self.trainer.state.global_step if hasattr(self, 'trainer') else None)
            return rewards
        

        def consensus_reward_func(completions: List[str], **kwargs) -> List[float]:
            """Reward function incorporating consensus agreement and psychometric confidence."""
            # Get configuration parameters with defaults
            consensus_batch_size = config.get('consensus_batch_size', 2)  # Conservative default
            
            # Get answers and prompts
            if config.get('use_majority_vote', True):
                labels = kwargs["majority_answer"]
            else:
                labels = kwargs["ground_truth"]
            
            prompts = kwargs.get("prompts", [])
            if not prompts:
                print("Warning: No prompts provided for consensus reward")
                return [0.0] * len(completions)
            
            # Process in smaller batches to prevent OOM during log probability computation
            rewards = []
            confidences = []  # Track confidence scores for logging
            total_examples = len(completions)
            
            for batch_start in range(0, total_examples, consensus_batch_size):
                batch_end = min(batch_start + consensus_batch_size, total_examples)
                
                # Extract batch
                batch_prompts = prompts[batch_start:batch_end]
                batch_completions = completions[batch_start:batch_end]
                batch_labels = labels[batch_start:batch_end]
                
                # print(f"   Processing consensus batch {batch_start//consensus_batch_size + 1}/{(total_examples + consensus_batch_size - 1)//consensus_batch_size} "
                #       f"(examples {batch_start}-{batch_end-1})")
                
                # Compute log probabilities for this batch only
                batch_log_probs = self.compute_log_probs(batch_prompts, batch_completions)
                
                # Process each example in the batch
                for response, answer, completion_log_probs in zip(batch_completions, batch_labels, batch_log_probs):
                    # Check if answer matches
                    matches = grade_answer(parse_answer(response, self.task), answer)
                    consensus = 1.0 if matches else -1.0
                    
                    # Compute confidence score
                    confidence = self._compute_confidence_score(response, completion_log_probs)
                    confidences.append(confidence)
                    
                    # Apply step transfer function: reward = 1.0 + consensus
                    reward = 1.0 + float(consensus)
                    rewards.append(reward)
            
            # Log metrics if W&B is configured
            if hasattr(self, 'wandb_run') and self.wandb_run:
                self.wandb_run.log({
                    f"Agent_{self.id}/RL_{round}/consensus_reward": torch.tensor(rewards).mean().item(),
                    f"Agent_{self.id}/RL_{round}/confidence": torch.tensor(confidences).mean().item(),
                }, step=self.trainer.state.global_step if hasattr(self, 'trainer') else None)
            
            pd(f"   Consensus reward computation completed: {len(rewards)} examples processed")
            return rewards

        # Select reward functions based on configuration
        reward_funcs = []
        
        # Always add consensus reward
        if config.get('use_consensus_reward', False):
            reward_funcs.append(consensus_reward_func)
        else:
            reward_funcs.append(correctness_reward_func)
        
        # Add format rewards if enabled
        if config.get('use_format_reward', False):
            reward_funcs.append(format_reward_func)
        
        
        # Configure LoRA parameters for adapter mode
        peft_config = None
        if self.adapter_mode or config['peft']:
            peft_config = LoraConfig(
                r=config.get('lora_r', 8),
                lora_alpha=config.get('lora_alpha', 32),
                target_modules=config.get('lora_target_modules', ["q_proj", "k_proj", "v_proj", "o_proj"]),
                task_type="CAUSAL_LM",
                lora_dropout=config.get('lora_dropout', 0.1),
            )
            pd(f"Using LoRA configuration for {'adapter-based' if self.adapter_mode else 'PEFT'} training")
        
        # When using quantization with specific device mapping, ensure single-device training
        additional_grpo_kwargs = {}
        
        # Check if the model is quantized - use stored state or model's built-in check
        is_quantized = self.quantization or (hasattr(self.model, "is_quantized") and self.model.is_quantized)
        
        # If we have quantization, completely disable all parallel training methods
        if is_quantized:
            # Extract device information for single-device configuration
            if isinstance(self.device_map, str) and self.device_map.startswith('cuda:'):
                device_id = int(self.device_map.split(':')[1])
            else:
                device_id = 0  # Default to first GPU
                
            # Only use valid GRPOConfig parameters for QLoRA
            additional_grpo_kwargs.update({
                # Valid training configuration parameters
                'dataloader_num_workers': 0,  # Reduce workers for stability
                'dataloader_pin_memory': False,  # Reduce memory pressure
                'gradient_checkpointing': True,  # Memory optimization
                'optim': "paged_adamw_8bit",  # QLoRA-optimized optimizer
                'fp16': False,  # Don't use fp16 with 4-bit quantization
                'fsdp': '',  # Disable FSDP for QLoRA (empty string, not None)
            })
            pd(f"Agent {self.id} - Using single-device GRPO config for quantized training on {self.device_map}")
        elif isinstance(self.device_map, dict):
            # Legacy dict format support for non-quantized models
            target_device = list(self.device_map.values())[0] if self.device_map else 'cuda:0'
            if isinstance(target_device, str) and target_device.startswith('cuda:'):
                device_id = int(target_device.split(':')[1])
                additional_grpo_kwargs.update({
                    # Only basic configuration for non-quantized
                })
                pd(f"Agent {self.id} - Using single-device config for non-quantized training")
        
        grpo_config = GRPOConfig(
            report_to=None,
            logging_strategy="steps",
            logging_steps=1,
            learning_rate=config['lr_rl'],
            weight_decay=config['weight_decay_rl'],
            num_train_epochs=config['epoch_rl'],
            per_device_train_batch_size=config['batch_rl'],
            gradient_accumulation_steps=config.get('gradient_accumulation_steps', 1),
            warmup_ratio=0.1,
            lr_scheduler_type='cosine',
            max_grad_norm=0.1,
            max_completion_length=256,
            max_prompt_length=1024,
            bf16=True,
            # Generation sampling parameters – pulled from global CLI args so
            # RL rollouts inherit the same behaviour as debate phase.
            temperature=config.get('temperature', 1.0),
            top_p=config.get('top_p', 0.9),
            num_generations=config['num_generations'],
            seed=self.seed,
            data_seed=self.seed,
            # use_vllm=True, #config.get('use_vllm', False),
            # vllm_mode="colocate",
            run_name=f'RL_Agent{self.id}_Round{round}',
            output_dir=f"{config['experiment_dir']}/logs/rl/{self.id}",
            logging_dir=f"{config['experiment_dir']}/logs/rl/{self.id}",
            loss_type="dr_grpo",
            beta=0.0, #kl_divergence
            # Proper way to disable distributed training for quantized models
            **additional_grpo_kwargs
        )

        # Load SFT adapter if needed BEFORE creating trainer
        rl_checkpoint_exists = self._has_valid_rl_training_checkpoint(config['experiment_dir'], self.id)
        sft_checkpoint_exists = os.path.exists(checkpoint_path)
        
        sft_adapter_loaded_for_rl = False # Flag to track if SFT adapter was loaded for RL
        if not rl_checkpoint_exists and sft_checkpoint_exists:
            # Starting RL training after SFT - need to load SFT adapter before creating trainer
            if self.adapter_mode and self.base_model_key:
                # Load SFT adapter using BaseModelManager
                BaseModelManager.set_active_adapter(self.base_model_key, checkpoint_path)
                # Update self.model to point to the model with loaded adapter
                self.model = BaseModelManager._base_models[self.base_model_key]["model"]
                print(f"Loaded SFT adapter from {checkpoint_path} for RL training")
                sft_adapter_loaded_for_rl = True
            elif not self.adapter_mode:
                print(f"Using SFT model loaded from {checkpoint_path}")

        trainer = GRPOTrainer(
            model=self.model,
            args=grpo_config,
            processing_class=self.tokenizer,
            train_dataset=train_data,
            reward_funcs=reward_funcs,
            peft_config=peft_config if not (self.adapter_mode and sft_adapter_loaded_for_rl) else None,
        )

        # Expose trainer early so reward functions (invoked during training) can access current step
        self.trainer = trainer

        trainer.add_callback(WandbLoggingCallback(agent_id=self.id, round=round, wandb_run=self.wandb_run, phase='RL', log_interval=grpo_config.logging_steps))

        # Determine checkpoint resume strategy (simplified since adapter loading is done above)
        if rl_checkpoint_exists:
            # Resuming interrupted RL training - RL checkpoint contains everything needed
            print(f"Resuming RL training from existing RL checkpoint in {grpo_config.output_dir}")
            trainer.train(resume_from_checkpoint=True)
        elif sft_checkpoint_exists:
            # Starting RL training after SFT - adapter already loaded above
            print(f"Starting RL training with SFT adapter from {checkpoint_path}")
            trainer.train(resume_from_checkpoint=False)
        else:
            # Fresh RL training - no prior SFT or RL checkpoints
            print(f"Starting fresh RL training")
            trainer.train(resume_from_checkpoint=False)

        trainer.save_model(checkpoint_path)
        print(f"RL Checkpoint is saved to {checkpoint_path}")

    # =============================================================
    #   KTO Preference Fine-tuning (class-level)
    # =============================================================
    def kto(
        self,
        train_data: Any,
        config: Dict[str, Any],
        round: int,
        checkpoint_path: str
    ) -> None:
        """Run Kahneman-Tversky Optimization preference tuning."""

        print(f"[Agent {self.id}] KTO | dataset size = {len(train_data)}")

        # Optional PEFT / LoRA
        peft_cfg = None
        if self.adapter_mode or config.get('peft', False):
            peft_cfg = LoraConfig(
                r=config.get('lora_r', 8),
                lora_alpha=config.get('lora_alpha', 16),
                target_modules=config.get('lora_target_modules', ["q_proj", "k_proj", "v_proj", "o_proj"]),
                lora_dropout=config.get('lora_dropout', 0.1),
                task_type="CAUSAL_LM",
            )

        kto_cfg = KTOConfig(
            report_to=None,
            logging_strategy="steps",
            logging_steps=1,
            per_device_train_batch_size=config.get('batch_kto', 8),
            num_train_epochs=config.get('epoch_kto', 3),
            gradient_accumulation_steps=config.get('gradient_accumulation_steps_kto', 1),
            learning_rate=config.get('lr_kto', 1e-6),
            beta=config.get('beta_kto', 0.1),
            max_prompt_length=1024,
            max_completion_length=256,
            bf16=True,
            seed=self.seed,
            data_seed=self.seed,
            run_name=f"KTO_Agent{self.id}_Round{round}",
            output_dir=f"{config['experiment_dir']}/logs/kto/{self.id}",
            logging_dir=f"{config['experiment_dir']}/logs/kto/{self.id}",
            desirable_weight=config.get('desirable_weight', 1.0),
            undesirable_weight=config.get('undesirable_weight', 1.0),
        )

        if self.quantization:
            kto_cfg.gradient_checkpointing = True
            kto_cfg.dataloader_num_workers = 0
            kto_cfg.dataloader_pin_memory = False
            kto_cfg.fp16 = False
            kto_cfg.fsdp = ''

        # -------------------------------------------------------------
        # Adapter handling (mirror RL logic)
        # -------------------------------------------------------------
        adapter_already_loaded = False
        if self.adapter_mode and os.path.exists(checkpoint_path):
            # Ensure the adapter from previous KTO run (if any) is active so
            # the model has trainable params before we build the trainer.
            if self.base_model_key:
                BaseModelManager.set_active_adapter(self.base_model_key, checkpoint_path)
                self.model = BaseModelManager._base_models[self.base_model_key]["model"]
                adapter_already_loaded = True

        # Decide whether we are resuming
        kto_checkpoint_exists = self._has_valid_kto_training_checkpoint(config['experiment_dir'], self.id)
        resume_flag = bool(kto_checkpoint_exists)
        if resume_flag:
            print(f"[Agent {self.id}] Resuming KTO training from existing checkpoint in {kto_cfg.output_dir}")

        # If adapter already on model, do NOT pass a new peft_config
        peft_cfg_for_trainer = None if adapter_already_loaded else peft_cfg

        trainer = KTOTrainer(
            model=self.model,
            args=kto_cfg,
            processing_class=self.tokenizer,
            train_dataset=train_data,
            peft_config=peft_cfg_for_trainer,
        )

        self.trainer = trainer
        trainer.add_callback(WandbLoggingCallback(agent_id=self.id, round=round, wandb_run=self.wandb_run, phase='KTO', log_interval=kto_cfg.logging_steps))
        if self.wandb_run:
            from utils import KTORewardLoggingCallback
            trainer.add_callback(KTORewardLoggingCallback(agent_id=self.id, round=round, wandb_run=self.wandb_run))
        trainer.train(resume_from_checkpoint=resume_flag)
        trainer.save_model(checkpoint_path)
        print(f"[Agent {self.id}] KTO | checkpoint saved to {checkpoint_path}")

    def _has_valid_kto_training_checkpoint(
        self,
        experiment_dir: str,
        agent_id: int
    ) -> bool:
        """
        Check if valid KTO training state checkpoints exist for resuming or skipping.

        Args:
            experiment_dir: Path to the experiment directory
            agent_id: Numerical id of the agent

        Returns:
            bool: True if a valid checkpoint dir with the essential files exists
        """
        kto_checkpoint_dir = os.path.join(experiment_dir, "logs", "kto", str(agent_id))
        if not os.path.isdir(kto_checkpoint_dir):
            return False

        # look for checkpoint-<step> sub-dirs
        ckpt_dirs = [d for d in os.listdir(kto_checkpoint_dir) if d.startswith("checkpoint-")]
        if not ckpt_dirs:
            return False

        latest_ckpt = sorted(ckpt_dirs, key=lambda d: int(d.split("-")[1]))[-1]
        ckpt_path = os.path.join(kto_checkpoint_dir, latest_ckpt)
        required_files = [
            "trainer_state.json",
            "adapter_model.safetensors",
            "adapter_config.json",
            "optimizer.pt",
            "scheduler.pt",
        ]
        return all(os.path.exists(os.path.join(ckpt_path, f)) for f in required_files)

    # =============================================================
    #   DPO Preference Fine-tuning (paired preferences)
    # =============================================================

    def _has_valid_dpo_training_checkpoint(
        self,
        experiment_dir: str,
        agent_id: int
    ) -> bool:
        """Return True if a resumable DPO checkpoint exists for this agent."""
        dpo_ckpt_dir = os.path.join(experiment_dir, "logs", "dpo", str(agent_id))
        if not os.path.isdir(dpo_ckpt_dir):
            return False

        ckpt_dirs = [d for d in os.listdir(dpo_ckpt_dir) if d.startswith("checkpoint-")]
        if not ckpt_dirs:
            return False

        latest = sorted(ckpt_dirs, key=lambda d: int(d.split("-")[1]))[-1]
        ckpt_path = os.path.join(dpo_ckpt_dir, latest)
        required = [
            "trainer_state.json",
            "adapter_model.safetensors",
            "adapter_config.json",
            "optimizer.pt",
            "scheduler.pt",
        ]
        return all(os.path.exists(os.path.join(ckpt_path, f)) for f in required)

    def dpo(
        self,
        train_data: Any,
        config: Dict[str, Any],
        round: int,
        checkpoint_path: str
    ) -> None:
        """Direct Preference Optimisation fine-tuning (paired data)."""

        print(f"[Agent {self.id}] DPO | dataset size = {len(train_data)} pairs")

        # -------------------- PEFT / LoRA ------------------------
        peft_cfg = None
        if self.adapter_mode or config.get('peft', False):
            peft_cfg = LoraConfig(
                r=config.get('lora_r', 8),
                lora_alpha=config.get('lora_alpha', 16),
                target_modules=config.get('lora_target_modules', ["q_proj", "k_proj", "v_proj", "o_proj"]),
                lora_dropout=config.get('lora_dropout', 0.1),
                task_type="CAUSAL_LM",
            )

        dpo_cfg = DPOConfig(
            report_to=None,
            logging_strategy="steps",
            logging_steps=1,
            per_device_train_batch_size=config.get('batch_dpo', 8),
            num_train_epochs=config.get('epoch_dpo', 3),
            gradient_accumulation_steps=config.get('gradient_accumulation_steps_dpo', 1),
            learning_rate=config.get('lr_dpo', 5e-6),
            beta=config.get('beta_dpo', 0.1),
            max_prompt_length=1024,
            max_completion_length=256,
            bf16=True,
            seed=self.seed,
            data_seed=self.seed,
            run_name=f"DPO_Agent{self.id}_Round{round}",
            output_dir=f"{config['experiment_dir']}/logs/dpo/{self.id}",
            logging_dir=f"{config['experiment_dir']}/logs/dpo/{self.id}",
        )

        # Quantisation adjustments identical to KTO logic
        if self.quantization:
            dpo_cfg.gradient_checkpointing = True
            dpo_cfg.dataloader_num_workers = 0
            dpo_cfg.dataloader_pin_memory = False
            dpo_cfg.fp16 = False
            dpo_cfg.fsdp = ''

        # ---------------------------------------------------------
        # Adapter handling (resume or fresh)
        # ---------------------------------------------------------
        adapter_loaded = False
        if self.adapter_mode and os.path.exists(checkpoint_path):
            # Ensure previous adapter active
            if self.base_model_key:
                BaseModelManager.set_active_adapter(self.base_model_key, checkpoint_path)
                self.model = BaseModelManager._base_models[self.base_model_key]["model"]
                adapter_loaded = True

        resume_flag = self._has_valid_dpo_training_checkpoint(config['experiment_dir'], self.id)
        if resume_flag:
            print(f"[Agent {self.id}] Resuming DPO from checkpoint in {dpo_cfg.output_dir}")

        # Decide peft_config for trainer
        peft_cfg_trainer = None if adapter_loaded else peft_cfg

        trainer = DPOTrainer(
            model=self.model,
            ref_model=None,  # let trainer build reference model automatically
            args=dpo_cfg,
            train_dataset=train_data,
            processing_class=self.tokenizer,
            peft_config=peft_cfg_trainer,
        )

        # For WandB logging reuse existing callback infrastructure
        trainer.add_callback(WandbLoggingCallback(agent_id=self.id, round=round, wandb_run=self.wandb_run, phase='DPO', log_interval=dpo_cfg.logging_steps))
        if self.wandb_run:
            from utils import DPORewardLoggingCallback
            trainer.add_callback(DPORewardLoggingCallback(agent_id=self.id, round=round, wandb_run=self.wandb_run))

        self.trainer = trainer

        trainer.train(resume_from_checkpoint=resume_flag)
        trainer.save_model(checkpoint_path)
        print(f"[Agent {self.id}] DPO | checkpoint saved to {checkpoint_path}")

