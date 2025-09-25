import os
import asyncio
import torch
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from model import BaseModelManager, Agent
from utils import pd
logger = logging.getLogger(__name__)

class GPUResourceTracker:
    """Manages the allocation and release of GPU resources."""
    def __init__(
        self,
        all_gpu_ids: List[int]
    ):
        self.available_gpus = set(all_gpu_ids)
        self.lock = asyncio.Lock()

    async def claim_gpus(
        self,
        num_required: int
    ) -> Optional[List[int]]:
        """Claims a specified number of GPUs from the available pool."""
        async with self.lock:
            if len(self.available_gpus) >= num_required:
                # Claim the GPUs with the lowest IDs first for consistency
                claimed = sorted(list(self.available_gpus))[:num_required]
                self.available_gpus -= set(claimed)
                pd(f"GPUResourceTracker: Claimed GPUs {claimed}. Available: {sorted(list(self.available_gpus))}")
                return claimed
            return None

    async def release_gpus(
        self,
        gpu_ids: List[int]
    ):
        """Releases a set of GPUs back to the available pool."""
        async with self.lock:
            self.available_gpus.update(gpu_ids)
            pd(f"GPUResourceTracker: Released GPUs {gpu_ids}. Available: {sorted(list(self.available_gpus))}")

class AdapterJobScheduler:
    """
    Scheduler for managing adapter-swapping jobs across multiple GPUs.
    
    This class maintains a queue of generation tasks and efficiently routes them
    to appropriate GPU resources based on adapter availability and device capacity.
    """
    
    def __init__(
        self,
        config: Dict[str, Any]
    ):
        """
        Initialize the scheduler with configuration parameters.
        
        Args:
            config: Dictionary containing configuration parameters including:
                - model: base model name
                - devices: list of available devices
                - max_agents_per_device: maximum number of models per GPU
                - experiment_dir: directory for checkpoints
                - use_quantization: whether to use quantization
                - use_adapter_mode: whether to use adapter swapping
        """
        self.config = config
        self.task_queue = asyncio.PriorityQueue()
        self.adapter_cache = {}  # Maps adapter_path -> model_key
        self.tokenizer_cache = {}  # Maps model_name -> tokenizer
        self.running = True
        
        # New: GPU resource management
        self.gpus_per_model = config.get('gpus_per_model', 1)
        if self.config.get('use_quantization', False):
            self.gpu_tracker = GPUResourceTracker([int(d.split(':')[-1]) for d in self.config['devices']])
        
        # To control the total number of concurrently running tasks
        self.max_concurrent_tasks = min(
            config.get('max_concurrent_tasks', len(config['devices']) * 2),
            len(config['devices']) // self.gpus_per_model if self.gpus_per_model > 0 else 1
        )
        
        self.model_id_map = {
            "mistral7b": "mistralai/Mistral-7B-Instruct-v0.3",
            "llama1b": "meta-llama/Llama-3.2-1B-Instruct",
            "llama3b": "meta-llama/Llama-3.2-3B-Instruct", 
            "llama8b": "meta-llama/Llama-3.1-8B-Instruct",
            "phi4b": "microsoft/Phi-3-mini-128k-instruct",
            "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
            "qwen2b": "Qwen/Qwen2.5-1.5B-Instruct",
            "gemma4b": "google/gemma-3-4b-it"
        }
        self.model_id = self.model_id_map.get(config['model'])
        
        # Initialize workers
        self.workers = []
        
    async def _preload_base_models(self):
        """Pre-load base models, potentially across multiple GPUs."""
        if not self.config.get('use_quantization', False):
            return

        pd(f"Pre-loading base models on {len(self.config['devices'])} GPUs, with {self.gpus_per_model} GPUs per model.")
        
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        instance_id = 0
        original_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)

        while True:
            # Claim a chunk of GPUs for the next model instance
            gpus_to_use = await self.gpu_tracker.claim_gpus(self.gpus_per_model)
            if gpus_to_use is None:
                pd("No more available GPUs to load models. Pre-loading complete.")
                break

            pd(f"Attempting to load model instance {instance_id} on GPUs: {gpus_to_use}")
            
            try:
                # Temporarily restrict visible devices for this loading operation
                os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(map(str, gpus_to_use))

                # For single-GPU, specify the device directly to avoid ambiguity. For multi-GPU, use 'auto'.
                if self.gpus_per_model == 1 and gpus_to_use:
                    # Single-GPU case – place the entire model on that GPU
                    device_map = f'cuda:{gpus_to_use[0]}'
                    pd(f"Manually mapping model instance {instance_id} to single device: {device_map}")
                else:
                    # Multi-GPU case – distribute layers as evenly as possible across
                    # the visible devices.  'balanced_low_0' is an Accelerate preset
                    # that assigns layers in a round-robin fashion starting from GPU 0
                    # and aims for a uniform memory footprint.
                    device_map = 'balanced_low_0'
                    pd(f"Using balanced device map for model instance {instance_id} across GPUs {gpus_to_use} with device_map='{device_map}'")

                # Create a unique key for this model instance based on its GPUs
                model_key = f"{self.model_id}_gpus_{'-'.join(map(str, gpus_to_use))}_instance_{instance_id}"
                
                model, _ = BaseModelManager.get_or_create_base_model(
                    self.model_id,
                    quantization_config=quantization_config,
                    torch_dtype=torch.float16,
                    device_map=device_map,
                    cache_dir='./checkpoints',
                    low_cpu_mem_usage=True,
                    # Pass specific metadata for storage
                    _model_key_override=model_key,
                    _gpu_ids=gpus_to_use
                )
                
                # The model is now loaded and tracked by BaseModelManager, no need to add to a separate pool
                pd(f"Successfully loaded model {model_key} on GPUs {gpus_to_use}.")
                instance_id += 1

            except Exception as e:
                print(f"Error pre-loading model instance on GPUs {gpus_to_use}: {e}")
                # Release the GPUs if loading failed
                await self.gpu_tracker.release_gpus(gpus_to_use)
                break # Stop trying to load more models if one fails
            finally:
                # Restore original environment
                if original_visible_devices is None:
                    del os.environ['CUDA_VISIBLE_DEVICES']
                else:
                    os.environ['CUDA_VISIBLE_DEVICES'] = original_visible_devices
        
        pd("\nBase Model Loading Summary:")
        for model_key, info in BaseModelManager._base_models.items():
            pd(f"Model {model_key}:")
            pd(f"  - Device Map: {info['device']}")
            pd(f"  - Assigned GPUs: {info.get('gpu_ids', 'N/A')}")
            pd(f"  - In use: {info['in_use']}")
            pd(f"  - Active adapter: {info['active_adapter']}")
        
    async def start(self):
        """Start the scheduler workers."""
        # Pre-load base models if using quantization
        await self._preload_base_models()
        
        num_workers = self.max_concurrent_tasks       
        self.workers = [asyncio.create_task(self._worker()) 
                        for _ in range(num_workers)]
        
    async def shutdown(
        self,
        timeout: float = 30.0
    ):
        """
        Shutdown the scheduler and release resources gracefully.
        
        Args:
            timeout: Maximum time to wait for tasks to complete
        """
        pd("Shutting down adapter job scheduler")
        
        # Stop accepting new tasks
        self.running = False
        
        # Wait for pending tasks to complete with timeout
        try:
            if not self.task_queue.empty():
                pd(f"Waiting up to {timeout}s for {self.task_queue.qsize()} pending tasks to complete...")
                await asyncio.wait_for(self.task_queue.join(), timeout=timeout)
        except asyncio.TimeoutError:
            print(f"Warning: Timeout waiting for tasks to complete. Forcing shutdown.")
        
        # Cancel all worker tasks
        for worker in self.workers:
            worker.cancel()
            
        try:
            # Wait briefly for workers to clean up
            await asyncio.wait_for(
                asyncio.gather(*self.workers, return_exceptions=True),
                timeout=5.0
            )
        except asyncio.TimeoutError:
            print("Warning: Workers did not shut down cleanly")
        except asyncio.CancelledError:
            pass
            
        # Clear caches
        pd("Clearing scheduler caches")
        self.adapter_cache = {}
        self.tokenizer_cache = {}
        
        # Release all base models
        model_keys = list(BaseModelManager._base_models.keys())
        for model_key in model_keys:
            # When shutting down, release the GPUs associated with the model
            model_info = BaseModelManager._base_models.get(model_key, {})
            if 'gpu_ids' in model_info:
                await self.gpu_tracker.release_gpus(model_info['gpu_ids'])
            BaseModelManager.delete_base_model(model_key)
        
        # Force cleanup of any remaining models
        for model_key in list(BaseModelManager._base_models.keys()):
            try:
                del BaseModelManager._base_models[model_key]
            except Exception as e:
                print(f"Warning: Error cleaning up model {model_key}: {e}")
        
        pd("Scheduler shutdown complete")
    
    async def _get_tokenizer(
        self,
        model_id
    ):
        """Get or create tokenizer for the given model ID."""
        if model_id not in self.tokenizer_cache:
            from transformers import AutoTokenizer
            pd(f"Loading tokenizer for {model_id}")
            tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir='./checkpoints', padding_side="left")
            tokenizer.pad_token = tokenizer.eos_token
            self.tokenizer_cache[model_id] = tokenizer
        return self.tokenizer_cache[model_id]
    
    async def schedule_generate(
        self,
        context: List[Dict[str, str]],
        agent_id: int,
        temperature: float = 1.0,
        top_p: float = 0.9
    ) -> Dict:
        """
        Schedule a generation task for a specific agent.
        
        Args:
            context: Conversation context as a list of message dicts
            agent_id: ID of the agent (maps to adapter path)
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            Completion dictionary with generated response
        """
        task = {
            "type": "generate",
            "context": context,
            "agent_id": agent_id,
            "temperature": temperature,
            "top_p": top_p,
            "future": asyncio.Future()
        }
        await self.task_queue.put(task)
        return await task["future"]
    
    async def schedule_batch_generate(
        self,
        contexts: List[List[Dict[str, str]]],
        agent_id: int,
        round_idx: int,
        temperature: float = 1.0,
        top_p: float = 0.9
    ) -> List[Dict]:
        """
        Schedule a batch generation task for a specific agent.
        
        Args:
            contexts: List of conversation contexts
            agent_id: ID of the agent (maps to adapter path)
            round_idx: The current debate round, used for prioritization.
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            List of completion dictionaries
        """
        task = {
            "type": "batch_generate",
            "contexts": contexts,
            "agent_id": agent_id,
            "temperature": temperature,
            "top_p": top_p,
            "future": asyncio.Future()
        }
        # Lower number = higher priority. 
        # We use round_idx so that Round 0 (priority 0) jobs are processed before Round 1 (priority 1).
        # To prioritize by depth, we would use a decreasing priority, e.g., -round_idx.
        # However, for this change, we'll keep it simple and prioritize by batch completion.
        # To ensure FIFO behavior for tasks with the same priority, we add a timestamp.
        priority = (-round_idx, time.time())
        await self.task_queue.put((priority, task))
        return await task["future"]
    
    async def _worker(self):
        """Worker coroutine that processes tasks from the queue."""
        from model import BaseModelManager
        
        while self.running:
            try:
                # Get next task with timeout. The item from PriorityQueue is (priority, task).
                priority, task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                
                model_key = None
                try:
                    # Acquire a model directly from the centralized manager
                    agent_id = task["agent_id"]
                    pd(f"Agent {agent_id} waiting for an available model...")
                    model_key = await BaseModelManager.acquire_model()
                    pd(f"Agent {agent_id} acquired model {model_key}.")

                    # Now that we have exclusive access to a model, process the task
                    result = await self._process_task(task, model_key)
                    task["future"].set_result(result)
                    
                except Exception as e:
                    print(f"Error processing task for agent {task.get('agent_id', 'N/A')}: {e}")
                    import traceback
                    traceback.print_exc()
                    if "future" in task:
                        task["future"].set_exception(e)
                finally:
                    # This block ensures resources are always released
                    if model_key:
                        # Release the model back to the centralized manager
                        await BaseModelManager.release_model(model_key)
                        pd(f"Agent {task.get('agent_id', 'N/A')} returned model {model_key}.")
                    
                    self.task_queue.task_done()
                    
            except asyncio.TimeoutError:
                # Normal when queue is empty
                continue
            except Exception as e:
                print(f"Worker loop error: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    async def _process_task(
        self,
        task,
        model_key: str
    ):
        """
        Process a task using a model that has already been locked for exclusive use.
        """
        from model import BaseModelManager
        
        agent_id = task["agent_id"]
        adapter_path = f"{self.config['experiment_dir']}/checkpoints/agent_{agent_id}"
        exists = os.path.exists(adapter_path)
        
        model_info = BaseModelManager._base_models[model_key]
        # Derive the primary device from the model parameters themselves. This is
        # safer than relying on the cached global GPU indices (which become
        # relative once CUDA_VISIBLE_DEVICES is changed during model loading).
        try:
            primary_device = next(model_info["model"].parameters()).device
        except StopIteration:
            # Fallback to stored value in the (unlikely) case the model has no parameters yet
            primary_device = model_info.get("device", "cpu")
        
        # Generate consistent adapter name
        adapter_name = f"adapter_agent_{agent_id}"
        
        # Load adapter if it exists. We already have the lock, so this is safe.
        if exists:
            pd(f"Loading adapter from {adapter_path} onto {model_key}")
            # This will only load the adapter if it's not already loaded
            BaseModelManager.add_adapter(model_key, adapter_path, adapter_name)
            # This will just switch to the already loaded adapter
            BaseModelManager.set_active_adapter(model_key, adapter_name)
        
        # Get tokenizer
        tokenizer = await self._get_tokenizer(self.model_id)
        
        try:
            # Execute task
            if task["type"] == "generate":
                return await self._execute_generate(
                    model_key, tokenizer, primary_device, 
                    task["context"], task["temperature"], task["top_p"]
                )
            elif task["type"] == "batch_generate":
                return await self._execute_batch_generate(
                    model_key, tokenizer, primary_device,
                    task["contexts"], task["temperature"], task["top_p"]
                )
        finally:
            # We don't remove the adapter anymore, just let it stay cached
            # The adapter will be cleaned up when the model is released
            pass
    
    async def _execute_generate(
        self,
        model_key: str,
        tokenizer,
        device: str,
        context: List[Dict[str, str]],
        temperature: float,
        top_p: float
    ) -> Dict:
        """
        Execute a single generation task.
        
        Args:
            model_key: Key identifying the model to use
            tokenizer: Tokenizer for encoding/decoding
            device: Device where model is located
            context: Conversation context
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            Completion dictionary
        """
        model = BaseModelManager._base_models[model_key]["model"]
        
        # Apply chat template
        input_text = tokenizer.apply_chat_template(
            context, tokenize=False, add_generation_prompt=True
        )
        
        # Tokenize
        encoded = tokenizer.encode_plus(
            input_text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_attention_mask=True
        ).to(device)
        
        # Generate in a separate thread to avoid blocking
        loop = asyncio.get_running_loop()
        outputs = await loop.run_in_executor(
            None,
            lambda: model.generate(
                input_ids=encoded.input_ids,
                attention_mask=encoded.attention_mask,
                max_new_tokens=256,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
        )
        
        # Decode
        generated_ids = outputs.sequences[0]
        new_tokens = generated_ids[encoded.input_ids.size(1):]
        completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        return {"choices": [{"message": {"role": "assistant", "content": completion}}]}
    
    async def _execute_batch_generate(
        self,
        model_key: str,
        tokenizer,
        device: str,
        contexts: List[List[Dict[str, str]]],
        temperature: float,
        top_p: float
    ) -> List[Dict]:
        """
        Execute a batch generation task.
        
        Args:
            model_key: Key identifying the model to use
            tokenizer: Tokenizer for encoding/decoding
            device: Device where model is located
            contexts: List of conversation contexts
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            List of completion dictionaries
        """
        model = BaseModelManager._base_models[model_key]["model"]
        
        try:
            # Apply chat template to each context
            input_texts = [
                tokenizer.apply_chat_template(
                    context, tokenize=False, add_generation_prompt=True
                )
                for context in contexts
            ]
            
            # Batch encode
            encoded = tokenizer.batch_encode_plus(
                input_texts,
                padding=True,
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors='pt',
                return_attention_mask=True,
            ).to(device)
            
            generate_kwargs = {
                'input_ids': encoded.input_ids,
                'attention_mask': encoded.attention_mask,
                'max_new_tokens': 256,
                'return_dict_in_generate': True,
                'pad_token_id': tokenizer.eos_token_id,
                'output_scores': True,
                'do_sample': True,
                'top_p': top_p,
                'temperature': temperature,
            }
            
            # Add use_cache for non-Gemma models
            if not 'gemma' in model_key.lower():
                generate_kwargs['use_cache'] = True
            
            # Generate
            loop = asyncio.get_running_loop()
            outputs = await loop.run_in_executor(
                None,
                lambda: model.generate(**generate_kwargs)
            )
            
            # Decode
            completions = []
            for i, generated_ids in enumerate(outputs.sequences):
                new_tokens = generated_ids[encoded.input_ids.size(1):]
                completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
                completions.append({"choices": [{"message": {"role": "assistant", "content": completion}}]})
            
            return completions
            
        finally:
            # CRITICAL: Aggressive memory cleanup after each batch to prevent accumulation
            try:
                # Clear local variables immediately
                if 'encoded' in locals():
                    del encoded
                if 'outputs' in locals():
                    del outputs
                
                # Force model to clear any cached states/KV cache
                if hasattr(model, 'reset_cache'):
                    model.reset_cache()
                
                torch.cuda.empty_cache()
                    
            except Exception as cleanup_error:
                print(f"Warning: Batch cleanup error (non-fatal): {cleanup_error}") 