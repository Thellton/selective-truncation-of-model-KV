from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import types # For monkey-patching
from typing import Tuple, Optional, Callable 
from transformers.cache_utils import Cache # For type hinting and checking
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs 
from transformers.utils import logging 

# Assuming Qwen3 reuses Qwen2's utility functions or they are similarly structured
# Adjust the import path if Qwen3 has its own dedicated modeling file in your transformers version
from transformers.models.qwen2.modeling_qwen2 import (
    eager_attention_forward,
    apply_rotary_pos_emb
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

logger = logging.get_logger(__name__)

# --- Configuration for Selective KV ---
REDUCED_KV_SIZE = 512
# For Qwen3-0.6B with 28 layers (0-27)
# Example: layers 20 and 23 get full KV from your last run
FULL_KV_LAYERS = {0, 1, 3, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 27}
# FULL_KV_LAYERS = {0, 27} # Example: first and last layer get full KV
max_new_tokens_to_generate = 1024
#test prompt
prompt = """A large language model (LLM) is a type of machine learning model designed for natural language processing tasks such as language generation. LLMs are language models with many parameters, and are trained with self-supervised learning on a vast amount of text. The largest and most capable LLMs are generative pretrained transformers (GPTs). Modern models can be fine-tuned for specific tasks or guided by prompt engineering.[1] These models acquire predictive power regarding syntax, semantics, and ontologies[2] inherent in human language corpora, but they also inherit inaccuracies and biases present in the data they are trained in. Before 2017, there were a few language models that were large as compared to capacities then available. In the 1990s, the IBM alignment models pioneered statistical language modelling. A smoothed n-gram model in 2001 trained on 0.3 billion words achieved state-of-the-art perplexity at the time.[4] In the 2000s, as Internet use became prevalent, some researchers constructed Internet-scale language datasets ("web as corpus"[5]), upon which they trained statistical language models.[6][7] In 2009, in most language processing tasks, statistical language models dominated over symbolic language models because they can usefully ingest large datasets."""


model_name = "D:/AI-Art-tools/Qwen3 safetensors/Model_files/Qwen3-0.6B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="cpu"
)
model.eval()

# Store original forward methods
original_attention_forwards = {}

def get_layer_specific_attention_forward(layer_idx, original_forward_method):
    def modified_attention_forward(
        self: "Qwen3Attention",
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs, 
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        input_shape = hidden_states.shape[:-1]
        
        q_hidden_shape = (*input_shape, self.config.num_attention_heads, self.head_dim)
        query_states = self.q_norm(self.q_proj(hidden_states).view(q_hidden_shape)).transpose(1, 2)

        kv_hidden_shape = (*input_shape, self.config.num_key_value_heads, self.head_dim)
        new_key_states = self.k_norm(self.k_proj(hidden_states).view(kv_hidden_shape)).transpose(1, 2)
        new_value_states = self.v_proj(hidden_states).view(kv_hidden_shape).transpose(1, 2)
        
        cos, sin = position_embeddings
        query_states, new_key_states = apply_rotary_pos_emb(query_states, new_key_states, cos, sin)

        key_for_attention = new_key_states
        value_for_attention = new_value_states
        effective_attention_mask = attention_mask

        use_cache = kwargs.get("use_cache", False) # Check if use_cache is active

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_for_attention, value_for_attention = past_key_value.update(
                new_key_states, new_value_states, self.layer_idx, cache_kwargs
            )
#start of mod
        is_limited_layer = self.layer_idx not in FULL_KV_LAYERS

        if is_limited_layer and past_key_value is not None: # Only truncate if there's a past to truncate from/with
            # 1. Truncate K/V that will be USED for the current attention computation
            # K/V shape is (batch_size, num_kv_heads, seq_len, head_dim)
            current_kv_seq_len = key_for_attention.shape[-2] # This is AFTER past_key_value.update()
            
            # Define the target size when truncation occurs
            # Ensure target size is at least 1 if REDUCED_KV_SIZE is very small (e.g., 1)
            TARGET_TRUNCATED_SIZE = max(1, REDUCED_KV_SIZE - 1)

            # Condition: if current length hits or exceeds the nominal REDUCED_KV_SIZE
            if current_kv_seq_len >= REDUCED_KV_SIZE: # Changed from > to >=
                #print(f"Layer {self.layer_idx} (ATTN): Length {current_kv_seq_len} >= {REDUCED_KV_SIZE}. Truncating K/V for attention to {TARGET_TRUNCATED_SIZE}")
                # print(f"[Layer {self.layer_idx}] post‐merge cache length (before this truncation): {past_key_value.key_cache[self.layer_idx].shape[-2]}") # This would be current_kv_seq_len

                key_for_attention = key_for_attention[:, :, -TARGET_TRUNCATED_SIZE:, :]
                value_for_attention = value_for_attention[:, :, -TARGET_TRUNCATED_SIZE:, :]
                if effective_attention_mask is not None:
                    # The mask should now correspond to the new TARGET_TRUNCATED_SIZE
                    effective_attention_mask = effective_attention_mask[..., :, -TARGET_TRUNCATED_SIZE:] 
            
            # 3. Truncate what's actually STORED in the cache object for *future* steps.
            # This must be done on the cache object's internal tensors.
            if use_cache and hasattr(past_key_value, 'key_cache') and hasattr(past_key_value, 'value_cache'):
                # These are lists of tensors, one per layer.
                # `past_key_value.key_cache[self.layer_idx]` is the tensor for the current layer.
                stored_k_cache = past_key_value.key_cache[self.layer_idx]
                if stored_k_cache.shape[-2] >= REDUCED_KV_SIZE: # Check if its current length triggers truncation
                    # print(f"Layer {self.layer_idx} (STORED_CACHE K): Length {stored_k_cache.shape[-2]} >= {REDUCED_KV_SIZE}. Truncating to {TARGET_TRUNCATED_SIZE}")
                    past_key_value.key_cache[self.layer_idx] = stored_k_cache[:, :, -TARGET_TRUNCATED_SIZE:, :]
                
                stored_v_cache = past_key_value.value_cache[self.layer_idx]
                if stored_v_cache.shape[-2] >= REDUCED_KV_SIZE:
                    # print(f"Layer {self.layer_idx} (STORED_CACHE V): Length {stored_v_cache.shape[-2]} >= {REDUCED_KV_SIZE}. Truncating to {TARGET_TRUNCATED_SIZE}")
                    past_key_value.value_cache[self.layer_idx] = stored_v_cache[:, :, -TARGET_TRUNCATED_SIZE:, :]
#end code block        
        attention_interface_fn: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                 logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention.'
                )
            else:
                try:
                    attention_interface_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
                except KeyError:
                    logger.warning_once(
                        f"Attention implementation {self.config._attn_implementation} not found. Falling back to eager."
                    )
                    attention_interface_fn = eager_attention_forward

        attn_output, attn_weights = attention_interface_fn(
            self, 
            query_states,
            key_for_attention,   
            value_for_attention, 
            effective_attention_mask, 
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window, 
            **kwargs, 
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
    return modified_attention_forward

# --- Apply the Patch ---
num_layers = model.config.num_hidden_layers
if num_layers is None:
    num_layers = len(model.model.layers)

for i in range(num_layers):
    try:
        attention_module = model.model.layers[i].self_attn
        original_attention_forwards[i] = attention_module.forward
        attention_module.forward = types.MethodType(
            get_layer_specific_attention_forward(i, original_attention_forwards[i]), 
            attention_module
        )
        print(f"Patched Qwen3Attention for layer {i}")
    except AttributeError as e:
        print(f"Could not patch layer {i}. Attribute error: {e}. Check model structure.")
        break

messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

prompt_num_tokens = model_inputs.input_ids.shape[1]
print(f"\nNumber of prompt tokens: {prompt_num_tokens}")

print(f"Generating with Selective KV (Reduced: {REDUCED_KV_SIZE}, Full for layers: {FULL_KV_LAYERS})...")


# Generate text and get the final KV cache
outputs = model.generate(
    **model_inputs,
    use_cache=True,
    max_new_tokens=max_new_tokens_to_generate,
    return_dict_in_generate=True,  # <<< Crucial for getting past_key_values
    # output_scores=True, # Not strictly needed for past_key_values but often used with return_dict
    # output_attentions=True, # Could also force past_key_values to be returned
    # output_hidden_states=True, # Could also force past_key_values to be returned
)
generated_sequences = outputs.sequences
final_past_key_values = outputs.past_key_values # This should be the Cache object

# Extract only newly generated token IDs
output_ids = generated_sequences[0][prompt_num_tokens:].tolist()
generated_num_tokens = len(output_ids)
total_sequence_tokens = prompt_num_tokens + generated_num_tokens # Total length of sequence in cache for full layers

print(f"Number of generated tokens: {generated_num_tokens}")
print(f"Total tokens in sequence (prompt + generated): {total_sequence_tokens}")

# Decode the generated content (handling potential Qwen "thinking" sections)
# This is a simplified way to remove common Qwen thinking blocks if present in output_ids.
# If your specific Qwen3 version doesn't do this or uses different tokens, adjust accordingly.
decoded_text = tokenizer.decode(output_ids, skip_special_tokens=False) # Decode with special tokens first
thinking_start_token = "<think>" # Replace with actual if different
thinking_end_token = "</think>" # Replace with actual if different

content_to_print = decoded_text
if thinking_start_token in decoded_text and thinking_end_token in decoded_text:
    start_think = decoded_text.find(thinking_start_token)
    end_think = decoded_text.rfind(thinking_end_token) + len(thinking_end_token)
    # Assuming thinking block is at the beginning
    if start_think == 0 and end_think > 0:
         # Extract content after thinking block
        actual_content_part = decoded_text[end_think:]
        # Re-decode only the thinking part if needed for display, and the rest with skip_special_tokens
        thinking_block_decoded = tokenizer.decode(tokenizer.encode(decoded_text[:end_think]), skip_special_tokens=True).strip()
        content_final_decoded = tokenizer.decode(tokenizer.encode(actual_content_part), skip_special_tokens=True).strip()
        
        print("\n--- Thinking Content (Decoded) ---")
        print(thinking_block_decoded)
        content_to_print = content_final_decoded
    else: # Fallback to simple decode if thinking block is not structured as expected
        content_to_print = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
else:
    content_to_print = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

print("\n--- Generated Content (Final) ---")
print(content_to_print)
print("--- End of Content ---")


# --- KV Cache Statistics ---
print("\n--- KV Cache Statistics (Post-Generation) ---")
if final_past_key_values is None:
    print("Could not retrieve final past_key_values from generate output.")
elif not isinstance(final_past_key_values, Cache):
    print(f"Retrieved past_key_values is not a Cache object. Type: {type(final_past_key_values)}")
    print("Cannot reliably parse KV cache statistics for this type.")
else:
    total_kv_cache_memory_bytes = 0
    
    # Access internal lists for key and value caches from the Cache object
    # This structure is typical for DynamicCache.
    try:
        key_caches_list = final_past_key_values.key_cache 
        value_caches_list = final_past_key_values.value_cache
        
        if not (isinstance(key_caches_list, list) and isinstance(value_caches_list, list) and \
                len(key_caches_list) == num_layers and len(value_caches_list) == num_layers):
            print("KV cache lists inside the Cache object are not structured as expected (list per layer).")
        else:
            for i in range(num_layers):
                k_cache_layer = key_caches_list[i] 
                v_cache_layer = value_caches_list[i]

                if k_cache_layer is None or v_cache_layer is None:
                    print(f"Layer {i:02d}: Cache tensor is None.")
                    continue

                actual_seq_len = k_cache_layer.shape[2] # Shape: (batch_size, num_kv_heads, seq_len, head_dim)
                
                k_mem_bytes = k_cache_layer.nelement() * k_cache_layer.element_size()
                v_mem_bytes = v_cache_layer.nelement() * v_cache_layer.element_size()
                layer_total_mem_bytes = k_mem_bytes + v_mem_bytes
                total_kv_cache_memory_bytes += layer_total_mem_bytes

                layer_type = "FULL   " if i in FULL_KV_LAYERS else "REDUCED"
                
                expected_len_info = ""
                if layer_type == "FULL   ":
                    expected_len_info = f"(Expected ~{total_sequence_tokens})"
                else: # REDUCED
                    expected_len_info = f"(Expected <={REDUCED_KV_SIZE if total_sequence_tokens > REDUCED_KV_SIZE else total_sequence_tokens})"


                print(f"Layer {i:02d} ({layer_type}): SeqLen={actual_seq_len:<5} {expected_len_info}, "
                      f"K_mem={k_mem_bytes/1024/1024:6.2f} MB, V_mem={v_mem_bytes/1024/1024:6.2f} MB, "
                      f"LayerTotal={layer_total_mem_bytes/1024/1024:6.2f} MB")
                # Uncomment for more detail:
                # print(f"                 K_shape: {list(k_cache_layer.shape)}, V_shape: {list(v_cache_layer.shape)}, Dtype: {k_cache_layer.dtype}")

            print(f"\nTotal Estimated KV Cache Memory (all layers): {total_kv_cache_memory_bytes / 1024 / 1024:.2f} MB")

    except AttributeError:
        print("The Cache object does not have '.key_cache' or '.value_cache' attributes as lists.")
        print("The internal structure of the returned Cache object might be different for this model/version.")
        print(f"Cache object type: {type(final_past_key_values)}")
        # You might need to inspect `final_past_key_values` to see how to access per-layer K/V tensors.
        # For example, it could be a tuple of tuples if not a Cache object:
        # if isinstance(final_past_key_values, tuple) and len(final_past_key_values) == num_layers:
        #    #  Process as tuple of (k,v) tuples per layer
        # else:
        #    print("Unhandled cache structure.")


# --- To restore original methods (optional) ---
# ... (your restoration code)

'''
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import types # For monkey-patching
from typing import Tuple, Optional, Callable # Added Callable
from transformers.cache_utils import Cache # Added Cache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter # Might be needed for mask adjustment details
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs # Added for type hinting **kwargs
from transformers.utils import logging # Added for type hinting **kwargs

# --- Configuration for Selective KV ---
REDUCED_KV_SIZE = 512 
# For Qwen3-0.6B with 28 layers (0-27)
FULL_KV_LAYERS = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15} # Example: first and last layer get full KV

# model_name = "./Model files/Qwen3-0.6B"
model_name = "D:/AI-Art-tools/Qwen3 safetensors/Model_files/Qwen3-0.6B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto", # Or torch.float32 if "auto" causes issues on CPU
    device_map="cpu"
)
model.eval()

# Store original forward methods
original_attention_forwards = {}

# --- Eager Attention Forward (copied from Qwen3 source for completeness if needed) ---
# We will call the original one, but having this as reference is good.
# The `module` argument in `eager_attention_forward` is `self` from `Qwen3Attention`
# from transformers.models.qwen2.modeling_qwen2 import eager_attention_forward, repeat_kv, apply_rotary_pos_emb 
# Assuming Qwen3 reuses Qwen2's eager_attention_forward, or has its own.
# For now, we'll rely on the original `attention_interface` call.

# Access to Qwen3 specific functions if not automatically imported or to ensure we use the right ones
from transformers.models.qwen2.modeling_qwen2 import ( # Adjust path if qwen3 has its own file
    eager_attention_forward, 
    # repeat_kv, # This is used inside eager_attention_forward
    # rotate_half, # Used by apply_rotary_pos_emb
    apply_rotary_pos_emb
)
# We need ALL_ATTENTION_FUNCTIONS if we want to replicate the SDPA selection logic
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

#logger = logging.getLogger(__name__) # For potential warnings

def get_layer_specific_attention_forward(layer_idx, original_forward_method):
    
    # These need to be available to the inner function
    # Alternatively, pass them as arguments or make them part of a class
    # For monkey-patching, they are global here for simplicity.
    # global REDUCED_KV_SIZE, FULL_KV_LAYERS

    def modified_attention_forward(
        self: "Qwen3Attention", # `self` is the instance of Qwen3Attention
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # **kwargs: dict, # Or just **kwargs without a specific type hint if preferred
        **kwargs, # Python itself handles **kwargs, the type hint is for static analysis
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]: # Matching expected output for decoder layer

        # --- Original Q, K, V projection and RoPE ---
        input_shape = hidden_states.shape[:-1]
        
        # Correctly determine hidden_shape for Q, K, V based on num_attention_heads vs num_key_value_heads
        # Q uses num_attention_heads
        q_hidden_shape = (*input_shape, self.config.num_attention_heads, self.head_dim)
        query_states = self.q_norm(self.q_proj(hidden_states).view(q_hidden_shape)).transpose(1, 2)

        # K and V use num_key_value_heads
        kv_hidden_shape = (*input_shape, self.config.num_key_value_heads, self.head_dim)
        new_key_states = self.k_norm(self.k_proj(hidden_states).view(kv_hidden_shape)).transpose(1, 2)
        new_value_states = self.v_proj(hidden_states).view(kv_hidden_shape).transpose(1, 2)
        
        cos, sin = position_embeddings
        query_states, new_key_states = apply_rotary_pos_emb(query_states, new_key_states, cos, sin)
        # `new_key_states` and `new_value_states` are for the current token(s)

        key_for_attention = new_key_states
        value_for_attention = new_value_states
        effective_attention_mask = attention_mask # Will be modified if K/V are truncated

        use_cache = kwargs.get("use_cache", False) # Check if use_cache is active from kwargs or config

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            # .update() concatenates new K/V to cache AND returns the full concatenated K/V for this step's attention
            # So, `key_for_attention` and `value_for_attention` will hold the full history here.
            key_for_attention, value_for_attention = past_key_value.update(
                new_key_states, new_value_states, self.layer_idx, cache_kwargs
            )

        is_limited_layer = self.layer_idx not in FULL_KV_LAYERS

        if is_limited_layer and past_key_value is not None : # Only truncate if there's a past to truncate from/with
            # 1. Truncate K/V that will be USED for the current attention computation
            # K/V shape is (batch_size, num_kv_heads, seq_len, head_dim)
            current_kv_seq_len = key_for_attention.shape[-2] # seq_len is dim 2
            if current_kv_seq_len > REDUCED_KV_SIZE:
                # print(f"Layer {self.layer_idx} (ATTN): Truncating K/V for attention from {current_kv_seq_len} to {REDUCED_KV_SIZE}")
                key_for_attention = key_for_attention[:, :, -REDUCED_KV_SIZE:, :]
                value_for_attention = value_for_attention[:, :, -REDUCED_KV_SIZE:, :]

                # 2. Truncate the attention mask's key dimension accordingly
                # attention_mask is typically (batch_size, 1, query_seq_len, key_seq_len)
                if effective_attention_mask is not None:
                    # print(f"Layer {self.layer_idx} (ATTN_MASK): Slicing mask from {effective_attention_mask.shape[-1]} to {key_for_attention.shape[-2]}")
                    effective_attention_mask = effective_attention_mask[..., :, -key_for_attention.shape[-2]:]
            
            # 3. Truncate what's actually STORED in the cache object for *future* steps.
            # This must be done on the cache object's internal tensors.
            if use_cache and hasattr(past_key_value, 'key_cache') and hasattr(past_key_value, 'value_cache'):
                # These are lists of tensors, one per layer.
                # `past_key_value.key_cache[self.layer_idx]` is the tensor for the current layer.
                stored_k_cache = past_key_value.key_cache[self.layer_idx]
                if stored_k_cache.shape[-2] > REDUCED_KV_SIZE:
                    # print(f"Layer {self.layer_idx} (STORED_CACHE K): Truncating from {stored_k_cache.shape[-2]} to {REDUCED_KV_SIZE}")
                    past_key_value.key_cache[self.layer_idx] = stored_k_cache[:, :, -REDUCED_KV_SIZE:, :]
                
                stored_v_cache = past_key_value.value_cache[self.layer_idx]
                if stored_v_cache.shape[-2] > REDUCED_KV_SIZE:
                    # print(f"Layer {self.layer_idx} (STORED_CACHE V): Truncating from {stored_v_cache.shape[-2]} to {REDUCED_KV_SIZE}")
                    past_key_value.value_cache[self.layer_idx] = stored_v_cache[:, :, -REDUCED_KV_SIZE:, :]
        
        # --- Original Attention Computation Logic ---
        attention_interface_fn: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            # output_attentions = kwargs.get("output_attentions", False) # Get from kwargs
            # Using self.training instead of output_attentions for SDPA check, as per original code.
            # However, the warning for SDPA is about output_attentions=True.
            # Let's try to mirror the original logic more closely.
            # The `output_attentions` flag is passed into the Qwen3DecoderLayer.forward and then to Qwen3Attention.forward via **kwargs.
            # Let's assume it's correctly in `kwargs` if needed.
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                 logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                try:
                    attention_interface_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
                except KeyError:
                    logger.warning_once(
                        f"Attention implementation {self.config._attn_implementation} not found. Falling back to eager."
                    )
                    attention_interface_fn = eager_attention_forward


        # Note: `kwargs` might contain `output_attentions`
        attn_output, attn_weights = attention_interface_fn(
            self, # module (Qwen3Attention instance)
            query_states,
            key_for_attention,    # Potentially truncated K
            value_for_attention,  # Potentially truncated V
            effective_attention_mask, # Potentially sliced mask
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window, # Qwen3Attention has this attribute
            **kwargs, # Pass through other flash attention kwargs
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights # attn_weights might be None if not computed

    return modified_attention_forward

# --- Apply the Patch ---
num_layers = model.config.num_hidden_layers
if num_layers is None: # Fallback for some model configs
    num_layers = len(model.model.layers)


for i in range(num_layers):
    try:
        attention_module = model.model.layers[i].self_attn
        original_attention_forwards[i] = attention_module.forward
        attention_module.forward = types.MethodType(
            get_layer_specific_attention_forward(i, original_attention_forwards[i]), 
            attention_module
        )
        print(f"Patched Qwen3Attention for layer {i}")
    except AttributeError as e:
        print(f"Could not patch layer {i}. Attribute error: {e}. Check model structure.")
        break

# --- The rest of your script for testing generation ---
# (Make sure to set use_cache=True in model.generate call)

# Example test
prompt = "what is your favourite place in the world? sell me on a vacation to it!"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

print(f"\nGenerating with Selective KV (Reduced: {REDUCED_KV_SIZE}, Full for layers: {FULL_KV_LAYERS})...")
# Ensure use_cache=True to activate KV caching logic
generated_ids = model.generate(
    **model_inputs,
    use_cache=True, 
    max_new_tokens=1024, # Keep it manageable for testing
    # past_key_values=None # Initially None
    # Add other generation params as needed, e.g., do_sample, temperature
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

# Handling potential thinking content if your model/tokenizer uses it (like Qwen1.5)
# For Qwen3, this might not be standard, adjust as needed.
# Assuming tokenizer.decode handles special tokens appropriately.
# If the Qwen3 tokenizer doesn't have specific thinking tokens like 151668, this part can be simplified.
try:
    # A generic way to remove prompt, assuming no complex thinking tokens here for simplicity
    # This part needs to align with how Qwen3's template and special tokens work if any are added by generate.
    # For now, let's assume a simple decode without specific thinking token parsing.
    content = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
except ValueError:
    content = "Error during decoding or no new tokens."


print("\n--- Generated Content ---")
print(content)
print("--- End of Content ---")

# --- To restore original methods (optional) ---
# for i in range(num_layers):
#     if i in original_attention_forwards:
#         try:
#             attention_module = model.model.layers[i].self_attn
#             attention_module.forward = original_attention_forwards[i]
#             # print(f"Restored original attention for layer {i}")
#         except AttributeError:
#             pass

'''