#!/usr/bin/env python3
"""
在评测前 patch baseline 模型的 remote code 文件，使其兼容 transformers >= 4.55。

用法:
    python eval/patch_baseline_model.py <model_path>

会就地修改模型目录下的 .py 文件（如果需要 patch）。
如果不想修改源文件，请先 cp 到临时目录再 patch。
"""

import sys
import os
import re
import json


def patch_configuration(filepath):
    """Fix rope_config_validation import and num_hidden_layers property."""
    with open(filepath, 'r') as f:
        content = f.read()

    modified = False

    # 1. Wrap rope_config_validation import in try/except
    old_import = "from transformers.modeling_rope_utils import rope_config_validation"
    if old_import in content and "try:" not in content.split(old_import)[0].split('\n')[-1]:
        new_import = """try:
    from transformers.modeling_rope_utils import rope_config_validation
except ImportError:
    def rope_config_validation(config):
        return config"""
        content = content.replace(old_import, new_import)
        modified = True
        print(f"  [PATCH] rope_config_validation import wrapped in try/except")

    if modified:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"  [OK] {filepath} patched")
    else:
        print(f"  [SKIP] {filepath} no changes needed")


def patch_config_json(model_path):
    """Patch HF config metadata for vLLM compatibility.

    1. Set dtype metadata to bfloat16 (avoids safetensors dtype scan).
    2. Add vLLM-expected MoE field aliases so that vLLM 0.21's
       TransformersMoEForCausalLM can find top_k, intermediate_size, etc.
    3. Remove invalid empty safetensors placeholders.
    """
    config_path = os.path.join(model_path, "config.json")
    index_path = os.path.join(model_path, "model.safetensors.index.json")

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)

        modified = False

        # --- dtype metadata ---
        for key in ("torch_dtype", "dtype"):
            if config.get(key) != "bfloat16":
                config[key] = "bfloat16"
                modified = True

        # --- vLLM MoE compatibility aliases ---
        # vLLM 0.21 moe.py reads: getattr_iter(config, ["num_experts_per_tok", "top_k"])
        # Longcat uses "moe_topk"
        if "moe_topk" in config and "num_experts_per_tok" not in config:
            config["num_experts_per_tok"] = config["moe_topk"]
            modified = True
            print(f"  [PATCH] Added num_experts_per_tok={config['num_experts_per_tok']} (from moe_topk)")

        # vLLM 0.21 moe.py reads: getattr_iter(config, ["moe_intermediate_size", "intermediate_size"])
        # Longcat uses "expert_ffn_hidden_size" for per-expert FFN dim
        if "expert_ffn_hidden_size" in config and "moe_intermediate_size" not in config:
            config["moe_intermediate_size"] = config["expert_ffn_hidden_size"]
            modified = True
            print(f"  [PATCH] Added moe_intermediate_size={config['moe_intermediate_size']} (from expert_ffn_hidden_size)")

        # vLLM get_num_experts() reads: "num_local_experts" or "n_routed_experts"
        # Longcat already has "n_routed_experts" so this should work, but add
        # "num_local_experts" as well for robustness
        if "n_routed_experts" in config and "num_local_experts" not in config:
            config["num_local_experts"] = config["n_routed_experts"]
            modified = True
            print(f"  [PATCH] Added num_local_experts={config['num_local_experts']} (from n_routed_experts)")

        if modified:
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
                f.write("\n")
            print("  [PATCH] config.json updated")
        else:
            print("  [SKIP] config.json no changes needed")
    else:
        print(f"  [WARN] {config_path} not found, skipping config patch")

    if not os.path.exists(index_path):
        print(f"  [WARN] {index_path} not found, skipping unused shard cleanup")
        return

    with open(index_path, "r") as f:
        index = json.load(f)

    referenced = set(index.get("weight_map", {}).values())
    removed = []
    for name in os.listdir(model_path):
        if name.endswith(".safetensors") and name not in referenced:
            shard_path = os.path.join(model_path, name)
            try:
                os.unlink(shard_path)
                removed.append(name)
            except OSError as exc:
                print(f"  [WARN] failed to unlink unused shard {name}: {exc}")

    if removed:
        print(f"  [PATCH] removed unused safetensors placeholders: {', '.join(sorted(removed))}")
    else:
        print("  [SKIP] no unused safetensors placeholders found")


def patch_modeling(filepath):
    """Fix compatibility issues with transformers >= 4.55."""
    with open(filepath, 'r') as f:
        content = f.read()

    modified = False

    # 0. Fix create_causal_mask() call: remove 'input_embeds' kwarg
    #    In transformers >= 4.55, create_causal_mask no longer accepts input_embeds.
    #    The model code passes input_embeds=... but the function expects inputs_embeds
    #    or doesn't accept it at all. We wrap the function to handle both cases.
    if 'create_causal_mask' in content:
        # Strategy A: If create_causal_mask is defined in this file, patch its signature
        # Use re.DOTALL for multi-line function definitions
        def_pattern = r'def create_causal_mask\(([^)]+)\)'
        def_match = re.search(def_pattern, content, re.DOTALL)
        if def_match:
            params = def_match.group(1)
            if 'input_embeds' not in params and 'inputs_embeds' not in params:
                # The function doesn't accept input_embeds but the caller passes it
                # Add **kwargs to the function signature to absorb extra args
                if '**kwargs' not in params:
                    # Strip trailing whitespace/newlines and commas before appending
                    new_params = params.rstrip().rstrip(',') + ', **kwargs'
                    content = content[:def_match.start(1)] + new_params + content[def_match.end(1):]
                    modified = True
                    print(f"  [PATCH] create_causal_mask: added **kwargs to definition")
            elif 'input_embeds' in params:
                # The function defines input_embeds in its signature but transformers
                # now uses inputs_embeds (with 's'). The caller also uses input_embeds.
                # This case means it's a local function with the old naming - should be fine.
                pass
        else:
            # Strategy B: create_causal_mask is imported from transformers
            # Wrap it at module level to accept and translate input_embeds -> inputs_embeds
            # Find where it's imported
            import_pattern = r'(from\s+[\w.]+\s+import\s+[^\n]*create_causal_mask[^\n]*)'
            import_match = re.search(import_pattern, content)
            if import_match:
                old_import_line = import_match.group(1)
                wrapper = f"""
{old_import_line.replace('create_causal_mask', 'create_causal_mask as _original_create_causal_mask')}
import inspect as _inspect

def create_causal_mask(*args, **kwargs):
    # Compat shim: handle input_embeds naming across transformers versions.
    # - transformers <= 5.8: accepts input_embeds as keyword arg
    # - transformers >= 5.9: removed input_embeds, may use inputs_embeds or none
    _sig = _inspect.signature(_original_create_causal_mask)
    if 'input_embeds' in kwargs:
        if 'input_embeds' in _sig.parameters:
            # Original function accepts input_embeds — pass through unchanged
            pass
        elif 'inputs_embeds' in _sig.parameters:
            # Renamed: input_embeds -> inputs_embeds
            kwargs['inputs_embeds'] = kwargs.pop('input_embeds')
        else:
            # Function no longer needs this arg at all — drop it
            kwargs.pop('input_embeds')
    return _original_create_causal_mask(*args, **kwargs)
"""
                content = content.replace(old_import_line, wrapper)
                modified = True
                print(f"  [PATCH] create_causal_mask: wrapped imported function with compat shim")
            else:
                # Strategy C: create_causal_mask is called but we can't find import or def
                # Replace `input_embeds=` with `inputs_embeds=` in the call site (multi-line safe)
                call_pattern = r'\binput_embeds\b(?=[^=]*=[^=])'
                # More targeted: only within create_causal_mask calls
                # Find all occurrences of input_embeds as a keyword arg near create_causal_mask
                if re.search(r'create_causal_mask\(', content):
                    # Simple but effective: replace input_embeds= that appears after create_causal_mask(
                    new_content = re.sub(
                        r'(create_causal_mask\([\s\S]*?)\binput_embeds\s*=',
                        r'\1inputs_embeds=',
                        content
                    )
                    if new_content != content:
                        content = new_content
                        modified = True
                        print(f"  [PATCH] create_causal_mask: renamed input_embeds -> inputs_embeds in call")

    # 1. Add compute_default_rope_parameters if missing
    if 'compute_default_rope_parameters' not in content:
        # Find the end of LongcatRotaryEmbedding.__init__ (before the next method)
        # Insert after _default_rope_init method
        patch_method = '''
    def compute_default_rope_parameters(self, device=None, seq_len=None):
        """Compatibility shim for transformers >= 4.55 which calls this during _init_weights."""
        import torch
        if device is not None and not isinstance(device, (str, torch.device)):
            device = None
        if device is None and hasattr(self, 'inv_freq'):
            device = self.inv_freq.device
        return self.rope_init_fn(self.config, device)
'''
        # Insert before the forward method of LongcatRotaryEmbedding
        # Find "@torch.no_grad()" that comes after _default_rope_init
        pattern = r'(    @staticmethod\s+\n    def _default_rope_init\(.*?\n(?:.*?\n)*?        return inv_freq, 1\.0\n)'
        match = re.search(pattern, content)
        if match:
            insert_pos = match.end()
            content = content[:insert_pos] + patch_method + content[insert_pos:]
            modified = True
            print(f"  [PATCH] compute_default_rope_parameters added")
        else:
            # Fallback: insert before @torch.no_grad() decorator in LongcatRotaryEmbedding
            # Find the first @torch.no_grad() after class LongcatRotaryEmbedding
            rope_class_match = re.search(r'class LongcatRotaryEmbedding\(nn\.Module\):', content)
            if rope_class_match:
                # Find @torch.no_grad() after rope class
                no_grad_pattern = r'(\n    @torch\.no_grad\(\))'
                no_grad_match = re.search(no_grad_pattern, content[rope_class_match.start():])
                if no_grad_match:
                    insert_pos = rope_class_match.start() + no_grad_match.start()
                    content = content[:insert_pos] + patch_method + content[insert_pos:]
                    modified = True
                    print(f"  [PATCH] compute_default_rope_parameters added (fallback)")

    if modified:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"  [OK] {filepath} patched")
    else:
        print(f"  [SKIP] {filepath} no changes needed")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <model_path>")
        sys.exit(1)

    model_path = sys.argv[1]

    patch_config_json(model_path)

    config_file = os.path.join(model_path, "configuration_longcat_clean.py")
    modeling_file = os.path.join(model_path, "modeling_longcat_clean.py")

    if os.path.exists(config_file):
        print(f"Patching configuration: {config_file}")
        patch_configuration(config_file)
    else:
        print(f"[WARN] {config_file} not found, skipping")

    if os.path.exists(modeling_file):
        print(f"Patching modeling: {modeling_file}")
        patch_modeling(modeling_file)
    else:
        print(f"[WARN] {modeling_file} not found, skipping")

    print("\nDone.")


if __name__ == "__main__":
    main()
