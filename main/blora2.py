# blocklora_full.py
import os
import json
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict

import torch
import torch.nn as nn
from safetensors.torch import save_file as safetensors_save, load_file as safetensors_load


# ============================================================
# 📌 Config
# ============================================================
@dataclass
class BlockLoraConfig:
    r: int = 8
    lora_alpha: int = 16
    target_modules: Optional[List[str]] = None
    lora_dropout: float = 0.0
    bias: str = "none"  # options: "none", "all"
    modules_to_save: Optional[List[str]] = None
    n_blocks: int = 2
    Cheer: str = "ฺBeta"


# ============================================================
# 📌 BlockLoRA Layer (supports multiple B blocks)
# ============================================================
class BlockLoraLayer(nn.Module):
    """
    Blora formulation:
      blora_A: (out_features, r)
      blora_B: list of blocks each (block_size, in_features)
    Effective low-rank delta W = sum_i (A[:, start:start+bs] @ B_i)
    """
    def __init__(self, org_module: nn.Linear, config: BlockLoraConfig):
        super().__init__()
        if not isinstance(org_module, nn.Linear):
            raise ValueError("BlockLoraLayer currently supports nn.Linear only.")
        self.org_module: nn.Linear = org_module
        self.config = config
        self.r = config.r
        self.lora_alpha = config.lora_alpha
        self.scaling = self.lora_alpha / max(1, self.r)
        self.n_blocks = config.n_blocks
        self.dropout = nn.Dropout(config.lora_dropout) if config.lora_dropout > 0 else nn.Identity()

        out_features, in_features = org_module.out_features, org_module.in_features

        # LoRA params
        # blora_A shape: (out_features, r)
        self.blora_A = nn.Parameter(torch.randn(out_features, self.r) * 0.01)
        # split r into blocks
        base = self.r // max(1, self.n_blocks)
        rem = self.r % max(1, self.n_blocks)
        self.block_sizes = [base + (1 if i < rem else 0) for i in range(self.n_blocks)]
        # each blora_B[i] shape: (block_size, in_features)
        self.blora_B = nn.ParameterList([nn.Parameter(torch.zeros(bs, in_features)) for bs in self.block_sizes])

        # bias LoRA
        if config.bias == "all":
            if org_module.bias is None:
                self.bias_lora = nn.Parameter(torch.zeros(out_features))
            else:
                self.bias_lora = nn.Parameter(torch.zeros_like(org_module.bias))
        else:
            self.bias_lora = None

        # merged flag
        self.merged = False

    def _concat_B(self):
        # returns tensor shape (r, in_features)
        if len(self.blora_B) == 1:
            return self.blora_B[0]
        return torch.cat([b for b in self.blora_B], dim=0)

    def forward(self, x):
        """
        Compute: out = x @ (W^T + dW^T) + bias (+ bias_lora)
        where dW = (A @ B_concat) * scaling
        """
        x_d = self.dropout(x)
        W = self.org_module.weight  # (out, in)

        if self.r == 0:
            # no LoRA
            out = x_d @ W.t()
        else:
            B_cat = self._concat_B()  # (r, in)
            dW = self.blora_A @ B_cat  # (out, in)
            dW = dW * self.scaling
            out = x_d @ (W.t() + dW.t())

        if self.org_module.bias is not None:
            out = out + self.org_module.bias

        if self.bias_lora is not None:
            out = out + self.bias_lora

        return out

    @torch.no_grad()
    def merge(self):
        if self.merged or self.r == 0:
            return
        B_cat = self._concat_B()
        dW = (self.blora_A @ B_cat) * self.scaling
        self.org_module.weight.data.add_(dW)
        if self.bias_lora is not None:
            if self.org_module.bias is None:
                # create bias parameter to hold merged bias
                self.org_module.bias = nn.Parameter(self.bias_lora.data.clone())
            else:
                self.org_module.bias.data.add_(self.bias_lora.data)
        # zero out LoRA params to avoid double-apply
        self.blora_A.data.zero_()
        for b in self.blora_B:
            b.data.zero_()
        if self.bias_lora is not None:
            self.bias_lora.data.zero_()
        self.merged = True

    @torch.no_grad()
    def unmerge(self, original_W: Optional[torch.Tensor] = None, original_bias: Optional[torch.Tensor] = None):
        """
        Undo merge only if we kept original W/bias. This API expects caller to manage originals.
        For safety this function does not attempt to reconstruct LoRA params from merged weights.
        """
        if not self.merged:
            return
        # Can't reliably unmerge unless original weights are kept externally.
        # So we do nothing by default. User should avoid merging if they need to preserve LoRA.
        raise RuntimeError("unmerge not supported: original weights not provided. Keep copies before merge.")


# ============================================================
# 📌 BlockLoRA Model
# ============================================================
class BlockLoraModel(nn.Module):
    def __init__(self, model: nn.Module, config: BlockLoraConfig):
        super().__init__()
        self.model = model
        self.config = config
        self.lora_layers: List[Tuple[str, BlockLoraLayer]] = []
        self.adapters: Dict[str, Dict[str, BlockLoraLayer]] = {}  # adapter_name -> {layer_name: BlockLoraLayer}
        self.active_adapter: Optional[str] = None
        self._replace_layers()

    # --- replace target modules ---
    def _replace_layers(self):
        """
        Walk model.named_modules(), replace matched nn.Linear modules by BlockLoraLayer
        and keep self.lora_layers as list of (full_name, layer)
        """
        for name, module in list(self.model.named_modules()):
            if isinstance(module, nn.Linear) and self._match(name):
                parent, attr = self._get_parent(name)
                # Create BlockLoraLayer wrapper that keeps org_module reference
                bl = BlockLoraLayer(module, self.config)
                # Replace in parent
                setattr(parent, attr, bl)
                # Store mapping
                self.lora_layers.append((name, bl))

    def _match(self, name: str) -> bool:
        if self.config.target_modules is None:
            return False
        return any(t in name for t in self.config.target_modules)

    def _get_parent(self, full_name: str):
        tokens = full_name.split(".")
        parent = self.model
        for t in tokens[:-1]:
            parent = getattr(parent, t)
        return parent, tokens[-1]

    # --- merge all layers ---
    def merge(self):
        for _, layer in self.lora_layers:
            layer.merge()

    # --- save pretrained (adapter) ---
    def save_pretrained(self, save_directory: str, merge: bool = False):
        """
        Save current LoRA parameters as an adapter in safetensors format.
        If merge=True, will merge LoRA into base weights before saving (and save merged base)
        """
        os.makedirs(save_directory, exist_ok=True)
        if merge:
            # If merged, we will call merge (this modifies base weights)
            self.merge()

        tensor_state = {}
        for name, layer in self.lora_layers:
            # store A
            tensor_state[name + ".A"] = layer.blora_A.data.clone().cpu()
            # store B blocks
            for i, b in enumerate(layer.blora_B):
                tensor_state[f"{name}.B.{i}"] = b.data.clone().cpu()
            # store bias if present
            if layer.bias_lora is not None:
                tensor_state[name + ".bias_lora"] = layer.bias_lora.data.clone().cpu()

        safetensors_save(tensor_state, os.path.join(save_directory, "adapter.safetensors"))

        # save config for adapter portability
        with open(os.path.join(save_directory, "adapter_config.json"), "w") as f:
            json.dump(self.config.__dict__, f, indent=2)

    # --- load pretrained into THIS model's layers (classmethod builds instance, this loads into instance) ---
    @classmethod
    def from_pretrained(cls, model: nn.Module, load_directory: str, config: Optional[BlockLoraConfig] = None):
        if config is None:
            with open(os.path.join(load_directory, "adapter_config.json"), "r") as f:
                cfg_dict = json.load(f)
            config = BlockLoraConfig(**cfg_dict)

        instance = cls(model, config)
        state = safetensors_load(os.path.join(load_directory, "adapter.safetensors"))

        for name, layer in instance.lora_layers:
            # load A
            keyA = name + ".A"
            if keyA in state:
                layer.blora_A.data.copy_(state[keyA].to(layer.blora_A.device, dtype=layer.blora_A.dtype))
            # load B blocks
            for i in range(len(layer.blora_B)):
                keyB = f"{name}.B.{i}"
                if keyB in state:
                    layer.blora_B[i].data.copy_(state[keyB].to(layer.blora_B[i].device, dtype=layer.blora_B[i].dtype))
            # load bias
            bias_key = name + ".bias_lora"
            if layer.bias_lora is not None and bias_key in state:
                layer.bias_lora.data.copy_(state[bias_key].to(layer.bias_lora.device, dtype=layer.bias_lora.dtype))

        return instance

    # --- print trainable ---
    def print_trainable_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Trainable params: {trainable:,} / {total:,} ({trainable/total:.2%})")

    # ============================================================
    # --- New Adapter Functions (fixed & robust) ---
    # ============================================================

    def _create_adapter_from_state(self, state: Dict[str, torch.Tensor], device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> Dict[str, BlockLoraLayer]:
        """
        Build adapter dict {layer_name: BlockLoraLayer} from a safetensors state mapping.
        Returns a mapping of layer_name -> BlockLoraLayer (disconnected from model).
        """
        adapter = {}
        for name, model_layer in self.lora_layers:
            # create a new BlockLoraLayer with same org_module signature
            new_layer = BlockLoraLayer(model_layer.org_module, self.config)
            target_device = device if device is not None else next(new_layer.parameters()).device
            # copy A
            a_key = name + ".A"
            if a_key not in state:
                raise KeyError(f"Missing key {a_key} in adapter state")
            new_layer.blora_A.data.copy_(state[a_key].to(target_device, dtype=dtype if dtype is not None else state[a_key].dtype))
            # copy B blocks (validate count)
            if len(new_layer.blora_B) == 0:
                # nothing to copy
                pass
            else:
                for i in range(len(new_layer.blora_B)):
                    b_key = f"{name}.B.{i}"
                    if b_key not in state:
                        raise KeyError(f"Missing key {b_key} in adapter state")
                    new_layer.blora_B[i].data.copy_(state[b_key].to(target_device, dtype=dtype if dtype is not None else state[b_key].dtype))
            # bias
            bias_key = name + ".bias_lora"
            if new_layer.bias_lora is not None:
                if bias_key in state:
                    new_layer.bias_lora.data.copy_(state[bias_key].to(target_device, dtype=dtype if dtype is not None else state[bias_key].dtype))
                else:
                    new_layer.bias_lora.data.zero_()
            adapter[name] = new_layer
        return adapter

    def load_adapter(self, load_directory: str, adapter_name: str, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        """
        Load adapter from safetensors file into memory (do NOT apply to model).
        adapter stored as dict of BlockLoraLayer (detached copies).
        """
        path = os.path.join(load_directory, "adapter.safetensors")
        state = safetensors_load(path)
        adapter = self._create_adapter_from_state(state, device=device, dtype=dtype)
        self.adapters[adapter_name] = adapter
        if self.active_adapter is None:
            self.active_adapter = adapter_name

    def apply_adapter_to_model(self, adapter_name: str, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        """
        Copy adapter weights into the model's lora layers (in-place).
        This makes the adapter active for forward() calls.
        """
        if adapter_name not in self.adapters:
            raise ValueError(f"Adapter '{adapter_name}' not loaded")
        adapter = self.adapters[adapter_name]
        # copy each layer
        for name, model_layer in self.lora_layers:
            if name not in adapter:
                raise KeyError(f"Adapter '{adapter_name}' missing layer '{name}'")
            src_layer = adapter[name]
            tgt_device = device if device is not None else next(model_layer.parameters()).device
            # copy A
            model_layer.blora_A.data.copy_(src_layer.blora_A.data.to(tgt_device, dtype=dtype if dtype is not None else src_layer.blora_A.dtype))
            # copy B blocks — validate same number of blocks
            if len(model_layer.blora_B) != len(src_layer.blora_B):
                raise ValueError(f"Block count mismatch for layer {name}: model has {len(model_layer.blora_B)} but adapter has {len(src_layer.blora_B)}")
            for i in range(len(model_layer.blora_B)):
                model_layer.blora_B[i].data.copy_(src_layer.blora_B[i].data.to(tgt_device, dtype=dtype if dtype is not None else src_layer.blora_B[i].dtype))
            # bias
            if model_layer.bias_lora is not None:
                if src_layer.bias_lora is not None:
                    model_layer.bias_lora.data.copy_(src_layer.bias_lora.data.to(tgt_device, dtype=dtype if dtype is not None else src_layer.bias_lora.dtype))
                else:
                    model_layer.bias_lora.data.zero_()
        self.active_adapter = adapter_name

    def set_adapter(self, adapter_name: str, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        """
        Convenience: apply adapter immediately and set active_adapter.
        """
        if adapter_name not in self.adapters:
            raise ValueError(f"Adapter '{adapter_name}' not found")
        self.apply_adapter_to_model(adapter_name, device=device, dtype=dtype)
        # active_adapter set by apply_adapter_to_model

    def delete_adapter(self, adapter_name: str):
        if adapter_name in self.adapters:
            del self.adapters[adapter_name]
            if self.active_adapter == adapter_name:
                self.active_adapter = None
        else:
            print(f"[WARNING] Adapter '{adapter_name}' not found")

    def add_weighted_adapter(self, adapters: List[str], weights: List[float], adapter_name: str, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None, combination_type: str = "sum"):
        """
        Merge multiple loaded adapters into a new adapter with given weights.
        - adapters: list of adapter names (must already be loaded into self.adapters)
        - weights: list of floats (same length)
        - adapter_name: name for the new merged adapter
        - combination_type: currently only "sum" is supported
        Result is stored in self.adapters[adapter_name], and also applied (set_adapter).
        """
        if len(adapters) != len(weights):
            raise ValueError("Length of adapters and weights must match.")

        # Ensure adapters exist
        for a in adapters:
            if a not in self.adapters:
                raise ValueError(f"Adapter '{a}' not found")

        # Build new adapter layers
        new_layers = {}
        for name, model_layer in self.lora_layers:
            # collect corresponding layers from each adapter
            layer_objs = [self.adapters[a][name] for a in adapters]
            # validate compatibility (r, block counts)
            r0 = layer_objs[0].blora_A.shape[1]
            n_blocks0 = len(layer_objs[0].blora_B)
            for lo in layer_objs:
                if lo.blora_A.shape[1] != r0:
                    raise ValueError(f"r mismatch in layer {name} among adapters")
                if len(lo.blora_B) != n_blocks0:
                    raise ValueError(f"block count mismatch in layer {name} among adapters")

            # device for accumulation
            device0 = device if device is not None else layer_objs[0].blora_A.device
            dtype0 = dtype if dtype is not None else layer_objs[0].blora_A.dtype

            # weighted sum for A
            A_acc = torch.zeros_like(layer_objs[0].blora_A, device=device0, dtype=dtype0)
            for lo, w in zip(layer_objs, weights):
                A_acc.add_(lo.blora_A.data.to(device0, dtype=dtype0) * float(w))

            # weighted sum for each B block
            B_acc = []
            for i in range(n_blocks0):
                b_acc = torch.zeros_like(layer_objs[0].blora_B[i], device=device0, dtype=dtype0)
                for lo, w in zip(layer_objs, weights):
                    b_acc.add_(lo.blora_B[i].data.to(device0, dtype=dtype0) * float(w))
                B_acc.append(b_acc)

            # weighted sum for bias if exists
            if layer_objs[0].bias_lora is not None:
                bias_acc = torch.zeros_like(layer_objs[0].bias_lora, device=device0, dtype=dtype0)
                for lo, w in zip(layer_objs, weights):
                    if lo.bias_lora is not None:
                        bias_acc.add_(lo.bias_lora.data.to(device0, dtype=dtype0) * float(w))
            else:
                bias_acc = None

            # create new BlockLoraLayer (detached from model) and fill params
            new_layer = BlockLoraLayer(model_layer.org_module, self.config)
            new_layer.to(device0)
            with torch.no_grad():
                new_layer.blora_A.data.copy_(A_acc)
                for i, b in enumerate(B_acc):
                    new_layer.blora_B[i].data.copy_(b)
                if bias_acc is not None and new_layer.bias_lora is not None:
                    new_layer.bias_lora.data.copy_(bias_acc)

            new_layers[name] = new_layer

        # register adapter
        self.adapters[adapter_name] = new_layers
        # set and apply
        self.set_adapter(adapter_name, device=device, dtype=dtype)

    # --- forward with adapter ---
    def forward(self, *args, **kwargs):
        """
        We assume `set_adapter` was called to apply adapter into model layers.
        If no adapter is active, model runs with whatever weights currently in-place.
        For compatibility, call like:
            blora_model(input_tensor)
        or if model expects different signature, pass accordingly.
        """
        return self.model(*args, **kwargs)


# ============================================================
# helper function
# ============================================================
def get_blocklora_model(model: nn.Module, config: BlockLoraConfig):
    return BlockLoraModel(model, config)


# ============================================================
# Example usage (runnable)
# ============================================================
if __name__ == "__main__":
    import shutil

    # ---------------------------
    # Simple baseline model
    # ---------------------------
    class SimpleModel(nn.Module):
        def __init__(self, in_dim=8, out_dim=4):
            super().__init__()
            self.fc1 = nn.Linear(in_dim, out_dim)
        def forward(self, x):
            return self.fc1(x)

    model = SimpleModel()
    x = torch.randn(2, 8)

    print("=== Original Model Output ===")
    print(model(x))

    # ---------------------------
    # Create BlockLoRA adapter & model wrapper
    # ---------------------------
    config = BlockLoraConfig(r=4, lora_alpha=8, target_modules=["fc1"], n_blocks=2, bias="all")
    blora_model = get_blocklora_model(model, config)

    print("\n=== After Applying BlockLoRA (Random Init) ===")
    # model is inside blora_model
    out = blora_model.model(x)
    print(out)

    # ---------------------------
    # Save adapter
    # ---------------------------
    save_dir = "./temp_adapter"
    shutil.rmtree(save_dir, ignore_errors=True)
    blora_model.save_pretrained(save_dir)
    print("\nSaved adapter to:", save_dir)

    # ---------------------------
    # Load adapter into memory (not applied yet)
    # ---------------------------
    blora_model.load_adapter(save_dir, "adapter_v1")
    print("\nLoaded adapter names:", list(blora_model.adapters.keys()))

    # ---------------------------
    # Apply adapter (set active)
    # ---------------------------
    blora_model.set_adapter("adapter_v1")
    print("Active adapter:", blora_model.active_adapter)

    out_applied = blora_model.model(x)
    print("\nOutput after applying adapter (should be same as earlier random init):")
    print(out_applied)

    # ---------------------------
    # Merge into base weights
    # ---------------------------
    blora_model.merge()
    print("\nAfter merge (LoRA merged into base weights):")
    out_merged = blora_model.model(x)
    print(out_merged)

    # ---------------------------
    # Print trainable (if any)
    # ---------------------------
    blora_model.print_trainable_parameters()
