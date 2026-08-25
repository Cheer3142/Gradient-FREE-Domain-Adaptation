import torch
import torch.nn as nn
import os
import json
from safetensors.torch import save_file as safetensors_save, load_file as safetensors_load
from dataclasses import dataclass
from typing import Optional, List

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
    Cheer: str = "Alpha"

# ============================================================
# 📌 BlockLoRA Layer
# ============================================================
class BlockLoraLayer(nn.Module):
    def __init__(self, org_module: nn.Linear, config: BlockLoraConfig):
        super().__init__()
        self.org_module = org_module
        self.config = config
        self.r = config.r
        self.lora_alpha = config.lora_alpha
        self.scaling = self.lora_alpha / self.r
        self.n_blocks = config.n_blocks
        self.dropout = nn.Dropout(config.lora_dropout) if config.lora_dropout > 0 else nn.Identity()

        out_features, in_features = org_module.out_features, org_module.in_features

        # --- LoRA params ---
        self.blora_A = nn.Parameter(torch.randn(out_features, self.r))
        base = self.r // self.n_blocks
        rem = self.r % self.n_blocks
        self.block_sizes = [base + (1 if i < rem else 0) for i in range(self.n_blocks)]
        self.blora_B = nn.ParameterList([nn.Parameter(torch.zeros(bs, in_features)) for bs in self.block_sizes])

        nn.init.normal_(self.blora_A, std=0.01)
        for b in self.blora_B:
            nn.init.zeros_(b)

        # --- Bias LoRA ---
        if config.bias == "all":
            if org_module.bias is None:
                self.bias_lora = nn.Parameter(torch.zeros(out_features))
            else:
                self.bias_lora = nn.Parameter(torch.zeros_like(org_module.bias))
        else:
            self.bias_lora = None

    def forward(self, x):
        x_d = self.dropout(x)
        W = self.org_module.weight

        delta = []
        start = 0
        for i, bs in enumerate(self.block_sizes):
            delta.append(self.blora_A[:, start:start+bs] @ self.blora_B[i])
            start += bs

        dW = torch.stack(delta, dim=0).sum(0) * self.scaling
        out = x_d @ (W.t() + dW.t())

        if self.org_module.bias is not None:
            out = out + self.org_module.bias
        if self.bias_lora is not None:
            out = out + self.bias_lora
            
        return out

    def merge(self):
        delta = []
        start = 0
        for i, bs in enumerate(self.block_sizes):
            delta.append(self.blora_A[:, start:start+bs] @ self.blora_B[i])
            start += bs
        dW = torch.stack(delta, dim=0).sum(0) * self.scaling
        with torch.no_grad():
            self.org_module.weight.data += dW
            if self.bias_lora is not None:
                if self.org_module.bias is None:
                    self.org_module.bias = nn.Parameter(self.bias_lora.data.clone())
                else:
                    self.org_module.bias.data += self.bias_lora.data
        # zero out
        self.blora_A.data.zero_()
        for b in self.blora_B:
            b.data.zero_()
        if self.bias_lora is not None:
            self.bias_lora.data.zero_()

# ============================================================
# 📌 BlockLoRA Model
# ============================================================
class BlockLoraModel(nn.Module):
    def __init__(self, model: nn.Module, config: BlockLoraConfig):
        super().__init__()
        self.model = model
        self.config = config
        self.lora_layers = []
        self._replace_layers()

    # --- replace target modules ---
    def _replace_layers(self):
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d, nn.Embedding)) and self._match(name):
                parent, attr = self._get_parent(name)
                lora_layer = BlockLoraLayer(module, self.config)
                setattr(parent, attr, lora_layer)
                self.lora_layers.append((name, lora_layer))

    def _match(self, name: str):
        if self.config.target_modules is None:
            return False
        return any(t in name for t in self.config.target_modules)

    def _get_parent(self, full_name):
        tokens = full_name.split(".")
        parent = self.model
        for t in tokens[:-1]:
            parent = getattr(parent, t)
        return parent, tokens[-1]

    # --- merge all layers ---
    def merge(self):
        for _, layer in self.lora_layers:
            layer.merge()

    # --- save pretrained ---
    def save_pretrained(self, save_directory: str, merge: bool = False):
        os.makedirs(save_directory, exist_ok=True)
        if merge:
            self.merge()

        state = {}
        for name, layer in self.lora_layers:
            state[name] = {
                "A": layer.blora_A.data.clone(),
                "B": [b.data.clone() for b in layer.blora_B],
                "bias_lora": layer.bias_lora.data.clone() if layer.bias_lora is not None else None
            }

        # convert to safetensors format
        tensor_state = {}
        for k, v in state.items():
            tensor_state[k + ".A"] = v["A"]
            for i, b in enumerate(v["B"]):
                tensor_state[f"{k}.B.{i}"] = b
            if v["bias_lora"] is not None:
                tensor_state[k + ".bias_lora"] = v["bias_lora"]
        safetensors_save(tensor_state, os.path.join(save_directory, "adapter.safetensors"))

        # save config
        with open(os.path.join(save_directory, "adapter_config.json"), "w") as f:
            json.dump(self.config.__dict__, f, indent=2)

    # --- load pretrained ---
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
            layer.blora_A.data.copy_(state[name + ".A"])
            # load B blocks
            for i in range(len(layer.blora_B)):
                layer.blora_B[i].data.copy_(state[f"{name}.B.{i}"])
            # load bias
            if layer.bias_lora is not None and (name + ".bias_lora") in state:
                layer.bias_lora.data.copy_(state[name + ".bias_lora"])
        return instance

    # --- print trainable ---
    def print_trainable_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Trainable params: {trainable:,} / {total:,} ({trainable/total:.2%})")

# ============================================================
# helper function
# ============================================================
def get_blocklora_model(model, config: BlockLoraConfig):
    return BlockLoraModel(model, config)



if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import shutil

    # =======================
    # 1️⃣ สร้างโมเดล baseline
    # =======================
    class SimpleModel(nn.Module):
        def __init__(self, in_dim=8, out_dim=4):
            super().__init__()
            self.fc1 = nn.Linear(in_dim, out_dim)
        def forward(self, x):
            return self.fc1(x)

    model = SimpleModel()
    x = torch.randn(2, 8)  # batch=2, in_features=8

    print("=== Original Model Output ===")
    print(model(x))

    # =======================
    # 2️⃣ สร้าง BlockLoRA Adapter
    # =======================
    config = BlockLoraConfig(r=4, lora_alpha=8, target_modules=["fc1"], n_blocks=2, bias="all")
    blora_model = get_blocklora_model(model, config)

    print("\n=== After Applying BlockLoRA (Random Init) ===")
    out = blora_model.model(x)
    print(out)

    # =======================
    # 3️⃣ Save Adapter
    # =======================
    save_dir = "./temp_adapter"
    shutil.rmtree(save_dir, ignore_errors=True)
    blora_model.save_pretrained(save_dir)
    print("\nSaved adapter to:", save_dir)

    # =======================
    # 4️⃣ Load Adapter
    # =======================
    loaded_model = SimpleModel()
    loaded_blora = BlockLoraModel.from_pretrained(loaded_model, save_dir, config)
    print("\n=== Loaded BlockLoRA Model Output ===")
    out_loaded = loaded_blora.model(x)
    print(out_loaded)

    # =======================
    # 5️⃣ Merge Adapter into Original Weight
    # =======================
    loaded_blora.merge()
    print("\n=== After Merge ===")
    out_merged = loaded_blora.model(x)
    print(out_merged)

    # =======================
    # 6️⃣ Trainable params
    # =======================
    loaded_blora.print_trainable_parameters()

    print("Using BLoRA")
    class SimpleModel(nn.Module):
        def __init__(self, in_dim=8, out_dim=4):
            super().__init__()
            self.fc1 = nn.Linear(in_dim, out_dim)
        def forward(self, x):
            return self.fc1(x)

    model = SimpleModel()
    x = torch.randn(16, 8)  # batch=16
    y = torch.randn(16, 4)  # target

    # ========================
    # 2️⃣ Apply BlockLoRA
    # ========================
    config = BlockLoraConfig(r=4, lora_alpha=8, target_modules=["fc1"], n_blocks=2, bias="all")
    blora_model = get_blocklora_model(model, config)

    # ========================
    # 3️⃣ Freeze original weights
    # ========================
    for name, param in blora_model.model.named_parameters():
        if "blora" not in name:
            param.requires_grad = False

    blora_model.print_trainable_parameters()  # ควรจะเป็น % น้อย ๆ

    # ========================
    # 4️⃣ Optimizer (แค่ trainable params)
    # ========================
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, blora_model.parameters()), lr=1e-2)
    criterion = nn.MSELoss()

    # ========================
    # 5️⃣ Training loop (สั้น ๆ)
    # ========================
    for step in range(10):
        optimizer.zero_grad()
        out = blora_model.model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        print(f"Step {step+1} | Loss: {loss.item():.4f}")

    # ========================
    # 6️⃣ ตรวจ output หลังเทรน
    # ========================
    out_trained = blora_model.model(x)
    print("\nOutput หลังเทรน BlockLoRA:")
    print(out_trained)