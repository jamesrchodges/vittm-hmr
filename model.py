import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearCrossAttention(nn.Module):
    """
    O(N) Efficient Attention mechanism.
    Can be used for READ (Process reads Memory) or WRITE (Memory reads Process).
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x_query, x_key_value):
        B, N_q, D = x_query.shape
        B, N_k, D = x_key_value.shape
        
        q = self.q_proj(x_query).reshape(B, N_q, self.num_heads, -1).permute(0, 2, 1, 3)
        k = self.k_proj(x_key_value).reshape(B, N_k, self.num_heads, -1).permute(0, 2, 1, 3)
        v = self.v_proj(x_key_value).reshape(B, N_k, self.num_heads, -1).permute(0, 2, 1, 3)

        # Kernel Trick (ELU + 1)
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0

        # Global Context Aggregation: O(N)
        kv = torch.matmul(k.transpose(-2, -1), v)
        z = 1.0 / (torch.matmul(q, k.sum(dim=-2, keepdim=True).transpose(-2, -1)) + 1e-6)
        
        attn_out = torch.matmul(q, kv) * z
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, N_q, D)
        return self.out_proj(attn_out)

class ViTTMBlock(nn.Module):
    """
    A True Turing Machine Block:
    1. READ: Process tokens read from Memory.
    2. COMPUTE: Process tokens talk to each other (Self-Attn).
    3. WRITE: Process tokens write updates back to Memory.
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., dropout=0.0):
        super().__init__()
        
        # 1. READ (Cross Attn: Query=Process, KV=Memory)
        self.norm_read = nn.LayerNorm(dim)
        self.read_attn = LinearCrossAttention(dim, num_heads=num_heads)
        
        # 2. PROCESS COMPUTE 
        self.norm_proc = nn.LayerNorm(dim)
        self.proc_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        
        # 3. WRITE (Cross Attn: Query=Memory, KV=Process)
        self.norm_write = nn.LayerNorm(dim)
        self.write_attn = LinearCrossAttention(dim, num_heads=num_heads)
        
        # MLPs
        self.norm_mlp_p = nn.LayerNorm(dim)
        self.mlp_process = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(), 
            nn.Linear(int(dim * mlp_ratio), dim)
        )
        
        self.norm_mlp_m = nn.LayerNorm(dim)
        self.mlp_memory = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(), 
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x_process, x_memory):
        # --- A. READ PHASE ---
        # Process tokens gather info from Memory
        x_p_norm = self.norm_read(x_process)
        x_read = self.read_attn(x_query=x_p_norm, x_key_value=x_memory)
        x_process = x_process + x_read
        
        # --- B. COMPUTE PHASE ---
        # Process tokens think/reason
        x_p_norm = self.norm_proc(x_process)
        attn_out, _ = self.proc_attn(x_p_norm, x_p_norm, x_p_norm)
        x_process = x_process + attn_out
        x_process = x_process + self.mlp_process(self.norm_mlp_p(x_process))
        
        # --- C. WRITE PHASE ---
        # Memory updates itself based on Process tokens
        x_m_norm = self.norm_write(x_memory)
        x_write = self.write_attn(x_query=x_m_norm, x_key_value=x_process) 
        x_memory = x_memory + x_write
        x_memory = x_memory + self.mlp_memory(self.norm_mlp_m(x_memory))
        
        return x_process, x_memory

class ViTTM_HMR(nn.Module):
    def __init__(self, 
                 img_size=256, 
                 memory_patch_size=16, 
                 process_patch_size=64, 
                 embed_dim=384, 
                 depth=6, 
                 num_heads=6):
        super().__init__()
        self.embed_dim = embed_dim
        
        # --- 1. INPUT STREAMS ---
        # A. Target Stream (High-Res Ground Truth)
        self.target_embed = nn.Conv2d(3, embed_dim, kernel_size=memory_patch_size, stride=memory_patch_size)
        
        # B. Process Stream (Low-Res Input)
        self.process_embed = nn.Conv2d(3, embed_dim, kernel_size=process_patch_size, stride=process_patch_size)
        self.num_process_tokens = (img_size // process_patch_size) ** 2
        
        # --- 2. MEMORY BANK ---

        self.num_memory_slots = 256
        self.memory_bank = nn.Parameter(torch.randn(1, self.num_memory_slots, embed_dim) * 0.02)
        
        self.process_pos_embed = nn.Parameter(torch.zeros(1, self.num_process_tokens, embed_dim))
        self.process_mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # --- 3. CONTROLLER ---
        self.blocks = nn.ModuleList([
            ViTTMBlock(dim=embed_dim, num_heads=num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # --- 4. DECODER ---
        scale_factor = process_patch_size // memory_patch_size
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=scale_factor, stride=scale_factor),
            nn.GroupNorm(32, embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        )
        
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.trunc_normal_(self.process_pos_embed, std=0.02)
        nn.init.kaiming_normal_(self.target_embed.weight, mode='fan_out')
        nn.init.kaiming_normal_(self.process_embed.weight, mode='fan_out')

    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_keep = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        mask_tokens = self.process_mask_token.repeat(N, L - len_keep, 1)
        x_masked = torch.cat([x_keep, mask_tokens], dim=1)
        x_out = torch.gather(x_masked, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))
        return x_out

    def forward(self, process_view, memory_view, mask_ratio=0.75):
        # 1. PREPARE TARGETS (High-Res)
        with torch.no_grad():
            target_tokens = self.target_embed(memory_view)
            target_tokens = target_tokens.flatten(2).transpose(1, 2)

        # 2. PREPARE INPUT (Low-Res)
        proc_tokens = self.process_embed(process_view)
        proc_tokens = proc_tokens.flatten(2).transpose(1, 2)
        proc_tokens = proc_tokens + self.process_pos_embed
        
        if mask_ratio > 0:
            proc_tokens = self.random_masking(proc_tokens, mask_ratio)

        # 3. INITIALIZE MEMORY STATE
        # Expand the learnable memory bank for this batch
        B = proc_tokens.shape[0]
        curr_memory = self.memory_bank.expand(B, -1, -1) 

        # 4. ViTTM LOOP (Read -> Compute -> Write)
        for blk in self.blocks:
            proc_tokens, curr_memory = blk(x_process=proc_tokens, x_memory=curr_memory)
        
        x = self.norm(proc_tokens)

        # 5. RECONSTRUCT TARGET
        H_p = int(self.num_process_tokens ** 0.5)
        x_2d = x.transpose(1, 2).reshape(B, self.embed_dim, H_p, H_p)
        pred_map = self.decoder(x_2d)
        pred_features = pred_map.flatten(2).transpose(1, 2)

        # 6. LOSS
        target_flat = target_tokens.reshape(-1, self.embed_dim)
        pred_flat = pred_features.reshape(-1, self.embed_dim)
        
        loss = -F.cosine_similarity(pred_flat, target_flat).mean()

        return loss, pred_features, target_tokens