import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearCrossAttention(nn.Module):
    """
    O(N) Efficient Attention mechanism for the 'Read' operation.
    Process Tokens (Query) attend to Memory Tokens (Key/Value).
    
    Formula: (elu(Q) + 1) @ ((elu(K) + 1).T @ V)
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
        """
        x_query: Process Tokens [B, N_proc, D]
        x_key_value: Memory Tokens [B, N_mem, D]
        """
        B, N_q, D = x_query.shape
        B, N_k, D = x_key_value.shape
        
        # 1. Project Q, K, V
        q = self.q_proj(x_query).reshape(B, N_q, self.num_heads, -1).permute(0, 2, 1, 3)
        k = self.k_proj(x_key_value).reshape(B, N_k, self.num_heads, -1).permute(0, 2, 1, 3)
        v = self.v_proj(x_key_value).reshape(B, N_k, self.num_heads, -1).permute(0, 2, 1, 3)

        # 2. Kernel Trick (ELU + 1) for non-negative feature map
        # This allows us to re-order multiplication: Q(K^T V) instead of (QK^T)V
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0

        # 3. Efficient Attention Aggregation: O(N)
        # Compute global context from Memory: K.T @ V -> [B, Heads, D_head, D_head]
        # This summarizes the ENTIRE memory bank into a small matrix
        kv = torch.matmul(k.transpose(-2, -1), v)
        
        # Normalize Key Sum for stable attention (denominator)
        z = 1.0 / (torch.matmul(q, k.sum(dim=-2, keepdim=True).transpose(-2, -1)) + 1e-6)
        
        # 4. Read from Memory: Q 
        # Process tokens reading the aggregated memory
        attn_out = torch.matmul(q, kv) * z
        
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, N_q, D)
        return self.out_proj(attn_out)

class ViTTMBlock(nn.Module):
    """
    A single layer of the Vision Token Turing Machine.
    1. Self-Attention (Process <-> Process)
    2. Linear Cross-Attention (Process reads Memory)
    3. MLP
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., dropout=0.0):
        super().__init__()
        
        # Standard Self-Attention for Process Tokens (Context sharing)
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        
        # Read Operation: Process tokens query the high-res Memory
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = LinearCrossAttention(dim, num_heads=num_heads)
        
        # Feed Forward
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )

    def forward(self, x_process, x_memory):
        # 1. Process Self-Interaction (Standard ViT part)
        x_norm = self.norm1(x_process)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x_process = x_process + attn_out
        
        # 2. Read from Memory (ViTTM specific)
        x_norm = self.norm2(x_process)
        read_out = self.cross_attn(x_query=x_norm, x_key_value=x_memory)
        x_process = x_process + read_out
        
        # 3. MLP
        x_process = x_process + self.mlp(self.norm3(x_process))
        
        return x_process

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
        
        # --- 1. TOKENIZATION ---
        
        # Stream A: High-Res Memory Tokens (16x16 patches)
        # 256 / 16 = 16x16 grid = 256 tokens (Large T)
        self.memory_embed = nn.Conv2d(3, embed_dim, kernel_size=memory_patch_size, stride=memory_patch_size)
        self.num_memory_tokens = (img_size // memory_patch_size) ** 2
        
        # Stream B: Low-Res Process Tokens (64x64 patches)
        # 256 / 64 = 4x4 grid = 16 tokens 
        self.process_embed = nn.Conv2d(3, embed_dim, kernel_size=process_patch_size, stride=process_patch_size)
        self.num_process_tokens = (img_size // process_patch_size) ** 2
        
        # Positional Embeddings
        self.memory_pos_embed = nn.Parameter(torch.zeros(1, self.num_memory_tokens, embed_dim))
        self.process_pos_embed = nn.Parameter(torch.zeros(1, self.num_process_tokens, embed_dim))
        self.process_mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # --- 2. ENCODER (The Turing Machine Controller) ---
        self.blocks = nn.ModuleList([
            ViTTMBlock(dim=embed_dim, num_heads=num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # --- 3. PREDICTION HEAD (Decoder) ---
        # Task: Reconstruct the High-Res Memory features from Low-Res Process tokens
        # Upsample 4x4 (Process) back to 16x16 (Memory)
        scale_factor = process_patch_size // memory_patch_size # 64/16 = 4
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=scale_factor, stride=scale_factor),
            nn.GroupNorm(32, embed_dim), # Normalization usually helps reconstruction
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        )
        
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize pos_embeds
        nn.init.trunc_normal_(self.memory_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.process_pos_embed, std=0.02)
        
        # Initialize patches 
        nn.init.kaiming_normal_(self.memory_embed.weight, mode='fan_out')
        nn.init.kaiming_normal_(self.process_embed.weight, mode='fan_out')

    def random_masking(self, x, mask_ratio):
        """
        Masks the Process Tokens (Low Res).
        Returns: masked_x, mask_indices (binary mask not needed for reconstruction here)
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_keep = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # Add mask tokens
        mask_tokens = self.process_mask_token.repeat(N, L - len_keep, 1)
        x_masked = torch.cat([x_keep, mask_tokens], dim=1)
        
        # Unshuffle to restore order 
        x_out = torch.gather(x_masked, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))
        
        return x_out

    def forward(self, process_view, memory_view, mask_ratio=0.75):
        """
        process_view: [B, 3, 256, 256] (Same image, effectively)
        memory_view:  [B, 3, 256, 256] (Same image)
        """
        
        # 1. EMBED MEMORY (Stream A - High Res)
        # [B, 3, 256, 256] -> [B, D, 16, 16] -> [B, 256, D]
        mem_tokens = self.memory_embed(memory_view)
        B, D, H_m, W_m = mem_tokens.shape
        mem_tokens = mem_tokens.flatten(2).transpose(1, 2)
        mem_tokens = mem_tokens + self.memory_pos_embed
        
        # Stop gradient on memory targets 
        with torch.no_grad():
            target_features = mem_tokens.clone().detach()

        # 2. EMBED PROCESS (Stream B - Low Res)
        # [B, 3, 256, 256] -> [B, D, 4, 4] -> [B, 16, D]
        proc_tokens = self.process_embed(process_view)
        proc_tokens = proc_tokens.flatten(2).transpose(1, 2)
        proc_tokens = proc_tokens + self.process_pos_embed
        
        # 3. MASK PROCESS TOKENS
        if mask_ratio > 0:
            proc_tokens = self.random_masking(proc_tokens, mask_ratio)

        # 4. ViTTM ENCODER (Read/Write Loop)
        x = proc_tokens
        for blk in self.blocks:
            # Process tokens update themselves AND read from static Memory tokens
            x = blk(x_process=x, x_memory=mem_tokens)
        
        x = self.norm(x) # [B, 16, D]

        # 5. RECONSTRUCT MEMORY (HMR Task)
        # Reshape to grid for ConvTranspose: [B, 16, D] -> [B, D, 4, 4]
        H_p = int(self.num_process_tokens ** 0.5)
        x_2d = x.transpose(1, 2).reshape(B, D, H_p, H_p)
        
        # Upsample: [B, D, 4, 4] -> [B, D, 16, 16]
        pred_map = self.decoder(x_2d)
        
        # Flatten back: [B, 256, D]
        pred_features = pred_map.flatten(2).transpose(1, 2)

        # 6. LOSS (Cosine Similarity)
        target_flat = target_features.reshape(-1, D)
        pred_flat = pred_features.reshape(-1, D)
        
        loss = -F.cosine_similarity(pred_flat, target_flat).mean()

        return loss, pred_features, target_features