import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ProcessEncoder(nn.Module):
    """
    The 'Student': Sees coarse patches (64x64), learns to predict fine features.
    """
    def __init__(self, img_size=256, patch_size=64, embed_dim=384, depth=6, num_heads=6):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # 256 / 64 = 4 -> 4x4 = 16 tokens
        self.num_patches = (img_size // patch_size) ** 2
        
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * .02)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True, norm_first=True, dropout=0.1
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def random_masking(self, x, mask_ratio):
        """ Standard MAE Masking """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_keep = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # Generate mask tokens and restore order
        mask_tokens = self.mask_token.repeat(N, L - len_keep, 1)
        x_masked = torch.cat([x_keep, mask_tokens], dim=1)
        x_out = torch.gather(x_masked, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))
        
        return x_out

    def forward(self, x, mask_ratio=0.0):
        x = self.patch_embed(x) # [B, D, 4, 4]
        x = x.flatten(2).transpose(1, 2) # [B, 16, D]
        x = x + self.pos_embed
        
        if mask_ratio > 0:
            x = self.random_masking(x, mask_ratio)
            
        x = self.blocks(x)
        x = self.norm(x)
        return x

class ViTTM_HMR(nn.Module):
    def __init__(self, 
                 img_size=256, 
                 embed_dim=384):
        super().__init__()
        
        # --- 1. MEMORY STREAM (Pre-trained Teacher) ---
        # Use ResNet18 trained on ImageNet.
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Remove the FC layer and AvgPool to keep spatial grid
        # ResNet18 structure: layer4 outputs 512 channels, downsample 32x
        # 256px input / 32 = 8x8 grid.
        self.memory_encoder = nn.Sequential(*list(resnet.children())[:-2])
        self.teacher_dim = 512
        
        for param in self.memory_encoder.parameters():
            param.requires_grad = False # FREEZE THE TEACHER

        # --- 2. PROCESS STREAM (Trainable Student) ---
        # Sees coarse context (Patch size 64 -> 16 tokens -> 4x4 grid)
        self.process_encoder = ProcessEncoder(img_size=256, patch_size=64, embed_dim=embed_dim, depth=6)

        # --- 3. PREDICTOR HEAD ---
        # Task: Map Student (4x4, dim=384) -> Teacher (8x8, dim=512)
        self.predictor = nn.Sequential(
            # 1. Upsample 4x4 -> 8x8 (Stride 2)
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2), 
            nn.GELU(),
            # 2. Project Dimensions 384 -> 512
            nn.Conv2d(embed_dim, self.teacher_dim, kernel_size=1) 
        )

    def forward(self, process_view, memory_view, mask_ratio=0.75):
        # 1. GENERATE TARGETS (Teacher)
        with torch.no_grad():
            # ResNet output: [B, 512, 8, 8]
            target_map = self.memory_encoder(memory_view)
            # Flatten to [B, 64, 512]
            target_features = target_map.flatten(2).transpose(1, 2)
            
        # 2. ENCODE PROCESS (Student)
        # Output: [B, 16, 384] (Standard Tokens)
        latent_process = self.process_encoder(process_view, mask_ratio=mask_ratio)
        
        # 3. PREDICT TEACHER FROM STUDENT
        # Reshape to grid: [B, 16, 384] -> [B, 384, 4, 4]
        B, N, D = latent_process.shape
        H_p = int(N**0.5) # Should be 4
        latent_2d = latent_process.transpose(1, 2).reshape(B, D, H_p, H_p)
        
        # Upsample & Project: [B, 384, 4, 4] -> [B, 512, 8, 8]
        pred_map = self.predictor(latent_2d)
        
        # Flatten: [B, 64, 512]
        pred_features = pred_map.flatten(2).transpose(1, 2)

        # 4. LOSS 
        # Minimize Negative Cosine Sim
        target_flat = target_features.reshape(-1, self.teacher_dim)
        pred_flat = pred_features.reshape(-1, self.teacher_dim)
        
        loss = -F.cosine_similarity(pred_flat, target_flat).mean()
        
        return loss, pred_features, target_features