"""OpenGraphAU encoder (FAU branch for deepfake detection).

ME-GraphAU trained on a 2M-image hybrid AU corpus → 27 main + 14 sub = 41 AU
heads. Used **frozen** in the deepfake pipeline: the downstream classifier
learns to read AU-trajectory anomalies that distinguish fakes from real
faces, without disturbing AU semantics.

Output
------
forward(x) -> (f_v, cl)
    f_v: [B, 27, out_channels]   per-AU node features after AFG + GNN
    cl : [B, 41]                 per-class scores (27 main + 14 sub)
"""

from __future__ import annotations

import os
import torch
import torch.nn.functional as F

from src.backbones.MEGraphAU.OpenGraphAU.model.MEFL import MEFARG as OpenMEFARG


class FAUEncoderDF(OpenMEFARG):
    def __init__(
        self,
        backbone: str = "swin_transformer_base",
        num_main_classes: int = 27,
        num_sub_classes: int = 14,
    ):
        super().__init__(
            num_main_classes=num_main_classes,
            num_sub_classes=num_sub_classes,
            backbone=backbone,
        )
        self.num_main_classes = num_main_classes
        self.num_sub_classes = num_sub_classes
        self.num_classes = num_main_classes + num_sub_classes

    def load_pretrained(self, checkpoint_path: str) -> bool:
        if not os.path.exists(checkpoint_path):
            print(f"❌ Файл не найден: {checkpoint_path}")
            return False

        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        print(f"✅ Веса загружены из {checkpoint_path}")
        if missing:
            print(f"🔹 Missing: {len(missing)} keys (first 5: {missing[:5]})")
        if unexpected:
            print(f"🔹 Unexpected: {len(unexpected)} keys (first 5: {unexpected[:5]})")
        return True

    def forward(self, x):
        """
        x: [B, 3, 224, 224]
        Returns:
            f_v: [B, 27, out_channels] per-AU node features after GNN
            cl : [B, 41] (main + sub) classification scores
        """
        x = self.backbone(x)
        x = self.global_linear(x)
        head = self.head

        f_u = []
        for layer in head.main_class_linears:
            f_u.append(layer(x).unsqueeze(1))
        f_u = torch.cat(f_u, dim=1)
        f_v = f_u.mean(dim=-2)

        f_e = head.edge_extractor(f_u, x)
        f_e = f_e.mean(dim=-2)
        f_v, _ = head.gnn(f_v, f_e)

        _, n, c = f_v.shape
        main_sc = F.normalize(head.relu(head.main_sc), p=2, dim=-1)
        main_cl = F.normalize(f_v, p=2, dim=-1)
        main_cl = (main_cl * main_sc.view(1, n, c)).sum(dim=-1)

        sub_cl = []
        for i, index in enumerate(head.sub_list):
            au_l, au_r = 2 * i, 2 * i + 1
            main_au = F.normalize(f_v[:, index], p=2, dim=-1)
            sc_l = F.normalize(head.relu(head.sub_sc[au_l]), p=2, dim=-1)
            sc_r = F.normalize(head.relu(head.sub_sc[au_r]), p=2, dim=-1)
            sub_cl.append((main_au * sc_l.view(1, c)).sum(dim=-1)[:, None])
            sub_cl.append((main_au * sc_r.view(1, c)).sum(dim=-1)[:, None])
        sub_cl = torch.cat(sub_cl, dim=-1)
        cl = torch.cat([main_cl, sub_cl], dim=-1)

        return f_v, cl


if __name__ == "__main__":
    enc = FAUEncoderDF(backbone="resnet50")
    x = torch.randn(2, 3, 224, 224)
    f_v, cl = enc(x)
    print("f_v:", f_v.shape, "  cl:", cl.shape, "  out_channels:", enc.out_channels)
