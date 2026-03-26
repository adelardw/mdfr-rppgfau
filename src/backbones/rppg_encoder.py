import torch
import torch.nn as nn
import os
from src.backbones.rPPGToolbox.neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX as PhysNet

class RPPGEncoder(PhysNet):
    def __init__(self, frames=128):
        super(RPPGEncoder, self).__init__(frames=frames)
        self.out_channels = self.ConvBlock10.in_channels

    def load_pretrained(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            print(f"❌ Файл не найден: {checkpoint_path}")
            return False

        state_dict = torch.load(checkpoint_path, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        print(f"✅ Веса загружены из {checkpoint_path}")
        print(f"🔹 Пропущен слой (нормально): {missing}")
        return True

    def forward(self, x):
        """
        Input: x (Batch, 3, Frames, 128, 128)
        Output: features (Batch, 64, Frames)
        """
        B, C, T, H, W = x.shape

        x = self.ConvBlock1(x)
        x = self.MaxpoolSpa(x)
        x = self.ConvBlock2(x)
        x_visual6464 = self.ConvBlock3(x)
        x = self.MaxpoolSpaTem(x_visual6464)
        x = self.ConvBlock4(x)
        x_visual3232 = self.ConvBlock5(x)
        x = self.MaxpoolSpaTem(x_visual3232)
        x = self.ConvBlock6(x)
        x_visual1616 = self.ConvBlock7(x)
        x = self.MaxpoolSpa(x_visual1616)
        x = self.ConvBlock8(x)
        x = self.ConvBlock9(x)
        x = self.upsample(x)
        x = self.upsample2(x)
        pool = self.poolspa(x)
        x = self.ConvBlock10(pool)

        rPPG = x.view(-1, T)
        return rPPG, pool.squeeze(-1).squeeze(-1).transpose(1, 2)


if __name__ == "__main__":

    frames = 16
    encoder = RPPGEncoder(frames=frames)
    encoder.load_pretrained('./src/backbones/rPPGToolbox/final_model_release/PURE_PhysNet_DiffNormalized.pth')
    encoder.eval()

    batch_size = 2
    channels = 3
    height = 224
    width = 224

    dummy_input = torch.randn(batch_size, channels, frames, height, width)
    print(f"\nВходной тензор: {dummy_input.shape} (Batch, C, T, H, W)")

    with torch.no_grad():
        rPPG, features= encoder(dummy_input)
        print(f"Выходные признаки: {features.shape} (Batch, T, 64)")
        print(f"Выходные признаки: {rPPG.shape} (Batch, T)")