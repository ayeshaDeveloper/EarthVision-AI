
import streamlit as st
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import segmentation_models_pytorch as smp
import torch.nn as nn

class SiameseUNet(nn.Module):
    def __init__(self, encoder_name="resnet34", pretrained=False):
        super().__init__()
        self.encoder = smp.encoders.get_encoder(encoder_name, in_channels=3, weights=None)
        encoder_channels = self.encoder.out_channels
        decoder_in_channels = [ch * 2 for ch in encoder_channels]
        
        self.decoder = nn.Sequential(
            nn.Conv2d(decoder_in_channels[-1], 256, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 1, 1)
        )
        
    def forward(self, x1, x2):
        feat1 = self.encoder(x1)
        feat2 = self.encoder(x2)
        fused = torch.cat([feat1[-1], feat2[-1]], dim=1)
        return self.decoder(fused)

@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    lc_model = smp.Unet(encoder_name="resnet34", in_channels=3, classes=4)
    lc_model.load_state_dict(torch.load("models/unet_landcover.pth", map_location=device))
    lc_model.to(device).eval()
    
    cd_model = SiameseUNet(encoder_name="resnet34")
    cd_model.load_state_dict(torch.load("models/siamese_cd.pth", map_location=device))
    cd_model.to(device).eval()
    
    return lc_model, cd_model, device

def preprocess(img, target_size=1024):
    # We resize to 1024x1024 (or keep original) to preserve small buildings
    img = img.convert("RGB").resize((target_size, target_size))
    return TF.normalize(TF.to_tensor(img), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).unsqueeze(0)

st.set_page_config(page_title="EarthVision AI", layout="wide")
st.title("🛰️ EarthVision AI: Urban Expansion Monitoring")

lc_model, cd_model, device = load_models()

col1, col2 = st.columns(2)
with col1:
    file_a = st.file_uploader("Upload BEFORE Image (Year A)", type=["png", "jpg", "jpeg"])
with col2:
    file_b = st.file_uploader("Upload AFTER Image (Year B)", type=["png", "jpg", "jpeg"])

if file_a and file_b:
    if st.button("🚀 Analyze Changes", type="primary"):
        img_a = Image.open(file_a)
        img_b = Image.open(file_b)
        
        # Use 512x512 as a middle ground for memory/speed
        TARGET_SIZE = 512
        t_a = preprocess(img_a, TARGET_SIZE).to(device)
        t_b = preprocess(img_b, TARGET_SIZE).to(device)
        
        with torch.no_grad():
            out_cd = cd_model(t_a, t_b)
            # Resize output back to target size
            out_cd = torch.nn.functional.interpolate(out_cd, size=(TARGET_SIZE, TARGET_SIZE), mode='bilinear')
            mask_prob = torch.sigmoid(out_cd).squeeze().cpu().numpy()
            
            # Standard threshold
            mask = (mask_prob > 0.5).astype(np.uint8)
            
            out_lc_a = lc_model(t_a).argmax(dim=1).squeeze().cpu().numpy()
            out_lc_b = lc_model(t_b).argmax(dim=1).squeeze().cpu().numpy()
            
        total_pixels = TARGET_SIZE * TARGET_SIZE
        changed_pixels = mask.sum()
        change_percent = (changed_pixels / total_pixels) * 100
        
        building_a = (out_lc_a == 0).sum()
        building_b = (out_lc_b == 0).sum()
        building_growth = ((building_b - building_a) / (building_a + 1e-5)) * 100
        
        st.markdown("### 📊 Results")
        m1, m2, m3 = st.columns(3)
        m1.metric("Changed Pixels", f"{changed_pixels:,} px")
        m2.metric("Total Area Changed", f"{change_percent:.2f}%")
        m3.metric("Building Growth", f"{building_growth:+.2f}%", delta=f"{building_growth:.2f}%")
        
        st.markdown("### 🖼️ Visualizations")
        v1, v2, v3 = st.columns(3)
        # Show original images
        v1.image(img_a, caption="Before (Year A)", use_container_width=True)
        v2.image(img_b, caption="After (Year B)", use_container_width=True)
        
        # Create red mask overlay
        colored_mask = np.zeros((TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8)
        colored_mask[mask == 1] = [255, 0, 0] 
        # Resize mask back to original image size for correct overlay
        mask_img = Image.fromarray(colored_mask).resize(img_a.size, Image.NEAREST)
        v3.image(mask_img, caption="Detected Changes (Red)", use_container_width=True)
