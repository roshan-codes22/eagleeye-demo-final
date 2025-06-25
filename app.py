import streamlit as st
import torch
import faiss
import json
import numpy as np
from PIL import Image
from torchvision import transforms

# -------------------- Paths --------------------
INDEX_FILE = "demo_image_index.faiss"
MAPPING_FILE = "demo_image_id_mapping.json"

# -------------------- Load Model --------------------
@st.cache_resource
def load_model():
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
    model.eval()
    return model

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model().to(device)

# -------------------- Image Transform --------------------
transform = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# -------------------- Load FAISS Index & Mapping --------------------
index = faiss.read_index(INDEX_FILE)
with open(MAPPING_FILE, "r") as f:
    image_id_mapping = json.load(f)

# -------------------- Search Logic --------------------
def find_most_similar(image):
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(image_tensor).cpu().numpy().astype("float32")
    D, I = index.search(embedding, 1)  # top 1 result
    return image_id_mapping[I[0][0]], D[0][0]

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Part Identifier", layout="centered")
st.title("🔎 Part Identification System (Demo)")
st.markdown("Upload an image to identify the closest matching part using visual similarity.")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing..."):
        result, score = find_most_similar(image)

    st.success("✅ Match Found!")
    st.markdown(f"""
    **Similarity Score:** {score:.2f}

    ### 🧾 Matched Part:
    - **Name:** {result['name']}
    - **Part Number:** {result['part_number']}
    - **NSN:** {result['nsn']}
    - **Reference Image:** `{result['image']}`
    """)

    image_path = f"demo_data_parts/{result['image']}"
    try:
        ref_image = Image.open(image_path)
        st.image(ref_image, caption="Most Similar Part", use_column_width=True)
    except:
        st.warning("Image preview not available.")
