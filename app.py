import os
import streamlit as st 
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from PIL import Image
from io import BytesIO

load_dotenv()

API_TOKEN = os.getenv("HF_API_TOKEN")

client = InferenceClient(
    provider="hf-inference",
    api_key=API_TOKEN
)

MODEL = "black-forest-labs/FLUX.1-schnell"


st.set_page_config(
    page_title="AI_IMAGE_GENERATOR",
    page_icon="🎨",
    layout="centered"
)

st.title("🎨 AI_IMAGE_GENERATOR")
st.write("Generate beautiful AI images using FLUX.1-dev")

prompt = st.text_area(
    "Enter your prompt",
    placeholder="A futuristic city at sunset with flying cars"
)

if st.button("Generate Image"):

    if not prompt.strip():
        st.warning("Please enter a prompt.")
        st.stop()

    if not API_TOKEN:
        st.error("HF_API_TOKEN not found in .env")
        st.stop()

    try:
        with st.spinner("Generating image..."):

            image = client.text_to_image(
                prompt,
                model=MODEL
            )

            st.image(image, caption="Generated Image",
            use_container_width=True)

            buffer = BytesIO()
            image.save(buffer, format="PNG")

            st.download_button(
                label="Download Image",
                data=buffer.getvalue(),
                file_name="generated_image.png",
                mime="image/png"
            )

    except Exception as e:
        st.error(f"Error: {e}")