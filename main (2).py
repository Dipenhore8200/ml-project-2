import streamlit as st
from PIL import Image

# Set page title
st.set_page_config(page_title="Image Viewer")

# Define a Streamlit file uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

# If an image is uploaded, display the image using a Streamlit image element
if uploaded_file is not None:
    # Use Pillow to open the uploaded image
    img = Image.open(uploaded_file)
    # Display the image using a Streamlit image element
    st.image(img, caption='Uploaded image')
