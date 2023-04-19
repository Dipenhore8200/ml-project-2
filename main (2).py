import streamlit as st
from PIL import Image

st.title("Upload and Display Image")

# Create a file uploader component in the sidebar
uploaded_file = st.sidebar.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# Display the uploaded image on the main page
if uploaded_file is not None:
    # Use PIL to open the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image')
else:
    st.warning("No image uploaded.")
