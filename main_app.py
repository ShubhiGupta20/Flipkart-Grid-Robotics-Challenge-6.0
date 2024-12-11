import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
import pandas as pd
import sqlite3
from datetime import datetime
from tasks.brand import detect_brand
from tasks.expiry import preprocess_image, extract_expiry_date
from tasks.count import count_objects
from tasks.freshness import detect_freshness

# Flipkart color scheme
PRIMARY_COLOR = "#2874F0"  # Flipkart blue
SECONDARY_COLOR = "#FFFFFF"  # White
ACCENT_COLOR = "#FFD700"  # Gold for highlights

# App styling
st.markdown(
    f"""
    <style>
        .main {{
            background-color: {SECONDARY_COLOR};
        }}
        .title-container {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 20px;
            margin-bottom: 20px;
        }}
        .title {{
            font-family: 'Arial', sans-serif;
            font-size: 32px;
            color: {PRIMARY_COLOR};
            text-align: center;
        }}
        .sidebar .sidebar-content {{
            background-color: {PRIMARY_COLOR};
            color: {SECONDARY_COLOR};
        }}
        .stButton>button {{
            background-color: {PRIMARY_COLOR};
            color: {SECONDARY_COLOR};
            border: None;
            border-radius: 8px;
        }}
        .stButton>button:hover {{
            background-color: {ACCENT_COLOR};
            color: black;
        }}
        .footer {{
            text-align: center;
            font-size: 14px;
            color: gray;
            margin-top: 20px;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Title with logo
st.markdown(
    """
    <div class="title-container">
        <img src="https://assets.entrepreneur.com/content/3x2/2000/20181003130628-20180704142151-20180511063849-flipkart-logo-detail-icon.jpeg" alt="Flipkart Logo" width="100">
        <h1 class="title">Smart Vision Quality Control</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar for task selection
task = st.sidebar.selectbox(
    "Select Task",
    ["Brand Detection", "Expiry Detection", "Freshness Detection", "Count Detection"],
)

# Image input options
input_type = st.radio("Choose Input Type", ["Upload Image", "Capture from Webcam"])
image = None
save_path = "saved_images"  # Directory to save images
os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists

if input_type == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        # Save uploaded image
        uploaded_image_path = os.path.join(save_path, uploaded_file.name)
        image.save(uploaded_image_path)
        st.info(f"Image saved to: {uploaded_image_path}")

elif input_type == "Capture from Webcam":
    capture_button = st.button("Capture Image")
    if capture_button:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if ret:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            st.image(image, caption="Captured Image", use_container_width=True)
            # Save captured image
            captured_image_path = os.path.join(save_path, "captured_image.jpg")
            image.save(captured_image_path)
            st.info(f"Image saved to: {captured_image_path}")

# Save results to database & Excel
def save_results(brand=None, expiry=None, freshness=None, count=None):
    # Get current timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Create a single row of data with provided results and NA for others
    data = [{
        "Timestamp": timestamp,
        "Brand_Detection": brand if brand else "NA",
        "Expiry": expiry if expiry else "NA",
        "Freshness": freshness if freshness else "NA",
        "Count": count if count else "NA",
    }]

    # Save to SQLite
    conn = sqlite3.connect("results.db")
    cursor = conn.cursor()

    # Check if the table exists and validate schema
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='results';")
    table_exists = cursor.fetchone()

    if table_exists:
        # Validate schema
        cursor.execute("PRAGMA table_info(results);")
        columns = [info[1] for info in cursor.fetchall()]
        expected_columns = ["Timestamp", "Brand_Detection", "Expiry", "Freshness", "Count"]

        if columns != expected_columns:
            # Drop the table if schema mismatches
            cursor.execute("DROP TABLE results;")
            conn.commit()

    # Create table with the correct schema if it doesn't exist
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS results (
            Timestamp TEXT,
            Brand_Detection TEXT,
            Expiry TEXT,
            Freshness TEXT,
            Count TEXT
        )
        """
    )
    conn.commit()

    # Save the data to the database
    pd.DataFrame(data).to_sql("results", conn, if_exists="append", index=False)
    conn.close()

    # Save to Excel
    excel_file = "results.xlsx"
    if os.path.exists(excel_file):
        # Append to existing Excel file
        existing_data = pd.read_excel(excel_file)
        updated_data = pd.concat([existing_data, pd.DataFrame(data)], ignore_index=True)
    else:
        # Create new Excel file
        updated_data = pd.DataFrame(data)

    # Save the updated data to Excel
    updated_data.to_excel(excel_file, index=False)

# Perform task
if st.button(f"Run {task}"):
    if not image:
        st.error("Please provide an image first.")
    else:
        with st.spinner(f"Running {task}..."):
            if task == "Brand Detection":
                result = detect_brand(image)
                save_results(brand=result)
            elif task == "Expiry Detection":
                image_cv2 = np.array(image)
                preprocessed = preprocess_image(image_cv2)
                result = extract_expiry_date(preprocessed)
                save_results(expiry=result)
            elif task == "Freshness Detection":
                temp_path = os.path.join(save_path, "temp_image.jpg")
                image.save(temp_path)
                result = detect_freshness(temp_path)
                save_results(freshness=result)
            elif task == "Count Detection":
                image_cv2 = np.array(image)
                result = count_objects(image_cv2)
                save_results(count=result)

        st.success(f"Result: {result}")
        st.info("Results saved to database and Excel file.")

# Footer
st.markdown('<div class="footer">Powered by Smart Vision Technology</div>', unsafe_allow_html=True)
