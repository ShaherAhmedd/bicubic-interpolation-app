import streamlit as st
import cv2
import numpy as np
import math
import time

st.set_page_config(page_title="Bicubic Interpolation", layout="wide")

st.title("Bicubic Interpolation Image Scaling")

def kernel(x):
    a = -0.5
    x = abs(x)

    if x <= 1:
        return (a + 2) * (x ** 3) - (a + 3) * (x ** 2) + 1

    if x < 2:
        return a * (x ** 3) - 5 * a * (x ** 2) + 8 * a * x - 4 * a

    return 0


def manual_bicubic(image, scale_factor):
    h, w, channels = image.shape
    new_h = int(h * scale_factor)
    new_w = int(w * scale_factor)

    result = np.zeros((new_h, new_w, channels), dtype=np.uint8)

    for y in range(new_h):
        for x in range(new_w):
            old_x = x / scale_factor
            old_y = y / scale_factor

            base_x = math.floor(old_x)
            base_y = math.floor(old_y)

            for c in range(channels):
                value = 0.0

                for m in range(-1, 3):
                    for n in range(-1, 3):
                        neighbor_y = min(max(base_y + m, 0), h - 1)
                        neighbor_x = min(max(base_x + n, 0), w - 1)

                        weight_y = kernel(old_y - (base_y + m))
                        weight_x = kernel(old_x - (base_x + n))

                        value += image[neighbor_y, neighbor_x, c] * weight_x * weight_y

                result[y, x, c] = np.clip(value, 0, 255)

    return result


low_size = st.sidebar.slider("Low Resolution Size", 20, 200, 40)
scale_factor = st.sidebar.slider("Scale Factor", 2, 10, 8)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("Could not read the uploaded image.")
        st.stop()

    original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.subheader("Original Image")
    st.image(original_rgb, width=350)

    low_res = cv2.resize(
        image,
        (low_size, low_size),
        interpolation=cv2.INTER_AREA
    )

    nearest = cv2.resize(
        low_res,
        None,
        fx=scale_factor,
        fy=scale_factor,
        interpolation=cv2.INTER_NEAREST
    )

    bilinear = cv2.resize(
        low_res,
        None,
        fx=scale_factor,
        fy=scale_factor,
        interpolation=cv2.INTER_LINEAR
    )

    opencv_bicubic = cv2.resize(
        low_res,
        None,
        fx=scale_factor,
        fy=scale_factor,
        interpolation=cv2.INTER_CUBIC
    )

    start_time = time.time()
    manual_result = manual_bicubic(low_res, scale_factor)
    end_time = time.time()

    st.success(f"Manual bicubic finished in {round(end_time - start_time, 2)} seconds")

    low_res_rgb = cv2.cvtColor(low_res, cv2.COLOR_BGR2RGB)
    nearest_rgb = cv2.cvtColor(nearest, cv2.COLOR_BGR2RGB)
    bilinear_rgb = cv2.cvtColor(bilinear, cv2.COLOR_BGR2RGB)
    opencv_bicubic_rgb = cv2.cvtColor(opencv_bicubic, cv2.COLOR_BGR2RGB)
    manual_rgb = cv2.cvtColor(manual_result, cv2.COLOR_BGR2RGB)

    st.subheader("Interpolation Results")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Low Resolution Input")
        st.image(low_res_rgb)

        st.markdown("### Nearest Neighbor")
        st.image(nearest_rgb)

        st.markdown("### Bilinear")
        st.image(bilinear_rgb)

    with col2:
        st.markdown("### OpenCV Bicubic")
        st.image(opencv_bicubic_rgb)

        st.markdown("### Manual Bicubic")
        st.image(manual_rgb)