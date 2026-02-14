import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Jute Bag Lab Analyzer")

st.header("Select Test Type")


# -------- Function: Extract inside calibration square --------
def extract_inside_square(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_square = None
    max_area = 0

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)

        if len(approx) == 4:
            area = cv2.contourArea(cnt)
            if area > 10000 and area > max_area:
                largest_square = approx
                max_area = area

    if largest_square is None:
        return None

    pts = largest_square.reshape(4, 2)

    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    side = 500

    dst = np.array([
        [0, 0],
        [side - 1, 0],
        [side - 1, side - 1],
        [0, side - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(frame, M, (side, side))

    return warped


# -------- Function: Count Ends & Picks --------
def count_density(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    vertical_projection = np.sum(thresh, axis=0)
    ends = np.sum(vertical_projection > np.mean(vertical_projection))

    horizontal_projection = np.sum(thresh, axis=1)
    picks = np.sum(horizontal_projection > np.mean(horizontal_projection))

    return ends, picks, thresh


# -------- Function: Count Bottom Stitches --------
def count_bottom_stitches(roi):
    h, w, _ = roi.shape
    bottom = roi[int(h*0.75):h, 0:w]

    gray = cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    _, thresh = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones((3,3), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    stitch_count = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 80 < area < 3000:
            stitch_count += 1

    return stitch_count, clean


# =====================================================
# 🧵 OPTION 1 — EDGE PICKS / FABRIC DENSITY
# =====================================================
st.subheader("🧵 Fabric Density Test (Ends & Picks)")

density_file = st.file_uploader(
    "Upload Image for Edge Picks / Fabric Density",
    type=["jpg", "png", "jpeg"],
    key="density"
)

if density_file is not None:
    image = Image.open(density_file)
    frame = np.array(image)

    st.image(frame, caption="Uploaded Fabric Image")

    roi = extract_inside_square(frame)

    if roi is None:
        st.error("Calibration square not detected")
    else:
        st.image(roi, caption="10cm × 10cm Area")

        ends, picks, thresh = count_density(roi)

        st.image(thresh, caption="Processed Fabric")

        st.write(f"Ends (Warp threads): {ends}")
        st.write(f"Picks (Weft threads): {picks}")


# =====================================================
# 🪡 OPTION 2 — SEAM STITCH TEST
# =====================================================
st.subheader("🪡 Seam Stitch Count Test")

stitch_file = st.file_uploader(
    "Upload Image for Seam Stitch Count",
    type=["jpg", "png", "jpeg"],
    key="stitch"
)

if stitch_file is not None:
    image = Image.open(stitch_file)
    frame = np.array(image)

    st.image(frame, caption="Uploaded Seam Image")

    roi = extract_inside_square(frame)

    if roi is None:
        st.error("Calibration square not detected")
    else:
        st.image(roi, caption="10cm × 10cm Area")

        stitches, stitch_mask = count_bottom_stitches(roi)

        st.image(stitch_mask, caption="Detected Bottom Stitches")

        st.write(f"Seam Stitches in 10cm area: {stitches}")