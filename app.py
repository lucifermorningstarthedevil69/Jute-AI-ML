import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Jute Fabric & Seam Stitch Analyzer")

st.write("Capture from phone camera OR upload images")

# =========================
# IMAGE INPUT OPTIONS
# =========================

fabric_img = st.camera_input("Capture Fabric (10x10 cm square visible)")
fabric_upload = st.file_uploader("OR Upload Fabric Image", type=["jpg","png","jpeg"])

st.markdown("---")

st.subheader("Upload Seam Stitch Image (Bottom Edge)")
stitches_upload = st.file_uploader("Upload Seam Stitch Image", type=["jpg","png","jpeg"])


# =========================
# FUNCTION: AUTO SQUARE DETECTION
# =========================

def detect_black_square(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect dark region (faded black still works)
    _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:  # ignore noise
            approx = cv2.approxPolyDP(cnt,
                                      0.02*cv2.arcLength(cnt, True),
                                      True)

            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                roi = frame[y:y+h, x:x+w]

                return roi, (x, y, w, h)

    return None, None


# =========================
# PROCESS FABRIC IMAGE
# =========================

frame = None

if fabric_img is not None:
    frame = np.array(Image.open(fabric_img))

elif fabric_upload is not None:
    frame = np.array(Image.open(fabric_upload))


if frame is not None:

    st.image(frame, caption="Input Fabric Image")

    roi, rect = detect_black_square(frame)

    if roi is not None:

        x, y, w, h = rect

        # draw overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (x,y), (x+w, y+h),
                      (0,255,0), 3)

        st.image(overlay,
                 caption="Detected 10x10 cm Square")

        st.subheader("Analyzing Inside Square")

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(gray, (5,5), 0)

        thresh = cv2.adaptiveThreshold(
            blur,255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            11,2
        )

        st.image(thresh, caption="Processed Fabric")

        # ===== Ends & Picks =====
        vertical_projection = np.sum(thresh, axis=0)
        ends = np.sum(vertical_projection >
                      np.mean(vertical_projection))

        horizontal_projection = np.sum(thresh, axis=1)
        picks = np.sum(horizontal_projection >
                       np.mean(horizontal_projection))

        st.subheader("Fabric Results (10cm × 10cm)")
        st.write(f"Ends (Warp threads): {ends}")
        st.write(f"Picks (Weft threads): {picks}")

    else:
        st.warning("Black square not detected")


# =========================
# PROCESS SEAM STITCH IMAGE
# =========================

if stitches_upload is not None:

    seam_img = np.array(Image.open(stitches_upload))
    st.image(seam_img, caption="Seam Stitch Image")

    gray = cv2.cvtColor(seam_img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5,5), 0)

    edges = cv2.Canny(blur, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                            80,
                            minLineLength=30,
                            maxLineGap=5)

    stitch_count = 0

    if lines is not None:
        stitch_count = len(lines)

        for line in lines:
            x1,y1,x2,y2 = line[0]
            cv2.line(seam_img, (x1,y1), (x2,y2),
                     (0,255,0), 2)

    st.image(seam_img, caption="Detected Stitches")

    st.subheader("Seam Stitch Count (Bottom)")
    st.write(f"Number of Stitches: {stitch_count}")