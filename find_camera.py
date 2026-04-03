"""Find available camera indices with different backends."""
import cv2

backends = [
    (cv2.CAP_MSMF, "MSMF"),
    (cv2.CAP_DSHOW, "DSHOW"),
    (cv2.CAP_ANY, "ANY"),
]

for backend_id, backend_name in backends:
    print(f"\n--- Backend: {backend_name} ---")
    for i in range(5):
        cap = cv2.VideoCapture(i, backend_id)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"  Camera found at index {i} ({frame.shape[1]}x{frame.shape[0]})")
            else:
                print(f"  Opened at index {i} but failed to read")
            cap.release()
    print("  (done)")
