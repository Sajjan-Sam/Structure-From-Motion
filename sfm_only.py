import cv2
import numpy as np
import os
from pathlib import Path
from scipy.optimize import least_squares

# ===============================================================
# STEP 1: Extract frames from video
# ===============================================================
def extract_frames(video_path, out_dir, step=2):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            fname = f"{out_dir}/{saved:05d}.jpg"
            cv2.imwrite(fname, frame)
            saved += 1
        idx += 1
    cap.release()
    print(f"[INFO] Extracted {saved} frames to {out_dir}")
    return saved


# ===============================================================
# STEP 2: Detect and match SIFT keypoints between two frames
# ===============================================================
def match_frames(img1, img2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    if des1 is None or des2 is None:
        return np.array([]), np.array([]), [], []

    # Brute-Force matcher with ratio test
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.7 * n.distance]

    if len(good) < 8:
        return np.array([]), np.array([]), [], []

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    return pts1, pts2, kp1, good


# ===============================================================
# STEP 3: Recover camera motion using Essential Matrix
# ===============================================================
def recover_motion(pts1, pts2, K):
    E, mask = cv2.findEssentialMat(pts1, pts2, K, cv2.RANSAC, 0.999, 1.0)
    if E is None:
        return np.eye(3), np.zeros((3,1)), np.zeros((len(pts1),1))
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)
    return R, t, mask_pose


# ===============================================================
# STEP 4: Triangulate 3D points from two views
# ===============================================================
def triangulate_points(K, R1, t1, R2, t2, pts1, pts2):
    pts1 = np.asarray(pts1, dtype=np.float64)
    pts2 = np.asarray(pts2, dtype=np.float64)
    if pts1.ndim != 2 or pts1.shape[1] != 2:
        pts1 = pts1.reshape(-1, 2)
    if pts2.ndim != 2 or pts2.shape[1] != 2:
        pts2 = pts2.reshape(-1, 2)
    pts1 = pts1.T
    pts2 = pts2.T

    P1 = K @ np.hstack((R1, t1))
    P2 = K @ np.hstack((R2, t2))

    pts4d = cv2.triangulatePoints(P1, P2, pts1, pts2)
    pts3d = (pts4d[:3] / pts4d[3]).T
    return pts3d


# ===============================================================
# STEP 5: Save 3D points with color into .PLY
# ===============================================================
def save_ply(filename, pts3d, colors):
    colors = np.clip(colors, 0, 255).astype(np.uint8)
    with open(filename, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts3d)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i, p in enumerate(pts3d):
            c = colors[i]
            f.write(f"{p[0]} {p[1]} {p[2]} {int(c[2])} {int(c[1])} {int(c[0])}\n")
    print(f"[DONE] Colored point cloud saved → {filename}")


# ===============================================================
# STEP 6: Main SfM Pipeline
# ===============================================================
def main(video_path):
    out_dir = "frames"
    extract_frames(video_path, out_dir, step=3)
    frames = sorted(Path(out_dir).glob("*.jpg"))
    print(f"[INFO] Found {len(frames)} frames")

    # ---- Camera Intrinsics ----
    K = np.array([[718.856, 0., 607.1928],
                  [0., 718.856, 185.2157],
                  [0., 0., 1.]], dtype=np.float64)

    all_points = []
    all_colors = []

    prev_img = cv2.imread(str(frames[0]))

    for i in range(1, len(frames), 3):
        curr_img = cv2.imread(str(frames[i]))
        pts1, pts2, kp1, good = match_frames(prev_img, curr_img)
        if len(pts1) < 8:
            print(f"[WARN] Skipping frame {i}, insufficient matches.")
            continue

        R, t, mask = recover_motion(pts1, pts2, K)
        mask = mask.ravel().astype(bool)
        pts1_inliers = pts1[mask]
        pts2_inliers = pts2[mask]

        if len(pts1_inliers) < 8:
            continue

        # Get 3D points
        pts3d = triangulate_points(K, np.eye(3), np.zeros((3,1)), R, t, pts1_inliers, pts2_inliers)

        # Get corresponding colors from the first frame
        colors = []
        for pt in pts1_inliers:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < prev_img.shape[1] and 0 <= y < prev_img.shape[0]:
                colors.append(prev_img[y, x])
            else:
                colors.append([255, 255, 255])  # default white if out-of-bounds
        colors = np.array(colors)

        all_points.append(pts3d)
        all_colors.append(colors)
        prev_img = curr_img
        print(f"[KF] Frame {i:03d} → {len(pts3d)} pts")

    # Combine all 3D points
    if len(all_points) == 0:
        print("[ERROR] No 3D points generated.")
        return

    all_points = np.vstack(all_points)
    all_colors = np.vstack(all_colors)

    save_ply("output_sfm_colored.ply", all_points, all_colors)


# ===============================================================
# ENTRY POINT
# ===============================================================
if __name__ == "__main__":
    video = "/data/Sajjan_Singh/new/video.mp4"
    main(video)
