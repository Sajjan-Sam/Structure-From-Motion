Only Three Files Contain the SFM Mainly(sfm_only.py, vid.mp4, output_sfm_cloud.ply) and all other files are beliong to COLMAP(better version but based on SFM) 
<img width="465" height="579" alt="image" src="https://github.com/user-attachments/assets/f40db1a8-9575-401d-a76a-14d552b56730" />

<img width="792" height="478" alt="image" src="https://github.com/user-attachments/assets/d621a3da-64c6-4fad-b006-ecede6487a0d" />

<img width="523" height="521" alt="image" src="https://github.com/user-attachments/assets/0d8b34df-1071-48f1-b45b-139a6a86c3aa" />


# 🧠 Structure from Motion (SfM) — Complete Implementation and COLMAP Comparison

This repository presents my exploration of **Structure from Motion (SfM)** — a technique in **Computer Vision** used to estimate the 3D structure of a scene from a set of 2D images or a video sequence.

The project is divided into **two major parts**:
1. **Custom SfM Implementation (from Scratch)** → `sfm_only.py`  
   - Built using **OpenCV**, **NumPy**, and **SciPy**.  
   - Produces a colored point cloud file `output_sfm_colored.ply`.
2. **COLMAP-based Reconstruction** (for comparison)  
   - Demonstrates how professional photogrammetry software performs the same pipeline automatically.  
   - Produces **dense point clouds** and **meshed 3D models**.

---

## 📘 Overview

**Structure from Motion (SfM)** is a computer vision pipeline that takes multiple overlapping images (or frames from a video) and reconstructs:
- The **3D geometry of the scene**  
- The **camera motion (position and orientation)**  

Essentially, SfM transforms a regular phone video into a **3D point cloud** that can be visualized using software like **MeshLab** or **CloudCompare**.

---

## 🧩 Project Components

| Component | File | Description |
|------------|------|-------------|
| 🧠 **My Custom SfM** | `sfm_only.py` | Python code implementing SfM from scratch (feature extraction, motion estimation, triangulation, bundle adjustment, and colored 3D reconstruction). |
| 🌈 **Output (My SfM)** | `output_sfm_colored.ply` | Final colored 3D point cloud generated using my SfM pipeline. |
| 🧱 **COLMAP Reconstruction** | Other `.py` and `.ply` files | Dense reconstruction and mesh generation using COLMAP commands. |
| 📸 **Artifacts Folder** | `artifacts/` | Contains reconstruction outputs (sparse cloud, dense model, mesh). |

---
