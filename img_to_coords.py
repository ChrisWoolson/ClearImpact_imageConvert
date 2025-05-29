import cv2
import numpy as np
from pyproj import Transformer
import matplotlib.pyplot as plt

# ─── USER CONFIG ───────────────────────────────────────────────────────────────

IMAGE_FILE = 'pn.png'

image_pts = np.array([
    [1485, 2854],
    [676, 2079],
    
], dtype=float)

latlon_pts = [
(34.417214192386055, -119.84538983932903),
(34.417009055595564, -119.84544397925174)
]

test_pixels = np.array([
    [1188, 1842]
], dtype=float)

CRS_LATLON  = "epsg:4326"
CRS_PLANAR  = "epsg:32611"

flip_image_horizontally = False
flip_image_vertically   = False
flip_planar_x = True
flip_planar_y = True

K = np.array([
    [1456, 0, 1048],
    [0, 1456, 540],
    [0, 0, 1]
], dtype=np.float64)  # camera intrinsic matrix

# ─── END CONFIG ────────────────────────────────────────────────────────────────

import cv2, numpy as np


def drop_point_to_ground(image_pts, world_pts, K, lp_px, dist_coeffs=None):
    """
    image_pts : (N,2) float64  ground‐plane reference pixels
    world_pts : (N,2) float64  their (X,Y) in meters at Z=0
    K         : (3,3)   float64 camera intrinsics
    lp_px     : (2,)    float    license‐plate center pixel (u,v)
    dist_coeffs: (5,1)  float    lens dist· (None→zeros)
    """
    # 1) prep
    if dist_coeffs is None:
        dist_coeffs = np.zeros((5,1), dtype=np.float64)
    image_pts = np.ascontiguousarray(image_pts, dtype=np.float64)
    world_pts = np.ascontiguousarray(world_pts, dtype=np.float64)
    K         = np.ascontiguousarray(K,         dtype=np.float64)

    # 2) solvePnP to get camera pose wrt ground plane
    obj3d = np.hstack([world_pts, np.zeros((world_pts.shape[0],1))])
    ok, rvec, tvec = cv2.solvePnP(obj3d, image_pts, K, dist_coeffs,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("solvePnP failed—check your ground refs")
    R_cam2world, _ = cv2.Rodrigues(rvec)
    tvec = tvec.ravel()

    # 3) on first call, draw ROI to measure both width & height in px
    if not hasattr(drop_point_to_ground, "_plate_wh_px"):
        x,y,w,h = cv2.selectROI("Draw plate box", img_rgb, False, False)
        cv2.destroyWindow("Draw plate box")
        drop_point_to_ground._plate_wh_px = (float(w), float(h))
    plate_w_px, plate_h_px = drop_point_to_ground._plate_wh_px

    # 4) compute depth estimates from known plate W=0.305m, H=0.1524m
    fx, fy = K[0,0], K[1,1]
    Zx = fx * 0.305  / plate_w_px
    Zy = fy * 0.1524 / plate_h_px
    Zc = 0.5 * (Zx + Zy)

    # 5) back‐project plate center into camera coords
    u,v = lp_px
    p_norm = np.linalg.inv(K).dot(np.array([u, v, 1.0], dtype=np.float64))
    P_cam  = p_norm * Zc   # (Xc, Yc, Zc)

    # 6) to world coords: Xw = Rᵀ (Xc - t)
    P_world = R_cam2world.T.dot(P_cam - tvec)  # (Xw, Yw, Zw)

    # 7) drop vertically to Z=0
    Xg, Yg = P_world[0], P_world[1]
    P_ground_w = np.array([Xg, Yg, 0.0], dtype=np.float64)

    # 8) re‐project that ground point:
    P_cam_g = R_cam2world.dot(P_ground_w) + tvec  # (Xc',Yc',Zc')
    xg = fx * (P_cam_g[0]/P_cam_g[2]) + K[0,2]
    yg = fy * (P_cam_g[1]/P_cam_g[2]) + K[1,2]
    print(f"({xg:.2f}, {yg:.2f})")
    return float(xg), float(yg)


#Load Image
img = cv2.imread(IMAGE_FILE)
if img is None:
    raise FileNotFoundError(f"Couldn’t load '{IMAGE_FILE}'")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w = img_rgb.shape[:2]

if flip_image_horizontally:
    img_rgb = np.flip(img_rgb, axis=1)
    image_pts[:,0] = w - image_pts[:,0]
    test_pixels[:,0] = w - test_pixels[:,0]
if flip_image_vertically:
    img_rgb = np.flip(img_rgb, axis=0)
    image_pts[:,1] = h - image_pts[:,1]
    test_pixels[:,1] = h - test_pixels[:,1]

# 2. Project lat/lon → planar (meters)
to_planar = Transformer.from_crs(CRS_LATLON, CRS_PLANAR, always_xy=True)
lons = [pt[1] for pt in latlon_pts]
lats = [pt[0] for pt in latlon_pts]
planar_pts = np.vstack(to_planar.transform(lons, lats)).T  # shape (N,2)

# 3. Solve homography
H, status = cv2.findHomography(image_pts, planar_pts, method=cv2.RANSAC)

# 4. Map all test pixels
ones      = np.ones((len(test_pixels),1))
hom_test  = np.hstack([test_pixels, ones])          # (M,3)
mapped_hom = (H @ hom_test.T).T                     # (M,3)
mapped_pts = mapped_hom[:,:2] / mapped_hom[:,2:3]   # (M,2)

dropped_points = []
for i in range(len(test_pixels)):
    ptx, pty = drop_point_to_ground(image_pts, planar_pts, K, test_pixels[i])
    dropped_points.append((ptx, pty))

# 5. Print planar results
print("=== Planar Mappings ===")
for src, dst in zip(test_pixels, mapped_pts):
    print(f"Pixel {src} → Easting/Northing (m): {dst}")

# 6. Inverse‐project back to lat/lon
from_planar = Transformer.from_crs(CRS_PLANAR, CRS_LATLON, always_xy=True)
# unpack X and Y arrays
xs = mapped_pts[:,0]
ys = mapped_pts[:,1]
lons_out, lats_out = from_planar.transform(xs, ys)

print("\n=== Back to Latitude/Longitude ===")
for (u,v), φ, λ in zip(test_pixels, lats_out, lons_out):
    print(f"Pixel {(u,v)} → Latitude: {φ:.8f},  Longitude: {λ:.8f}")

# PLOT RESULTS
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,6))

ax1.imshow(img_rgb)
ax1.scatter(image_pts[:,0], image_pts[:,1],
            s=60, facecolors='none', edgecolors='r', label='Refs')
ax1.scatter(test_pixels[:,0], test_pixels[:,1],
            s=60, marker='x', color='green', label='Test pts')
xs, ys = zip(*dropped_points)
ax1.scatter(xs, ys, s=60, marker='o', color='black', label='Dropped pts')
ax1.set_title("Input Image")
ax1.legend()
ax1.axis('off')

ax2.scatter(planar_pts[:,0], planar_pts[:,1],
            s=60, facecolors='none', edgecolors='r', label='Refs')

ax2.scatter(mapped_pts[:,0], mapped_pts[:,1],
            s=60, marker='x', color='green', label='Test pts')

#ax2.scatter(xs, ys, s=60, marker='o', color='black', label='Dropped pts')
ax2.set_aspect('equal', 'box')
ax2.set_title("Planar Coordinates (meters)")
ax2.set_xlabel("Easting (m)")
ax2.set_ylabel("Northing (m)")

if flip_planar_x:
    ax2.invert_xaxis()
if flip_planar_y:
    ax2.invert_yaxis()

ax2.legend()
plt.tight_layout()
plt.show()
