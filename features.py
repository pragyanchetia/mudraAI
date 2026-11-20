"""Landmark normalisation + the 33 geometric features (13 angles + 20 distances)."""
import numpy as np
from scipy.spatial.distance import euclidean

ANGLE_TRIPLETS = [
    (2, 3, 4), (1, 2, 3), (6, 7, 8), (5, 6, 7), (10, 11, 12),
    (9, 10, 11), (14, 15, 16), (13, 14, 15), (18, 19, 20),
    (17, 18, 19), (5, 0, 9), (9, 0, 13), (13, 0, 17),
]
FINGERTIPS = [4, 8, 12, 16, 20]                      # thumb..pinky
TIP_TO_TIP = [(4, 8), (8, 12), (12, 16), (16, 20)]
TIP_TO_MCP = [(4, 5), (8, 5), (12, 9), (16, 13), (20, 17)]
MCP_TO_MCP = [(5, 9), (9, 13), (13, 17)]
TIP_TO_NEXT_MCP = [(8, 9), (12, 13), (16, 17)]

ANGLE_HEADERS = [f"angle_{i}" for i in range(13)]
CURL_HEADERS = ["curl_thumb", "curl_index", "curl_middle", "curl_ring", "curl_pinky"]
TIP_TIP_HEADERS = ["dist_thumb_index", "dist_index_middle", "dist_middle_ring", "dist_ring_pinky"]
TIP_MCP_HEADERS = ["dist_thumbTip_mcp", "dist_indexTip_mcp", "dist_middleTip_mcp",
                   "dist_ringTip_mcp", "dist_pinkyTip_mcp"]
MCP_MCP_HEADERS = ["dist_index_middle_mcp", "dist_middle_ring_mcp", "dist_ring_pinky_mcp"]
TIP_NEXT_MCP_HEADERS = ["dist_indexTip_middleMcp", "dist_middleTip_ringMcp", "dist_ringTip_pinkyMcp"]

FEATURE_NAMES = (ANGLE_HEADERS + CURL_HEADERS + TIP_TIP_HEADERS + TIP_MCP_HEADERS
                 + MCP_MCP_HEADERS + TIP_NEXT_MCP_HEADERS)
CSV_HEADER = ["label"] + FEATURE_NAMES
NUM_FEATURES = len(FEATURE_NAMES)  # 33


def calculate_angle(a, b, c):
    """Angle (radians) at point b formed by a-b-c."""
    a, b, c = np.asarray(a), np.asarray(b), np.asarray(c)
    ba, bc = a - b, c - b
    n1, n2 = np.linalg.norm(ba), np.linalg.norm(bc)
    if n1 == 0 or n2 == 0:
        return 0.0
    cos = np.dot(ba, bc) / (n1 * n2)
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))


def landmarks_to_array(hand_landmarks):
    """MediaPipe hand landmarks -> (21, 3) numpy array."""
    return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])


def normalize_landmarks(landmarks):
    """Translate to wrist origin and scale by max distance from the wrist.
    Returns None for a degenerate hand."""
    rel = landmarks - landmarks[0]
    max_dist = np.max(np.linalg.norm(rel, axis=1))
    if max_dist == 0:
        return None
    return rel / max_dist


def augment_landmarks(landmarks, noise_level, rng=None):
    rng = rng or np.random
    return landmarks + rng.normal(0, noise_level, size=landmarks.shape)


def extract_features(lm):
    """33 features from a normalised (21, 3) landmark array."""
    f = [calculate_angle(lm[p1], lm[p2], lm[p3]) for p1, p2, p3 in ANGLE_TRIPLETS]
    f += [float(euclidean(lm[0], lm[t])) for t in FINGERTIPS]
    for pairs in (TIP_TO_TIP, TIP_TO_MCP, MCP_TO_MCP, TIP_TO_NEXT_MCP):
        f += [float(euclidean(lm[a], lm[b])) for a, b in pairs]
    assert len(f) == NUM_FEATURES
    return f
