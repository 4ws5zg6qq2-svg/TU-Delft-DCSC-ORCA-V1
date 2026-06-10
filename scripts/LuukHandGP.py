#!/usr/bin/env python3
# uv run python scripts/LuukHandGP.py orca_core/models/v1/orcahand_right/config.yaml
import socket
import json
import math
import argparse
import time


from common import add_hand_arguments, connect_hand, create_hand, shutdown_hand
from orca_core import OrcaJointPositions

# ==============================
# CONFIG
# ==============================

UDP_IP = "0.0.0.0"
UDP_PORT = 14043

SMOOTHING_ALPHA = 0.7
CONTROL_DT = 0.02  # 50 Hz

FINGERS = {
    "thumb": 1,
    "index": 2,
    "middle": 3,
    "ring": 4,
    "pinky": 5
}

SEGMENTS = ["Metacarpal", "Proximal", "Medial", "Distal", "Tip"]

prev_cmd = {finger: [0.0, 0.0, 0.0] for finger in FINGERS}
prev_fractions = None
last_time = 0

# ==============================
# ORCA JOINT MAPPING
# ==============================

FINGER_TO_JOINTS = {
    "index":  ["index_mcp", "index_pip", "index_abd"],
    "middle": ["middle_mcp", "middle_pip", "middle_abd"],
    "ring":   ["ring_mcp", "ring_pip", "ring_abd"],
    "pinky":  ["pinky_mcp", "pinky_pip", "pinky_abd"],
    "thumb":  ["thumb_mcp", "thumb_pip", "thumb_abd", "thumb_dip"],
}

# ==============================
# MATH
# ==============================

def quat_to_euler(q):
    if not q:
        return None

    x, y, z, w = q["x"], q["y"], q["z"], q["w"]

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = math.asin(max(-1, min(1, sinp)))

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)]

# ==============================
# DATA EXTRACTION
# ==============================

def extract_hand(joint_map, side="Right"):
    hand_data = {}

    for finger_name, finger_id in FINGERS.items():
        hand_data[finger_name] = {}

        for segment in SEGMENTS:
            joint_name = f"{side}Finger{finger_id}{segment}"
            joint = joint_map.get(joint_name)

            if joint:
                rot = joint.get("rotation")
                hand_data[finger_name][segment] = {
                    "quat": rot
                }

    return hand_data

# ==============================
# PROCESSING
# ==============================

def normalize(angle, min_in=-90, max_in=90):
    angle = max(min(angle, max_in), min_in)
    return (angle - min_in) / (max_in - min_in)


def smooth(prev, current, alpha=0.7):
    return alpha * prev + (1 - alpha) * current

def get_joint(segments, name):
    return segments.get(name, {}).get("quat")

# ==============================
# QUATRATIC MATH
# ==============================

def quat_mul(a, b):
    return {
        "w": a["w"]*b["w"] - a["x"]*b["x"] - a["y"]*b["y"] - a["z"]*b["z"],
        "x": a["w"]*b["x"] + a["x"]*b["w"] + a["y"]*b["z"] - a["z"]*b["y"],
        "y": a["w"]*b["y"] - a["x"]*b["z"] + a["y"]*b["w"] + a["z"]*b["x"],
        "z": a["w"]*b["z"] + a["x"]*b["y"] - a["y"]*b["x"] + a["z"]*b["w"],
    }

def quat_inv(q):
    return {
        "w": q["w"],
        "x": -q["x"],
        "y": -q["y"],
        "z": -q["z"]
    }

def quat_normalize(q):
    mag = math.sqrt(q["w"]**2 + q["x"]**2 + q["y"]**2 + q["z"]**2)
    return {
        "w": q["w"]/mag,
        "x": q["x"]/mag,
        "y": q["y"]/mag,
        "z": q["z"]/mag,
    }

# ==============================
# DEADZONE / BUFFER
# ==============================

def deadzone(value, threshold):
    if abs(value) < threshold:
        return 0.0
    return value

# ==============================
# MAIN BUILD
# ==============================

def build_fractions(hand_data):
    global prev_cmd
    fractions = {}

    for finger, segments in hand_data.items():

        try:

            # =====================================================
            # NORMAL FINGERS
            # =====================================================

            if finger != "thumb":

                meta = segments["Metacarpal"]["quat"]
                prox = segments["Proximal"]["quat"]
                med  = segments["Medial"]["quat"]
                dist = segments["Distal"]["quat"]

                meta = quat_normalize(meta)
                prox = quat_normalize(prox)
                med  = quat_normalize(med)
                dist = quat_normalize(dist)

                # Relative rotations
                mcp_rel = quat_mul(quat_inv(meta), prox)
                pip_rel = quat_mul(quat_inv(prox), med)
                dip_rel = quat_mul(quat_inv(med), dist)

                # Euler
                mcp_e = quat_to_euler(mcp_rel)
                pip_e = quat_to_euler(pip_rel)
                dip_e = quat_to_euler(dip_rel)

                # Stable axes
                mcp = deadzone(abs(mcp_e[0]), 4.0)
                pip = abs(pip_e[0])
                dip = abs(dip_e[0])

                abd = deadzone(mcp_e[2], 2.0)

                # Normalize
                mcp_n = normalize(mcp, 0, 100)
                pip_n = normalize(pip, 0, 100)
                dip_n = normalize(dip, 0, 120)

                abd_n = 1.0 - normalize(abd, -40, 40)

                # Smoothing
                smoothed = [
                    smooth(prev_cmd[finger][0], mcp_n, SMOOTHING_ALPHA),
                    smooth(prev_cmd[finger][1], pip_n, SMOOTHING_ALPHA),
                    smooth(prev_cmd[finger][2], dip_n, SMOOTHING_ALPHA),
                ]

                prev_cmd[finger] = smoothed

                joint_names = FINGER_TO_JOINTS[finger]

                fractions[joint_names[0]] = smoothed[0]
                fractions[joint_names[1]] = smoothed[1]
                fractions[joint_names[2]] = abd_n

            # =====================================================
            # THUMB (FIXED MODEL)
            # =====================================================
               
            else:

                meta = get_joint(segments, "Metacarpal")
                prox = get_joint(segments, "Proximal")
                dist = get_joint(segments, "Distal")

                if not meta or not prox or not dist:
                    continue

                meta = quat_normalize(meta)
                prox = quat_normalize(prox)
                dist = quat_normalize(dist)

                # MCP
                mcp_rel = quat_mul(quat_inv(meta), prox)
                mcp_e = quat_to_euler(mcp_rel)

                thumb_mcp = abs(mcp_e[0])

                # Tip curl
                tip_rel = quat_mul(quat_inv(meta), dist)
                tip_e = quat_to_euler(tip_rel)

                thumb_tip = abs(tip_e[0])

                # Abduction
                abd = abs(mcp_e[2])

                # Normalize
                mcp_n = normalize(thumb_mcp, 0, 70)
                tip_n = normalize(thumb_tip, 0, 110)
                abd_n = normalize(abd, -40, 40)

                # Smooth
                smoothed = [
                    smooth(prev_cmd["thumb"][0], mcp_n, SMOOTHING_ALPHA),
                    smooth(prev_cmd["thumb"][1], tip_n, SMOOTHING_ALPHA),
                    smooth(prev_cmd["thumb"][2], abd_n, SMOOTHING_ALPHA),
                ]

                prev_cmd["thumb"] = smoothed

                # Output
                fractions["thumb_mcp"] = smoothed[0]
                fractions["thumb_pip"] = smoothed[1]
                fractions["thumb_dip"] = smoothed[1]
                fractions["thumb_abd"] = abd_n
               

        except Exception:
            continue

    return fractions

# ==============================
# ORCA POSE
# ==============================

def pose_from_fractions(hand, fractions):
    pose = dict(hand.config.neutral_position)

    for joint, fraction in fractions.items():
        if joint not in hand.config.joint_roms_dict:
            continue

        joint_min, joint_max = hand.config.joint_roms_dict[joint]
        pose[joint] = joint_min + fraction * (joint_max - joint_min)

    return OrcaJointPositions.from_dict(pose)

# ==============================
# MAIN
# ==============================

def main():
    parser = argparse.ArgumentParser(description="Stable Glove → ORCA teleoperation")
    add_hand_arguments(parser)
    args = parser.parse_args()

    # UDP (NON-BLOCKING INPUT)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    sock.setblocking(False)

    # ORCA hand
    hand = create_hand(args.config_path, use_mock=args.mock)

    latest_hand = None

    try:
        connect_hand(hand)
        hand.init_joints(force_calibrate=args.mock)

        print("Stable teleoperation started (50 Hz control loop)")

        while True:
            loop_start = time.time()

            # ==============================
            # 1. READ LATEST Glove DATA
            # ==============================
            try:
                data, _ = sock.recvfrom(65535)
                msg = json.loads(data.decode("utf-8"))

                newtons = msg.get("scene", {}).get("newtons", [])
                if newtons:
                    joints = newtons[0].get("joints", [])
                    joint_map = {j["name"]: j for j in joints}
                    latest_hand = extract_hand(joint_map, "Right")

            except BlockingIOError:
                pass
            except:
                pass

            # ==============================
            # 2. CONTROL HAND (FIXED RATE)
            # ==============================
            if latest_hand is not None:
                fractions = build_fractions(latest_hand)
                pose = pose_from_fractions(hand, fractions)

                # SAFE MOTION COMMAND
                hand.set_joint_positions(pose)

            # ==============================
            # 3. RATE LIMIT (CRITICAL)
            # ==============================
            elapsed = time.time() - loop_start
            time.sleep(max(0, CONTROL_DT - elapsed))

    finally:
        shutdown_hand(hand)
       
if __name__ == "__main__":
    raise SystemExit(main())
