import time
import math
import argparse
import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs

# Importeert de MediaPipe Tasks API voor live hand-landmarkdetectie.
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from common import add_hand_arguments, connect_hand, create_hand, shutdown_hand
from orca_core import OrcaJointPositions

# Pad naar het MediaPipe-model dat de 21 hand-landmarks detecteert.
MODEL_PATH = "hand_landmarker.task"

# Het meest recente detectieresultaat van MediaPipe.
latest_result = None

# Verbindingsparen tussen MediaPipe-landmarks, gebruikt om het handskelet op het scherm te tekenen.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

# Per gewricht kan de fractie worden omgekeerd als de robotische hand fysiek de verkeerde kant op beweegt.
INVERT_FRACTIONS = {
    "thumb_abd": False,
    "thumb_mcp": False,
    "thumb_pip": False,
    "thumb_dip": False,

    "index_abd": False,
    "index_mcp": False,
    "index_pip": False,

    "middle_abd": False,
    "middle_mcp": False,
    "middle_pip": False,

    "ring_abd": False,
    "ring_mcp": False,
    "ring_pip": False,

    "pinky_abd": False,
    "pinky_mcp": False,
    "pinky_pip": False,

    "wrist": False,
}

# Houdt de vorige gewrichtsfracties bij zodat nieuwe waarden kunnen worden afgevlakt.
prev_fractions = {}


def result_callback(result, output_image, timestamp_ms):
    # Slaat het nieuwste MediaPipe-resultaat op
    global latest_result
    latest_result = result


def vec(a, b):
    return [b[i] - a[i] for i in range(len(a))]


def dot(a, b):
    return sum(a[i] * b[i] for i in range(len(a)))


def norm(v):
    return math.sqrt(sum(x * x for x in v))


def normalize(v):
    n = norm(v)
    if n < 1e-9:
        return [0.0 for _ in v]
    return [x / n for x in v]


def cross(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def map_range(x, in_min, in_max, out_min, out_max):
    # Zet een waarde uit een invoerbereik om naar een waarde in een uitvoerbereik.
    x = clamp(x, in_min, in_max)
    if abs(in_max - in_min) < 1e-9:
        return out_min
    t = (x - in_min) / (in_max - in_min)
    return out_min + t * (out_max - out_min)


def angle_between(v1, v2):
    # Berekent de hoek in graden tussen twee vectoren.
    n1 = norm(v1)
    n2 = norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    c = dot(v1, v2) / (n1 * n2)
    c = clamp(c, -1.0, 1.0)
    return math.degrees(math.acos(c))


def angle_3pts(a, b, c):
    # Berekent de hoek a-b-c, waarbij b het draaipunt is.
    ba = [a[i] - b[i] for i in range(len(a))]
    bc = [c[i] - b[i] for i in range(len(c))]
    return angle_between(ba, bc)


def signed_angle_on_plane(v1, v2, plane_normal):
    # Projecteert twee vectoren op een vlak en berekent daarna de getekende hoek tussen die projecties.
    n = normalize(plane_normal)

    # Haalt de component loodrecht op het vlak uit de vectoren zodat alleen de vlakcomponent overblijft.
    v1p = [v1[i] - dot(v1, n) * n[i] for i in range(3)]
    v2p = [v2[i] - dot(v2, n) * n[i] for i in range(3)]

    # Bij ongeldige of bijna nulvectoren is er geen betrouwbare hoek te berekenen.
    n1 = norm(v1p)
    n2 = norm(v2p)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0

    # Normaliseert de geprojecteerde vectoren voordat de hoek wordt bepaald.
    v1p = [x / n1 for x in v1p]
    v2p = [x / n2 for x in v2p]

    # De grootte van de hoek komt uit angle_between; het teken komt uit de richting van het kruisproduct.
    unsigned = angle_between(v1p, v2p)
    c = cross(v1p, v2p)
    sign = 1.0 if dot(c, n) >= 0 else -1.0
    return unsigned * sign


def smooth_fractions(new_vals, prev_vals, alpha=0.25):
    # Past exponentiële smoothing toe zodat de robotische hand niet schokkerig beweegt.
    out = {}
    for k, v in new_vals.items():
        prev = prev_vals.get(k, v)
        out[k] = prev + alpha * (v - prev)
    return out

#Deze smoothing filter heeft een beter controle maar na testen vonden wij de oude die veel minder effect heeft toch beter.
"""""
def smooth_fractions(new_vals, prev_vals, alpha=0.35, deadband=0.015, max_step=0.1):
    out = {}

    for k, v in new_vals.items():
        prev = prev_vals.get(k, v)

        # 1) negeer hele kleine veranderingen
        if abs(v - prev) < deadband:
            target = prev
        else:
            target = v

        # 2) lichte exponential smoothing
        smoothed = prev + alpha * (target - prev)

        # 3) begrens hoeveel de waarde per frame mag veranderen
        delta = smoothed - prev
        if delta > max_step:
            smoothed = prev + max_step
        elif delta < -max_step:
            smoothed = prev - max_step

        out[k] = clamp(smoothed, 0.0, 1.0)

    return out
"""""


def apply_fraction_inversions(fractions):
    # Past per gewricht eventueel een omkering toe en houdt elke fractie tussen 0 en 1.
    out = {}
    for joint, value in fractions.items():
        value = clamp(value, 0.0, 1.0)
        if INVERT_FRACTIONS.get(joint, False):
            value = 1.0 - value
        out[joint] = clamp(value, 0.0, 1.0)
    return out


def pose_from_fractions(hand, fractions: dict[str, float]) -> OrcaJointPositions:
    # Start vanuit de neutrale handpositie zodat niet-aangestuurde gewrichten veilig neutraal blijven.
    pose = dict(hand.config.neutral_position)

    # Zet elke fractie om naar een echte gewrichtspositie binnen het bereik van dat gewricht.
    for joint, fraction in fractions.items():
        if joint not in hand.config.joint_roms_dict:
            continue

        joint_min, joint_max = hand.config.joint_roms_dict[joint]
        fraction = clamp(fraction, 0.0, 1.0)
        pose[joint] = joint_min + fraction * (joint_max - joint_min)

    return OrcaJointPositions.from_dict(pose)


def neutral_fractions_from_config(hand):
    # Berekent voor elk gewricht welke fractie overeenkomt met de neutrale positie uit de configuratie.
    fractions = {}

    for joint, neutral in hand.config.neutral_position.items():
        if joint not in hand.config.joint_roms_dict:
            continue

        joint_min, joint_max = hand.config.joint_roms_dict[joint]
        if abs(joint_max - joint_min) < 1e-9:
            fractions[joint] = 0.0
        else:
            fractions[joint] = clamp(
                (neutral - joint_min) / (joint_max - joint_min),
                0.0,
                1.0
            )

    return fractions


def ordered_joint_names(available_joints):
    # Bepaalt een vaste volgorde voor debug-uitvoer, zodat de waarden op het scherm steeds hetzelfde staan.
    preferred_order = [
        "thumb_abd", "thumb_mcp", "thumb_pip", "thumb_dip",
        "index_abd", "index_mcp", "index_pip",
        "middle_abd", "middle_mcp", "middle_pip",
        "ring_abd", "ring_mcp", "ring_pip",
        "pinky_abd", "pinky_mcp", "pinky_pip",
        "wrist",
    ]
    return [j for j in preferred_order if j in available_joints]


def get_median_depth(depth_frame, x, y, image_w, image_h, radius=2):
    # Leest dieptewaarden rond één pixel en gebruikt de mediaan om ruis te verminderen.
    vals = []

    for yy in range(y - radius, y + radius + 1):
        for xx in range(x - radius, x + radius + 1):
            if 0 <= xx < image_w and 0 <= yy < image_h:
                d = depth_frame.get_distance(xx, yy)
                if d > 0.0:
                    vals.append(d)

    # Als er geen geldige diepte is, wordt 0 teruggegeven als foutwaarde.
    if not vals:
        return 0.0

    vals.sort()
    return vals[len(vals) // 2]


def get_landmark_3d_points(hand_landmarks, depth_frame, color_intrinsics, image_w, image_h):
    # Zet MediaPipe-landmarks om naar metrische 3D-punten met behulp van de uitgelijnde RealSense-diepte.
    # De functie geeft een lijst van 21 punten terug; elk punt is [X, Y, Z] of None bij ongeldige diepte.
    pts_3d = []

    for lm in hand_landmarks:
        # MediaPipe geeft x en y genormaliseerd tussen 0 en 1; dit wordt omgezet naar pixelcoördinaten.
        x = int(lm.x * image_w)
        y = int(lm.y * image_h)

        # Zorgt dat de pixel altijd binnen de afbeelding valt.
        x = max(0, min(image_w - 1, x))
        y = max(0, min(image_h - 1, y))

        # Haalt een stabiele dieptewaarde op rond de landmarkpixel.
        depth_m = get_median_depth(depth_frame, x, y, image_w, image_h, radius=2)
        if depth_m <= 0.0:
            pts_3d.append(None)
            continue

        # Deprojecteert de 2D-pixel plus diepte naar een echt 3D-punt.
        point_3d = rs.rs2_deproject_pixel_to_point(color_intrinsics, [x, y], depth_m)
        pts_3d.append(point_3d)

    return pts_3d


def get_depth_point_or_none(pts3d, idx):
    if pts3d is None:
        return None
    if idx < 0 or idx >= len(pts3d):
        return None
    return pts3d[idx]


def depth_angle_flex_or_fallback(pts3d, a, b, c, fallback):
    # Probeert een buighoek uit 3D-dieptepunten te berekenen; valt terug op de RGB/MediaPipe-waarde bij ontbrekende diepte.
    pa = get_depth_point_or_none(pts3d, a)
    pb = get_depth_point_or_none(pts3d, b)
    pc = get_depth_point_or_none(pts3d, c)
    if pa is None or pb is None or pc is None:
        return fallback
    return 180.0 - angle_3pts(pa, pb, pc)


def extract_fraction_commands(
    hand_landmarks,
    available_joints,
    depth_frame=None,
    color_intrinsics=None,
    image_w=None,
    image_h=None
):
    # Zet de ruwe MediaPipe-landmarks om naar robotcommando's als fracties tussen 0 en 1.
    pts = [(lm.x, lm.y, lm.z) for lm in hand_landmarks]

    # Landmark 0 is de pols.
    wrist = pts[0]

    # Punten van de duim volgens de MediaPipe-indexering.
    thumb_cmc_pt = pts[1]
    thumb_mcp_pt = pts[2]
    thumb_ip_pt = pts[3]
    thumb_tip = pts[4]

    # Punten van de wijsvinger.
    index_mcp_pt = pts[5]
    index_pip_pt = pts[6]
    index_dip_pt = pts[7]

    # Punten van de middelvinger.
    middle_mcp_pt = pts[9]
    middle_pip_pt = pts[10]
    middle_dip_pt = pts[11]

    # Punten van de ringvinger.
    ring_mcp_pt = pts[13]
    ring_pip_pt = pts[14]
    ring_dip_pt = pts[15]

    # Punten van de pink.
    pinky_mcp_pt = pts[17]
    pinky_pip_pt = pts[18]
    pinky_dip_pt = pts[19]

    # Maakt een palmvlak in beeldcoördinaten als fallback wanneer dieptegegevens ontbreken.
    palm_x = vec(wrist, index_mcp_pt)
    palm_y = vec(wrist, pinky_mcp_pt)
    palm_normal = cross(palm_x, palm_y)

    # Richtingsvector van de duimbasis.
    thumb_base_dir = vec(thumb_cmc_pt, thumb_mcp_pt)

    # Richtingsvectoren van de vingers vanaf het MCP-gewricht naar het PIP-gewricht.
    index_dir = vec(index_mcp_pt, index_pip_pt)
    middle_dir = vec(middle_mcp_pt, middle_pip_pt)
    ring_dir = vec(ring_mcp_pt, ring_pip_pt)
    pinky_dir = vec(pinky_mcp_pt, pinky_pip_pt)

    # Berekent buiging van duim en vingers op basis van de 2D/MediaPipe-landmarks.
    rgb_thumb_mcp_flex = 180.0 - angle_3pts(thumb_cmc_pt, thumb_mcp_pt, thumb_ip_pt)
    rgb_thumb_distal_flex = 180.0 - angle_3pts(thumb_mcp_pt, thumb_ip_pt, thumb_tip)

    rgb_index_mcp_flex = 180.0 - angle_3pts(wrist, index_mcp_pt, index_pip_pt)
    rgb_index_pip_flex = 180.0 - angle_3pts(index_mcp_pt, index_pip_pt, index_dip_pt)

    rgb_middle_mcp_flex = 180.0 - angle_3pts(wrist, middle_mcp_pt, middle_pip_pt)
    rgb_middle_pip_flex = 180.0 - angle_3pts(middle_mcp_pt, middle_pip_pt, middle_dip_pt)

    rgb_ring_mcp_flex = 180.0 - angle_3pts(wrist, ring_mcp_pt, ring_pip_pt)
    rgb_ring_pip_flex = 180.0 - angle_3pts(ring_mcp_pt, ring_pip_pt, ring_dip_pt)

    rgb_pinky_mcp_flex = 180.0 - angle_3pts(wrist, pinky_mcp_pt, pinky_pip_pt)
    rgb_pinky_pip_flex = 180.0 - angle_3pts(pinky_mcp_pt, pinky_pip_pt, pinky_dip_pt)

    # Berekent abductie: zijwaartse spreiding van duim en vingers ten opzichte van het palmvlak.
    rgb_thumb_abd = signed_angle_on_plane(palm_x, thumb_base_dir, palm_normal)
    rgb_index_abd = signed_angle_on_plane(middle_dir, index_dir, palm_normal)
    rgb_middle_abd = 0.0
    rgb_ring_abd = signed_angle_on_plane(middle_dir, ring_dir, palm_normal)
    rgb_pinky_abd = signed_angle_on_plane(ring_dir, pinky_dir, palm_normal)

    # Maakt optioneel 3D-punten aan wanneer dieptebeeld en camera-intrinsics beschikbaar zijn.
    pts3d = None
    if depth_frame is not None and color_intrinsics is not None and image_w is not None and image_h is not None:
        pts3d = get_landmark_3d_points(
            hand_landmarks, depth_frame, color_intrinsics, image_w, image_h
        )

    # De duim gebruikt bij voorkeur diepte-informatie, omdat de duim vaak lastig correct uit alleen 2D te schatten is.
    human_thumb_mcp_flex = depth_angle_flex_or_fallback(pts3d, 1, 2, 3, rgb_thumb_mcp_flex)
    human_thumb_distal_flex = depth_angle_flex_or_fallback(pts3d, 2, 3, 4, rgb_thumb_distal_flex)

    # De MCP-buiging van de vier vingers blijft hier op de RGB/MediaPipe-berekening staan.
    human_index_mcp_flex = rgb_index_mcp_flex
    human_middle_mcp_flex = rgb_middle_mcp_flex
    human_ring_mcp_flex = rgb_ring_mcp_flex
    human_pinky_mcp_flex = rgb_pinky_mcp_flex

    # De PIP-buiging van de vier vingers blijft ook gebaseerd op RGB/MediaPipe.
    human_index_pip_flex = rgb_index_pip_flex
    human_middle_pip_flex = rgb_middle_pip_flex
    human_ring_pip_flex = rgb_ring_pip_flex
    human_pinky_pip_flex = rgb_pinky_pip_flex

    # Haalt 3D-punten op die nodig zijn om de duimabductie met diepte te berekenen.
    p0 = get_depth_point_or_none(pts3d, 0)
    p1 = get_depth_point_or_none(pts3d, 1)
    p2 = get_depth_point_or_none(pts3d, 2)
    p5 = get_depth_point_or_none(pts3d, 5)
    p17 = get_depth_point_or_none(pts3d, 17)

    # Als de nodige 3D-punten bestaan, wordt het palmvlak in 3D berekend voor een betere duimabductie.
    if p0 is not None and p1 is not None and p2 is not None and p5 is not None and p17 is not None:
        palm_x_3d = vec(p0, p5)
        palm_y_3d = vec(p0, p17)
        palm_normal_3d = cross(palm_x_3d, palm_y_3d)
        thumb_base_dir_3d = vec(p1, p2)
        human_thumb_abd = signed_angle_on_plane(palm_x_3d, thumb_base_dir_3d, palm_normal_3d)
    else:
        human_thumb_abd = rgb_thumb_abd

    # De abductie van de vingers blijft op de RGB/MediaPipe-berekening staan.
    human_index_abd = rgb_index_abd
    human_middle_abd = rgb_middle_abd
    human_ring_abd = rgb_ring_abd
    human_pinky_abd = rgb_pinky_abd

    # Dictionary waarin alle doelwaarden voor de robotgewrichten als fracties worden opgeslagen.
    fractions = {}

    mcp_gain = 1.7

    # Bij ongeveer 0 graden staat een vinger recht; richting 90 graden is de vinger sterk gebogen.
    # Tweaking van MediaPipe consistente errors, kan hier bepaalde posities optimaliseren:
    if human_index_mcp_flex > 40 and human_middle_mcp_flex > 40 and human_ring_mcp_flex > 40 and human_pinky_mcp_flex > 40:
        pip_gain_Index = 1.3
        pip_gain_middle = 1.3
        pip_gain_ring = 1.3
        pip_gain_pinky = 1.3
        vuist = True
    else:
        vuist = False
    if not vuist:
        if human_index_mcp_flex < 32:
            pip_gain_Index = 2.5
        else:
            pip_gain_Index = 0.65
        if human_middle_mcp_flex < 32:
            pip_gain_middle = 2.5
        else:
            pip_gain_middle = 0.65
        if human_ring_mcp_flex < 32:
            pip_gain_ring = 2
        else:
            pip_gain_ring = 0.65
        if human_pinky_mcp_flex < 32:
            pip_gain_pinky = 1.6
        else:
            pip_gain_pinky = 1

    # Past de abductieversterking aan per vinger, afhankelijk van hoe ver de vinger gebogen is.
    if human_index_mcp_flex > 40:
        abd_gain_Index = 1
    else:
        abd_gain_Index = 1.5
    if human_middle_mcp_flex < 32:
        abd_gain_middle = 1
    else:
        abd_gain_middle = 1.5
    if human_ring_mcp_flex > 40:
        abd_gain_ring = 1
    else:
        abd_gain_ring = 1.5
    if human_pinky_mcp_flex > 40:
        abd_gain_pinky = 1
    else:
        abd_gain_pinky = 2.3

    # Zet alles om naar robotfracties:
    if "thumb_abd" in available_joints:
        fractions["thumb_abd"] = map_range(
            human_thumb_abd,
            -15.0, 15.0, 0.0, 1.0
        )
    if "thumb_mcp" in available_joints:
        fractions["thumb_mcp"] = map_range(
            human_thumb_mcp_flex,
            0.0, 70.0, 0.0, 1.0
        )
    if "thumb_pip" in available_joints:
        fractions["thumb_pip"] = map_range(
            human_thumb_distal_flex,
            0.0, 90.0, 0.0, 1.0
        )
    if "thumb_dip" in available_joints:
        fractions["thumb_dip"] = clamp(
            0.45 * fractions["thumb_pip"],
            0.0, 1.0
        )
    if "index_abd" in available_joints:
        fractions["index_abd"] = map_range(
            human_index_abd * abd_gain_Index + 20,
            -30.0, 30.0, 0.0, 1.0
        )
    if "index_mcp" in available_joints:
        fractions["index_mcp"] = map_range(
            human_index_mcp_flex * mcp_gain,
            0.0, 90.0, 0.0, 1.0
        )
    if "index_pip" in available_joints:
        fractions["index_pip"] = map_range(
            human_index_pip_flex * pip_gain_Index,
            0.0, 110.0, 0.0, 1.0
        )
    if "middle_abd" in available_joints:
        fractions["middle_abd"] = map_range(
            human_middle_abd * abd_gain_middle,
            -20.0, 20.0, 0.0, 1.0
        )
    if "middle_mcp" in available_joints:
        fractions["middle_mcp"] = map_range(
            human_middle_mcp_flex * 2,
            0.0, 90.0, 0.0, 1.0
        )
    if "middle_pip" in available_joints:
        fractions["middle_pip"] = map_range(
            human_middle_pip_flex * pip_gain_middle,
            0.0, 110.0, 0.0, 1.0
        )
    if "ring_abd" in available_joints:
        fractions["ring_abd"] = map_range(
            human_ring_abd * abd_gain_ring - 20,
            -30.0, 30.0, 0.0, 1.0
        )
    if "ring_mcp" in available_joints:
        fractions["ring_mcp"] = map_range(
            human_ring_mcp_flex * mcp_gain,
            0.0, 90.0, 0.0, 1.0
        )
    if "ring_pip" in available_joints:
        fractions["ring_pip"] = map_range(
            human_ring_pip_flex * pip_gain_ring,
            0.0, 110.0, 0.0, 1.0
        )
    if "pinky_abd" in available_joints:
        fractions["pinky_abd"] = map_range(
            human_pinky_abd * abd_gain_pinky - 30,
            -30.0, 30.0, 0.0, 1.0
        )
    if "pinky_mcp" in available_joints:
        fractions["pinky_mcp"] = map_range(
            human_pinky_mcp_flex,
            0.0, 95.0, 0.0, 1.0
        )
    if "pinky_pip" in available_joints:
        fractions["pinky_pip"] = map_range(
            human_pinky_pip_flex * pip_gain_pinky,
            0.0, 110.0, 0.0, 1.0
        )

    if "wrist" in available_joints:
        fractions["wrist"] = 0.5

    return apply_fraction_inversions(fractions)


def draw_hand_and_control(
    frame,
    result,
    hand,
    depth_frame=None,
    color_intrinsics=None,
    send_to_hand=True
):
    global prev_fractions

    h, w, _ = frame.shape
    available_joints = set(hand.config.joint_roms_dict.keys())
    left_hand_found = False
    current_fractions = None
    depth_valid_points = 0

    # Verwerk alleen als MediaPipe al een detectieresultaat heeft geleverd.
    if result is not None:
        for hand_idx, hand_landmarks in enumerate(result.hand_landmarks):
            # Slaat onvolledige handedness-informatie over.
            if hand_idx >= len(result.handedness) or len(result.handedness[hand_idx]) == 0:
                continue

            # Label geeft aan of MediaPipe de hand als links of rechts ziet; score is de zekerheid.
            label = result.handedness[hand_idx][0].category_name
            score = result.handedness[hand_idx][0].score

            # Door de horizontale beeldspiegeling wordt hier alleen de gedetecteerde rechterhand gebruikt.
            if label != "Right":
                continue

            left_hand_found = True

            # Zet landmarks om naar schermpunten en tekent groene punten op elk gewricht.
            points = []
            for lm in hand_landmarks:
                x = w - 1 - int(lm.x * w)
                y = int(lm.y * h)
                points.append((x, y))
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            # Tekent blauwe lijnen tussen de landmarkpunten zodat het handskelet zichtbaar is.
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, points[a], points[b], (255, 0, 0), 2)

            # Telt hoeveel landmarks een geldige dieptewaarde hebben.
            if depth_frame is not None and color_intrinsics is not None:
                pts3d = get_landmark_3d_points(hand_landmarks, depth_frame, color_intrinsics, w, h)
                depth_valid_points = sum(p is not None for p in pts3d)

            current_fractions = extract_fraction_commands(
                hand_landmarks,
                available_joints,
                depth_frame=depth_frame,
                color_intrinsics=color_intrinsics,
                image_w=w,
                image_h=h
            )

            # Vlakt de fracties af en bewaart ze voor het volgende frame.
            current_fractions = smooth_fractions(current_fractions, prev_fractions, alpha=0.25)
            prev_fractions.update(current_fractions)

            # Schrijft het handlabel en de detectiezekerheid naast de pols.
            x0, y0 = points[0]
            cv2.putText(
                frame,
                f"{label} {score:.2f}",
                (x0, max(30, y0 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )
            break

    # Als er geen bruikbare hand wordt gevonden, gaat de robot veilig naar de neutrale houding.
    if not left_hand_found:
        current_fractions = neutral_fractions_from_config(hand)
        current_fractions = smooth_fractions(current_fractions, prev_fractions, alpha=0.20)
        prev_fractions.update(current_fractions)

        cv2.putText(
            frame,
            "Geen rechterhand gedetecteerd - neutrale positie wordt gestuurd",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

    # Maakt debugregels waarin telkens drie gewrichten per regel worden weergegeven.
    debug_order = ordered_joint_names(available_joints)
    debug_lines = []
    line = []

    for idx, joint in enumerate(debug_order):
        line.append(f"{joint}:{current_fractions[joint]:.2f}")
        if len(line) == 3 or idx == len(debug_order) - 1:
            debug_lines.append("  ".join(line))
            line = []

    # Zet de debugwaarden op het videobeeld.
    for i, txt in enumerate(debug_lines[:6]):
        cv2.putText(
            frame,
            txt,
            (10, 85 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )

    # Toont hoeveel van de 21 landmarks geldige diepte-informatie hadden.
    cv2.putText(
        frame,
        f"Geldige diepte-landmarks: {depth_valid_points}/21",
        (10, 220),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 0),
        1
    )

    if send_to_hand:
        # Stuurt de berekende houding naar de robotische hand
        pose = pose_from_fractions(hand, current_fractions)
        hand.set_joint_positions(pose)

    # Print de fracties als CSV-achtige debugregel in de terminal.
    print(",".join(f"{current_fractions[j]:.2f}" for j in debug_order))

    return frame


def main():
    # Maakt de command-line parser aan voor modelpad, schermschaal, camerainstellingen en robotopties.
    parser = argparse.ArgumentParser(
        description="MediaPipe-handtracking met RealSense-diepte naar Orca-handgewrichtfracties"
    )
    add_hand_arguments(parser)
    parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    parser.add_argument("--display-scale", type=float, default=1.0)
    parser.add_argument("--no-send", action="store_true", help="Alleen volgen, geen commando's naar de hand sturen")
    parser.add_argument("--color-width", type=int, default=640)
    parser.add_argument("--color-height", type=int, default=480)
    parser.add_argument("--depth-width", type=int, default=640)
    parser.add_argument("--depth-height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    hand = create_hand(args.config_path, use_mock=args.mock)

    # Pipeline wordt vooraf op None gezet zodat de finally-blok veilig kan controleren of die gestart is.
    pipeline = None

    try:
        # Verbindt met de robotische hand en initialiseert de gewrichten.
        connect_hand(hand)
        hand.init_joints(force_calibrate=args.mock)

        # Initialiseert de RealSense-pipeline en configureert kleur- en dieptestreams.
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, args.color_width, args.color_height, rs.format.bgr8, args.fps)
        config.enable_stream(rs.stream.depth, args.depth_width, args.depth_height, rs.format.z16, args.fps)
        profile = pipeline.start(config)
        align = rs.align(rs.stream.color)
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        color_intrinsics = color_profile.get_intrinsics()

        # Configureert MediaPipe HandLandmarker in live-streammodus met één hand tegelijk.
        base_options = python.BaseOptions(model_asset_path=args.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            result_callback=result_callback
        )

        # Maakt de landmarker aan en houdt die actief zolang de hoofdloop loopt.
        with vision.HandLandmarker.create_from_options(options) as landmarker:
            prev = time.time()
            last_timestamp_ms = -1

            while True:
                # Wacht op nieuwe cameraframes en lijn diepte uit op kleur.
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)

                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                # Zet het RealSense-kleurframe om naar een NumPy-array en daarna naar RGB voor MediaPipe.
                raw_frame = np.asanyarray(color_frame.get_data())
                raw_rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)

                # Maakt een strikt oplopende timestamp in milliseconden voor detect_async.
                timestamp_ms = int(time.time() * 1000)
                if timestamp_ms <= last_timestamp_ms:
                    timestamp_ms = last_timestamp_ms + 1
                last_timestamp_ms = timestamp_ms

                # Stuurt het RGB-frame asynchroon naar MediaPipe.
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=raw_rgb)
                landmarker.detect_async(mp_image, timestamp_ms)

                # Spiegelt het beeld (voor een natuurlijker webcambeeld).
                display_frame = cv2.flip(raw_frame.copy(), 1)

                # Tekent de hand, berekent de robotcommando's en stuurt die door.
                display_frame = draw_hand_and_control(
                    display_frame,
                    latest_result,
                    hand,
                    depth_frame=depth_frame,
                    color_intrinsics=color_intrinsics,
                    send_to_hand=not args.no_send
                )

                # Berekent de actuele frames per seconde.
                now = time.time()
                fps = 1.0 / max(now - prev, 1e-6)
                prev = now
                # Toon de FPS-informatie.
                cv2.putText(
                    display_frame,
                    f"FPS: {int(fps)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

                # Past optioneel de weergaveschaal aan.
                final_frame = display_frame
                if abs(args.display_scale - 1.0) > 1e-9:
                    final_frame = cv2.resize(
                        display_frame,
                        None,
                        fx=args.display_scale,
                        fy=args.display_scale
                    )

                # Toont het beeld in een OpenCV-venster. fx=2, fy=2 is dubbel zo groot.
                cv2.imshow("Hand Tracking", cv2.resize(final_frame, None, fx=2, fy=2))

                # Stop met q of Escape.
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break

        # Sluit alle OpenCV-vensters als de hoofdloop normaal eindigt.
        cv2.destroyAllWindows()
        return 0

    except KeyboardInterrupt:
        # Zorgt dat Ctrl+C netjes wordt afgehandeld.
        print("\nOnderbroken.")
        return 0
    finally:
        # Stop de camera alleen als de pipeline effectief is aangemaakt.
        if pipeline is not None:
            pipeline.stop()

        # Sluit de robotische hand en alle OpenCV-vensters altijd af, ook bij fouten.
        shutdown_hand(hand)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
