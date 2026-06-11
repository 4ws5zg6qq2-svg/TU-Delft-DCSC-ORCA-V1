import cv2
import numpy as np
import pandas as pd
import math
from pathlib import Path

# Dit Python-script zoekt eerst naar een groene lijn en een gele lijn en berekent daarna de hoek tussen die twee lijnen, in alle foto's in the map Camera. 
# Als er geen combinatie van een groene en gele lijn wordt gevonden, gebruikt het script een fallback-methode waarbij het zoekt naar twee groene lijnen.
# Bij die fallback-methode moeten de twee groene lijnrichtingen minstens 6 graden van elkaar verschillen. 
# Als het verschil kleiner is dan 6 graden, worden ze beschouwd als dezelfde richting en dus niet als twee aparte lijnen.

# Instellingen voor mappen, uitvoerbestanden en kleurdetectie.
INPUT_FOLDER = "Camera"
OUTPUT_EXCEL = "gedetecteerdeHoeken.xlsx"
DEBUG_FOLDER = "debug_output"

MIN_SECOND_DIRECTION_DIFF = 6

LOWER_GREEN = np.array([55, 120, 100])
UPPER_GREEN = np.array([80, 255, 255])

LOWER_YELLOW = np.array([20, 100, 100])
UPPER_YELLOW = np.array([38, 255, 255])


def detect_best_colored_line(image, lower_color, upper_color):
    # Zoekt de langste duidelijke lijn binnen een opgegeven HSV-kleurbereik.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_color, upper_color)

    # DOet kleine ruis weg en sluit kleine gaten in het masker.
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    # Beoordeelt elke contour op grootte, lengte en richting.
    for contour in contours:
        area = cv2.contourArea(contour)

        if area < 50:
            continue

        # Bepaalt de begrenzende rechthoek rond de contour.
        x, y, w, h = cv2.boundingRect(contour)
        length_estimate = max(w, h)

        if length_estimate < 60:
            continue

        vx, vy, x0, y0 = cv2.fitLine(
            contour,
            cv2.DIST_L2,
            0,
            0.01,
            0.01
        )

        vx = float(vx[0])
        vy = float(vy[0])
        x0 = float(x0[0])
        y0 = float(y0[0])

        angle = math.degrees(math.atan2(vy, vx)) % 180

        candidates.append({
            "angle": angle,
            "length": length_estimate,
            "contour": contour,
            "fit": (vx, vy, x0, y0),
            "mask": mask
        })

    # Geeft None terug wanneer er geen bruikbare lijn gevonden is.
    if len(candidates) == 0:
        return None, mask

    # Kiest de langste kandidaat als beste lijn.
    best = sorted(candidates, key=lambda item: item["length"], reverse=True)[0]

    return best, mask


def line_length(line):
    x1, y1, x2, y2 = line
    return math.hypot(x2 - x1, y2 - y1)


def line_angle_deg(line):
    x1, y1, x2, y2 = line
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    return angle % 180


def angle_difference_180(a, b):
    # Berekent het kleinste hoekverschil tussen twee lijnrichtingen.
    diff = abs(a - b) % 180
    if diff > 90:
        diff = 180 - diff
    return diff


def weighted_average_angle(angles, weights):
    x = 0.0
    y = 0.0

    for angle, weight in zip(angles, weights):
        rad = math.radians(2 * angle)
        x += weight * math.cos(rad)
        y += weight * math.sin(rad)

    avg = 0.5 * math.degrees(math.atan2(y, x))
    return avg % 180


def cluster_lines_into_two_directions(lines):
    # Verdeelt gevonden lijnsegmenten in twee hoofdrichtingen.
    info = []

    for line in lines:
        angle = line_angle_deg(line)
        length = line_length(line)

        if length > 0:
            info.append((line, angle, length))

    if len(info) < 2:
        raise RuntimeError("Niet genoeg geldige lijn segmenten gevonden.")

    info = sorted(info, key=lambda x: x[2], reverse=True)

    direction1 = info[0][1]
    direction2 = None

    for _, angle, _ in info[1:]:
        if angle_difference_180(direction1, angle) > MIN_SECOND_DIRECTION_DIFF:
            direction2 = angle
            break

    if direction2 is None:
        raise RuntimeError("Alleen 1 duidelijke groene lijn richting gevonden.")

    cluster1 = []
    cluster2 = []

    for _ in range(10):
        cluster1 = []
        cluster2 = []

        for line, angle, length in info:
            d1 = angle_difference_180(angle, direction1)
            d2 = angle_difference_180(angle, direction2)

            if d1 <= d2:
                cluster1.append((line, angle, length))
            else:
                cluster2.append((line, angle, length))

        if len(cluster1) == 0 or len(cluster2) == 0:
            raise RuntimeError("Kon de lijnen niet in 2 richtingen splitsen.")

        direction1 = weighted_average_angle(
            [x[1] for x in cluster1],
            [x[2] for x in cluster1]
        )

        direction2 = weighted_average_angle(
            [x[1] for x in cluster2],
            [x[2] for x in cluster2]
        )

    return direction1, direction2, cluster1, cluster2


def detect_angle_in_photo(image_path, save_debug=True):
    # Bepaalt de hoek in één foto en bewaart optioneel debugbeelden.
    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError("Kon foto niet inlezen.")

    green_line, green_mask = detect_best_colored_line(
        image,
        LOWER_GREEN,
        UPPER_GREEN
    )

    yellow_line, yellow_mask = detect_best_colored_line(
        image,
        LOWER_YELLOW,
        UPPER_YELLOW
    )

    # Eerst wordt geprobeerd om de hoek tussen een groene en gele lijn te nemen.
    if green_line is not None and yellow_line is not None:
        small_angle = angle_difference_180(
            green_line["angle"],
            yellow_line["angle"]
        )

        method = "green_yellow"

        if save_debug:
            debug_image = image.copy()

            for line_data, color in [
                (green_line, (0, 0, 255)),
                (yellow_line, (255, 0, 0))
            ]:
                contour = line_data["contour"]
                vx, vy, x0, y0 = line_data["fit"]

                cv2.drawContours(debug_image, [contour], -1, color, 2)

                h_img, w_img = debug_image.shape[:2]
                line_length = max(h_img, w_img)

                x1 = int(x0 - vx * line_length)
                y1 = int(y0 - vy * line_length)
                x2 = int(x0 + vx * line_length)
                y2 = int(y0 + vy * line_length)

                cv2.line(debug_image, (x1, y1), (x2, y2), color, 2)

            cv2.putText(
                debug_image,
                f"Small angle: {small_angle:.2f} deg",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

            debug_path = Path(DEBUG_FOLDER)
            debug_path.mkdir(exist_ok=True)

            cv2.imwrite(
                str(debug_path / f"{image_path.stem}_debug.jpg"),
                debug_image
            )

            combined_mask = cv2.bitwise_or(green_mask, yellow_mask)

            cv2.imwrite(
                str(debug_path / f"{image_path.stem}_mask.jpg"),
                combined_mask
            )

        return {
            "filename": image_path.name,
            "small_angle_deg": round(small_angle, 2),
            "status": "ok",
            "error": ""
        }

    # Als groen-geel niet lukt, wordt teruggevallen op twee groene lijnrichtingen.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    edges = cv2.Canny(mask, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=25,
        minLineLength=40,
        maxLineGap=30
    )

    if lines is None:
        raise RuntimeError("Geen groen/geel paar gevonden EN geen 2 groene lijnen gevonden")

    lines = [line[0] for line in lines]

    direction1, direction2, cluster1, cluster2 = cluster_lines_into_two_directions(lines)

    small_angle = angle_difference_180(direction1, direction2)

    if save_debug:
        debug_image = image.copy()

        for line, _, _ in cluster1:
            x1, y1, x2, y2 = line
            cv2.line(debug_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        for line, _, _ in cluster2:
            x1, y1, x2, y2 = line
            cv2.line(debug_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        cv2.putText(
            debug_image,
            f"Small angle: {small_angle:.2f} deg",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        debug_path = Path(DEBUG_FOLDER)
        debug_path.mkdir(exist_ok=True)

        cv2.imwrite(
            str(debug_path / f"{image_path.stem}_debug.jpg"),
            debug_image
        )

        cv2.imwrite(
            str(debug_path / f"{image_path.stem}_mask.jpg"),
            mask
        )

    return {
        "filename": image_path.name,
        "small_angle_deg": round(small_angle, 2),
        "status": "ok",
        "error": ""
    }


def process_folder():
    # Verwerkt alle foto's in de inputmap en schrijft de resultaten naar Excel.
    folder = Path(INPUT_FOLDER)

    if not folder.exists():
        raise FileNotFoundError(f"Folder bestaat niet: {INPUT_FOLDER}")

    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

    image_files = [
        file for file in folder.iterdir()
        if file.suffix.lower() in image_extensions
    ]

    if len(image_files) == 0:
        raise RuntimeError(f"Geen foto files gevonden in folder: {INPUT_FOLDER}")

    results = []

    for image_path in sorted(image_files):
        print(f"Processing: {image_path.name}")

        try:
            result = detect_angle_in_photo(image_path, save_debug=True)

        except Exception as e:
            result = {
                "filename": image_path.name,
                "small_angle_deg": "",
                "status": "failed",
                "error": str(e)
            }

        results.append(result)

    df = pd.DataFrame(results)

    df.to_excel(OUTPUT_EXCEL, index=False)

    print()
    print(f"Done.")
    print(f"Excel file saved as: {OUTPUT_EXCEL}")
    print(f"Debug images saved in: {DEBUG_FOLDER}")


if __name__ == "__main__":
    process_folder()
