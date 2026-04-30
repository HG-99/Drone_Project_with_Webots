import os
import cv2
import yaml
import numpy as np


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_aruco_dictionary(dict_name: str):
    aruco = cv2.aruco
    if not hasattr(aruco, dict_name):
        raise ValueError(f"Unknown ArUco dictionary: {dict_name}")
    return aruco.getPredefinedDictionary(getattr(aruco, dict_name))


def generate_marker_image(dictionary, marker_id, marker_code_size_px, canvas_size_px, margin_px):
    aruco = cv2.aruco

    if hasattr(aruco, "generateImageMarker"):
        marker = aruco.generateImageMarker(dictionary, marker_id, marker_code_size_px)
    else:
        marker = np.zeros((marker_code_size_px, marker_code_size_px), dtype=np.uint8)
        aruco.drawMarker(dictionary, marker_id, marker_code_size_px, marker, 1)

    canvas = np.ones((canvas_size_px, canvas_size_px), dtype=np.uint8) * 255
    canvas[
        margin_px:margin_px + marker_code_size_px,
        margin_px:margin_px + marker_code_size_px
    ] = marker
    return canvas


def main():
    config_path = "project/config/config.yaml"
    cfg = load_config(config_path)

    dict_name = cfg["marker"]["dictionary"]
    marker_ids = cfg["marker"]["ids"]

    canvas_size_px = cfg["texture"]["canvas_size_px"]
    marker_code_size_px = cfg["texture"]["marker_code_size_px"]
    margin_px = cfg["texture"]["margin_px"]
    save_dir = cfg["texture"]["save_dir"]

    marker_outer_size_m = cfg["marker"]["outer_size_m"]

    os.makedirs(save_dir, exist_ok=True)

    dictionary = get_aruco_dictionary(dict_name)

    code_ratio = marker_code_size_px / canvas_size_px
    code_size_m = marker_outer_size_m * code_ratio

    print(f"Config loaded: {config_path}")
    print(f"Marker outer size in world : {marker_outer_size_m:.3f} m")
    print(f"Marker code size in world  : {code_size_m:.3f} m")
    print(f"Save dir                  : {save_dir}")

    for marker_id in marker_ids:
        img = generate_marker_image(
            dictionary=dictionary,
            marker_id=marker_id,
            marker_code_size_px=marker_code_size_px,
            canvas_size_px=canvas_size_px,
            margin_px=margin_px,
        )

        save_path = os.path.join(save_dir, f"aruco_{dict_name}_{marker_id}.png")
        ok = cv2.imwrite(save_path, img)

        if ok:
            print(f"Saved: {save_path}")
        else:
            print(f"Failed: {save_path}")


if __name__ == "__main__":
    main()