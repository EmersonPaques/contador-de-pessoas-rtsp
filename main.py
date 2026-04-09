# People Counter RTSP - YOLOv12 + OpenCV
# Modo mosaico para ate 16 cameras configuradas por .env

import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import IterableSimpleNamespace, yaml_load
from ultralytics.utils.checks import check_yaml


DEFAULT_ROI_POLYGON = [
    (700, 120),
    (1750, 120),
    (1910, 1020),
    (850, 1020),
]
DEFAULT_REFERENCE_SIZE = (1920, 1080)
DEFAULT_LINE_IN_Y = 750
DEFAULT_LINE_OUT_Y = 400
DEFAULT_MODEL_PATH = "yolov12n.pt"
DEFAULT_CONFIDENCE = 0.35
DEFAULT_IMG_SIZE = 960
DEFAULT_RECONNECT_DELAY_SECONDS = 5
DEFAULT_PERSON_CLASS_ID = 0
DEFAULT_TRACK_TIMEOUT = 30
DEFAULT_LINE_OFFSET = 20
DEFAULT_TILE_WIDTH = 640
DEFAULT_TILE_HEIGHT = 360
DEFAULT_MAX_CAMERAS = 16
ENV_FILE = Path(".env")


@dataclass
class CameraConfig:
    slot: int
    name: str
    host: str
    channel: int
    url: str
    reference_size: tuple[int, int]
    roi_polygon: list[tuple[int, int]]
    line_in_y: int
    line_out_y: int


@dataclass
class CameraState:
    config: CameraConfig
    tracker: BYTETracker
    cap: cv2.VideoCapture | None = None
    online: bool = False
    next_reconnect_at: float = 0.0
    consecutive_failures: int = 0
    in_count: int = 0
    out_count: int = 0
    total_unique_crossings: int = 0
    track_history: defaultdict = field(default_factory=lambda: defaultdict(dict))
    fps_counter: int = 0
    fps_start_time: float = field(default_factory=time.time)
    current_fps: float = 0.0


def load_env_file(path: Path) -> None:
    """Carrega um arquivo .env simples sem dependencias extras."""
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def env_str(key: str, default: str) -> str:
    return os.getenv(key, default).strip()


def env_int(key: str, default: int) -> int:
    value = os.getenv(key)
    if value is None or value.strip() == "":
        return default
    return int(value)


def env_float(key: str, default: float) -> float:
    value = os.getenv(key)
    if value is None or value.strip() == "":
        return default
    return float(value)


def env_bool(key: str, default: bool) -> bool:
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "sim"}


def parse_reference_size(value: str) -> tuple[int, int]:
    parts = [part.strip() for part in value.replace("x", ",").split(",") if part.strip()]
    if len(parts) != 2:
        return DEFAULT_REFERENCE_SIZE
    return int(parts[0]), int(parts[1])


def parse_polygon(value: str) -> list[tuple[int, int]]:
    if not value.strip():
        return DEFAULT_ROI_POLYGON

    points: list[tuple[int, int]] = []
    for raw_point in value.split(";"):
        pieces = [piece.strip() for piece in raw_point.split(",") if piece.strip()]
        if len(pieces) != 2:
            continue
        points.append((int(pieces[0]), int(pieces[1])))

    return points if len(points) >= 3 else DEFAULT_ROI_POLYGON


def scale_point(point: tuple[int, int], sx: float, sy: float) -> tuple[int, int]:
    return int(point[0] * sx), int(point[1] * sy)


def get_scaled_geometry(config: CameraConfig, frame_shape: tuple[int, ...]):
    """Retorna linhas e ROI escaladas para o tamanho atual do frame."""
    frame_h, frame_w = frame_shape[:2]
    ref_w, ref_h = config.reference_size
    sx = frame_w / ref_w
    sy = frame_h / ref_h

    roi_polygon = np.array([scale_point(point, sx, sy) for point in config.roi_polygon], dtype=np.int32)
    min_x = int(np.min(roi_polygon[:, 0]))
    max_x = int(np.max(roi_polygon[:, 0]))
    line_in_y = int(config.line_in_y * sy)
    line_out_y = int(config.line_out_y * sy)

    line_in_start = (min_x, line_in_y)
    line_in_end = (max_x, line_in_y)
    line_out_start = (min_x, line_out_y)
    line_out_end = (max_x, line_out_y)

    return line_in_start, line_in_end, line_out_start, line_out_end, roi_polygon


def mask_frame_to_roi(frame: np.ndarray, roi_polygon: np.ndarray) -> np.ndarray:
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [roi_polygon], 255)
    return cv2.bitwise_and(frame, frame, mask=mask)


def point_in_roi(point: tuple[int, int], roi_polygon: np.ndarray) -> bool:
    return cv2.pointPolygonTest(roi_polygon.astype(np.float32), point, False) >= 0


def get_side_of_lines(center: tuple[int, int], line_in_y: int, line_out_y: int, line_offset: int) -> str:
    _, cy = center
    if cy < line_out_y - line_offset:
        return "above_out"
    if line_out_y - line_offset <= cy <= line_in_y + line_offset:
        return "between"
    return "below_in"


def check_line_crossing(
    state: CameraState,
    track_id: int,
    current_center: tuple[int, int],
    line_in_y: int,
    line_out_y: int,
    line_offset: int,
) -> bool:
    """Conta entrada e saida por camera usando o historico do track."""
    if track_id not in state.track_history:
        state.track_history[track_id] = {
            "last_center": current_center,
            "last_side": get_side_of_lines(current_center, line_in_y, line_out_y, line_offset),
            "last_seen": time.time(),
            "passed_red": False,
            "passed_green": False,
            "crossed": False,
        }
        return False

    last_side = state.track_history[track_id]["last_side"]
    current_side = get_side_of_lines(current_center, line_in_y, line_out_y, line_offset)

    state.track_history[track_id]["last_center"] = current_center
    state.track_history[track_id]["last_side"] = current_side
    state.track_history[track_id]["last_seen"] = time.time()

    if last_side != current_side:
        if last_side == "above_out" and current_side == "between":
            state.track_history[track_id]["passed_red"] = True

        if last_side == "below_in" and current_side == "between":
            state.track_history[track_id]["passed_green"] = True

        if not state.track_history[track_id]["crossed"]:
            if (last_side == "above_out" and current_side == "below_in") or (
                last_side == "between"
                and current_side == "below_in"
                and state.track_history[track_id]["passed_red"]
            ):
                state.in_count += 1
                state.total_unique_crossings += 1
                print(f"[CAM {state.config.slot:02d}] ENTRADA: ID {track_id} - Total IN: {state.in_count}")
                state.track_history[track_id]["crossed"] = True
                state.track_history[track_id]["passed_red"] = False
                state.track_history[track_id]["passed_green"] = False
                return True

            if (last_side == "below_in" and current_side == "above_out") or (
                last_side == "between"
                and current_side == "above_out"
                and state.track_history[track_id]["passed_green"]
            ):
                state.out_count += 1
                state.total_unique_crossings += 1
                print(f"[CAM {state.config.slot:02d}] SAIDA: ID {track_id} - Total OUT: {state.out_count}")
                state.track_history[track_id]["crossed"] = True
                state.track_history[track_id]["passed_red"] = False
                state.track_history[track_id]["passed_green"] = False
                return True

    if current_side == "between":
        state.track_history[track_id]["crossed"] = False

    return False


def cleanup_stale_tracks(state: CameraState, track_timeout: int) -> None:
    current_time = time.time()
    stale_tracks = [
        track_id
        for track_id, data in state.track_history.items()
        if current_time - data["last_seen"] > track_timeout
    ]

    for track_id in stale_tracks:
        del state.track_history[track_id]


def resize_for_processing(frame: np.ndarray, max_height: int) -> np.ndarray:
    if max_height <= 0:
        return frame

    height, width = frame.shape[:2]
    if height <= max_height:
        return frame

    scale = max_height / height
    new_width = int(width * scale)
    return cv2.resize(frame, (new_width, max_height))


def create_byte_tracker() -> BYTETracker:
    tracker_cfg = check_yaml("bytetrack.yaml")
    tracker_args = IterableSimpleNamespace(**yaml_load(tracker_cfg))
    return BYTETracker(args=tracker_args, frame_rate=30)


def open_rtsp_stream(url: str):
    print(f"🔗 Tentando conectar ao RTSP: {url}")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if cap.isOpened():
        print("✅ RTSP conectado com sucesso!")
        return cap, True
    print("❌ Falha ao conectar ao RTSP")
    return None, False


def ensure_camera_connected(state: CameraState, reconnect_delay_seconds: int) -> bool:
    if state.online and state.cap is not None:
        return True

    if time.time() < state.next_reconnect_at:
        return False

    cap, success = open_rtsp_stream(state.config.url)
    if success:
        state.cap = cap
        state.online = True
        state.consecutive_failures = 0
        state.next_reconnect_at = 0.0
        print(f"📹 Camera {state.config.slot:02d} online: {state.config.name}")
        return True

    state.online = False
    state.next_reconnect_at = time.time() + reconnect_delay_seconds
    return False


def release_camera(state: CameraState) -> None:
    if state.cap is not None:
        state.cap.release()
        state.cap = None
    state.online = False


def build_placeholder_tile(state: CameraState, tile_size: tuple[int, int]) -> np.ndarray:
    tile_w, tile_h = tile_size
    frame = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
    title = f"CAM {state.config.slot:02d} - {state.config.name}"
    cv2.putText(frame, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, "SEM SINAL", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    cv2.putText(frame, state.config.host, (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
    cv2.putText(frame, f"Canal {state.config.channel}", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
    return frame


def draw_overlay(
    state: CameraState,
    frame: np.ndarray,
    results,
    line_in_start: tuple[int, int],
    line_in_end: tuple[int, int],
    line_out_start: tuple[int, int],
    line_out_end: tuple[int, int],
    roi_polygon: np.ndarray,
    confidence: float,
    person_class_id: int,
    line_offset: int,
) -> np.ndarray:
    cv2.polylines(frame, [roi_polygon], isClosed=True, color=(255, 0, 0), thickness=2)
    cv2.line(frame, line_in_start, line_in_end, (0, 255, 0), 2)
    cv2.putText(
        frame,
        "IN",
        (line_in_start[0] + 10, max(20, line_in_start[1] - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )
    cv2.line(frame, line_out_start, line_out_end, (0, 0, 255), 2)
    cv2.putText(
        frame,
        "OUT",
        (line_out_start[0] + 10, max(20, line_out_start[1] - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
    )

    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            if int(box.cls) != person_class_id:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            if conf < confidence:
                continue

            track_id = None
            if hasattr(box, "id") and box.id is not None:
                track_id = int(box.id[0])

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            if not point_in_roi((cx, cy), roi_polygon):
                continue

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
            label = f"ID:{track_id if track_id is not None else '?'} {conf:.2f}"
            cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if track_id is not None:
                crossed = check_line_crossing(
                    state,
                    track_id,
                    (cx, cy),
                    line_in_start[1],
                    line_out_start[1],
                    line_offset,
                )
                if crossed:
                    cv2.line(frame, line_in_start, line_in_end, (255, 255, 255), 4)
                    cv2.line(frame, line_out_start, line_out_end, (255, 255, 255), 4)

    tile_title = f"CAM {state.config.slot:02d} - {state.config.name}"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 78), (0, 0, 0), -1)
    cv2.putText(frame, tile_title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"IN: {state.in_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.putText(frame, f"OUT: {state.out_count}", (110, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    cv2.putText(frame, f"TOTAL: {state.total_unique_crossings}", (230, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {state.current_fps:.1f}", (430, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return frame


def process_camera_frame(
    state: CameraState,
    model: YOLO,
    confidence: float,
    img_size: int,
    person_class_id: int,
    line_offset: int,
    track_timeout: int,
    use_half: bool,
    reconnect_delay_seconds: int,
) -> np.ndarray:
    if not ensure_camera_connected(state, reconnect_delay_seconds):
        return None

    ret, frame = state.cap.read()
    if not ret:
        state.consecutive_failures += 1
        state.online = False
        state.next_reconnect_at = time.time() + reconnect_delay_seconds
        print(f"❌ Camera {state.config.slot:02d} falhou ao ler frame ({state.consecutive_failures})")
        release_camera(state)
        return None

    state.consecutive_failures = 0
    state.online = True
    frame = resize_for_processing(frame, img_size)
    line_in_start, line_in_end, line_out_start, line_out_end, roi_polygon = get_scaled_geometry(state.config, frame.shape)
    frame_for_tracking = mask_frame_to_roi(frame, roi_polygon)

    results = model.predict(
        frame_for_tracking,
        conf=confidence,
        classes=[person_class_id],
        half=use_half,
        verbose=False,
    )

    if results and results[0].boxes is not None:
        det = results[0].boxes.cpu().numpy()
        if len(det) > 0:
            tracks = state.tracker.update(det, frame)
            if len(tracks) > 0:
                idx = tracks[:, -1].astype(int)
                results[0] = results[0][idx]
                results[0].update(boxes=torch.as_tensor(tracks[:, :-1]))

    annotated = draw_overlay(
        state,
        frame,
        results,
        line_in_start,
        line_in_end,
        line_out_start,
        line_out_end,
        roi_polygon,
        confidence,
        person_class_id,
        line_offset,
    )

    state.fps_counter += 1
    elapsed = time.time() - state.fps_start_time
    if elapsed >= 1.0:
        state.current_fps = state.fps_counter / elapsed
        state.fps_counter = 0
        state.fps_start_time = time.time()

    cleanup_stale_tracks(state, track_timeout)
    return annotated


def build_mosaic(frames: list[np.ndarray], tile_size: tuple[int, int]) -> np.ndarray:
    tile_w, tile_h = tile_size
    total = len(frames)
    cols = math.ceil(math.sqrt(total))
    rows = math.ceil(total / cols)
    resized_frames = [cv2.resize(frame, (tile_w, tile_h)) for frame in frames]

    canvas_rows = []
    index = 0
    blank = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
    for _ in range(rows):
        row_tiles = []
        for _ in range(cols):
            row_tiles.append(resized_frames[index] if index < total else blank.copy())
            index += 1
        canvas_rows.append(np.hstack(row_tiles))
    return np.vstack(canvas_rows)


def build_rtsp_url(
    username: str,
    password: str,
    host: str,
    channel: int,
    port: int,
    subtype: int,
    path_template: str,
) -> str:
    return path_template.format(
        username=quote(username, safe=""),
        password=quote(password, safe=""),
        host=host,
        port=port,
        channel=channel,
        subtype=subtype,
    )


def load_camera_configs() -> list[CameraConfig]:
    username = env_str("RTSP_USERNAME", "admin")
    password = env_str("RTSP_PASSWORD", "")
    port = env_int("RTSP_PORT", 554)
    subtype = env_int("RTSP_SUBTYPE", 0)
    path_template = env_str(
        "RTSP_PATH_TEMPLATE",
        "rtsp://{username}:{password}@{host}:{port}/cam/realmonitor?channel={channel}&subtype={subtype}",
    )
    max_cameras = min(env_int("APP_MAX_CAMERAS", DEFAULT_MAX_CAMERAS), DEFAULT_MAX_CAMERAS)
    default_reference_size = parse_reference_size(
        env_str("CAM_DEFAULT_REFERENCE_SIZE", f"{DEFAULT_REFERENCE_SIZE[0]},{DEFAULT_REFERENCE_SIZE[1]}")
    )
    default_roi_polygon = parse_polygon(
        env_str(
            "CAM_DEFAULT_ROI_POLYGON",
            ";".join(f"{x},{y}" for x, y in DEFAULT_ROI_POLYGON),
        )
    )
    default_line_in_y = env_int("CAM_DEFAULT_LINE_IN_Y", DEFAULT_LINE_IN_Y)
    default_line_out_y = env_int("CAM_DEFAULT_LINE_OUT_Y", DEFAULT_LINE_OUT_Y)

    camera_configs: list[CameraConfig] = []
    for slot in range(1, max_cameras + 1):
        prefix = f"CAM_{slot:02d}"
        enabled = env_bool(f"{prefix}_ENABLED", False)
        host = env_str(f"{prefix}_HOST", "")
        channel = env_int(f"{prefix}_CHANNEL", 0) if os.getenv(f"{prefix}_CHANNEL") else 0

        if not enabled or not host or channel <= 0:
            continue

        name = env_str(f"{prefix}_NAME", f"Camera {slot:02d}")
        reference_size = parse_reference_size(
            env_str(f"{prefix}_REFERENCE_SIZE", f"{default_reference_size[0]},{default_reference_size[1]}")
        )
        roi_polygon = parse_polygon(
            env_str(
                f"{prefix}_ROI_POLYGON",
                ";".join(f"{x},{y}" for x, y in default_roi_polygon),
            )
        )
        line_in_y = env_int(f"{prefix}_LINE_IN_Y", default_line_in_y)
        line_out_y = env_int(f"{prefix}_LINE_OUT_Y", default_line_out_y)
        camera_subtype = env_int(f"{prefix}_SUBTYPE", subtype)

        camera_configs.append(
            CameraConfig(
                slot=slot,
                name=name,
                host=host,
                channel=channel,
                url=build_rtsp_url(username, password, host, channel, port, camera_subtype, path_template),
                reference_size=reference_size,
                roi_polygon=roi_polygon,
                line_in_y=line_in_y,
                line_out_y=line_out_y,
            )
        )

    return camera_configs


def main():
    load_env_file(ENV_FILE)

    model_path = env_str("MODEL_PATH", DEFAULT_MODEL_PATH)
    confidence = env_float("CONFIDENCE", DEFAULT_CONFIDENCE)
    img_size = env_int("IMG_SIZE", DEFAULT_IMG_SIZE)
    person_class_id = env_int("PERSON_CLASS_ID", DEFAULT_PERSON_CLASS_ID)
    reconnect_delay_seconds = env_int("RECONNECT_DELAY_SECONDS", DEFAULT_RECONNECT_DELAY_SECONDS)
    track_timeout = env_int("TRACK_TIMEOUT", DEFAULT_TRACK_TIMEOUT)
    line_offset = env_int("LINE_OFFSET", DEFAULT_LINE_OFFSET)
    tile_size = (
        env_int("MOSAIC_TILE_WIDTH", DEFAULT_TILE_WIDTH),
        env_int("MOSAIC_TILE_HEIGHT", DEFAULT_TILE_HEIGHT),
    )
    show_window = env_bool("SHOW_WINDOW", True)
    window_name = env_str("WINDOW_NAME", "People Counter RTSP Mosaic")

    camera_configs = load_camera_configs()
    if not camera_configs:
        print("❌ Nenhuma camera ativa encontrada no .env")
        print("   Configure CAM_01_ENABLED=true, CAM_01_HOST e CAM_01_CHANNEL")
        return

    print("🚀 Iniciando People Counter RTSP em modo mosaico")
    print(f"🤖 Modelo: {model_path}")
    print(f"🎯 Confianca: {confidence}")
    print(f"📏 IMG_SIZE: {img_size}")
    print(f"🧩 Cameras ativas: {len(camera_configs)}")
    for config in camera_configs:
        print(f"   CAM {config.slot:02d}: {config.name} -> {config.host} canal {config.channel}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    use_half = device.startswith("cuda")
    print(f"🖥️ Device: {device}")
    print("⏳ Carregando modelo compartilhado...")
    shared_model = YOLO(model_path).to(device)
    camera_states = [CameraState(config=config, tracker=create_byte_tracker()) for config in camera_configs]
    print("✅ Modelo carregado")

    try:
        while True:
            mosaic_frames = []
            for state in camera_states:
                frame = process_camera_frame(
                    state,
                    model=shared_model,
                    confidence=confidence,
                    img_size=img_size,
                    person_class_id=person_class_id,
                    line_offset=line_offset,
                    track_timeout=track_timeout,
                    use_half=use_half,
                    reconnect_delay_seconds=reconnect_delay_seconds,
                )
                if frame is None:
                    frame = build_placeholder_tile(state, tile_size)
                mosaic_frames.append(frame)

            mosaic = build_mosaic(mosaic_frames, tile_size)

            if show_window:
                cv2.imshow(window_name, mosaic)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    print("👋 Saindo...")
                    break
            else:
                time.sleep(0.01)
    finally:
        for state in camera_states:
            release_camera(state)
        cv2.destroyAllWindows()

        print("📊 Resumo final:")
        for state in camera_states:
            print(
                f"   CAM {state.config.slot:02d} {state.config.name}: "
                f"IN={state.in_count} OUT={state.out_count} TOTAL={state.total_unique_crossings}"
            )


if __name__ == "__main__":
    main()
