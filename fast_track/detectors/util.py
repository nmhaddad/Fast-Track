"""Utilities file for object detection."""

from typing import List, Tuple

from .object_detector import ObjectDetector
from .third_party.yolo_nas.yolo_nas import YOLONAS
from .third_party.yolov8 import YOLOv8, YOLOv8ONNX
from .third_party.yolov7 import YOLOv7ONNX
from .third_party.yolov9 import YOLOv9ONNX
from .third_party.rfdetr import RFDETR


MODELS = {
    "RF-DETR Large": "rfdetr_large",
    "RF-DETR Base": "rfdetr_base",
    "YOLO-NAS L": "yolo_nas_l",
    "YOLO-NAS M": "yolo_nas_m",
    "YOLO-NAS S": "yolo_nas_s",
    "YOLO-NAS": "yolo_nas_custom",
    "YOLOv8 N": "yolov8_n",
    "YOLOv8 S": "yolov8_s",
    "YOLOv8 M": "yolov8_m",
    "YOLOv8 L": "yolov8_l",
    "YOLOv8 X": "yolov8_x",
    "YOLOv8": "yolov8_custom",
    "YOLOv7": "yolov7_custom",
    "YOLOv9": "yolov9_custom",
}


def get_detector(
    weights_path: str, detector_type: str, names: List[str], image_shape: Tuple[int, int], detector_params: dict = {}
) -> ObjectDetector:
    """Detector selector.

    Args:
        weights_path: path to a weights file.
        detector_type: type of the detector to setup.
        names: labels associated with model.
        image_shape: used for input shape for model.
        detector_params: key value parameter pairs.

    Returns:
        An object detector.
    """
    if weights_path == detector_type:
        weights_path = MODELS[detector_type]
    detector_type = MODELS[detector_type]

    if detector_type.startswith("yolo_nas"):
        return YOLONAS(weights_path=weights_path, names=names, image_shape=image_shape, **detector_params)
    elif detector_type.startswith("yolov8"):
        if detector_type.endswith("custom") and weights_path.endswith(".onnx"):
            return YOLOv8ONNX(weights_path=weights_path, names=names, image_shape=image_shape, **detector_params)
        return YOLOv8(weights_path=weights_path, names=names, image_shape=image_shape, **detector_params)
    elif detector_type.startswith("yolov7"):
        return YOLOv7ONNX(weights_path=weights_path, names=names, image_shape=image_shape, **detector_params)
    elif detector_type.startswith("yolov9"):
        return YOLOv9ONNX(weights_path=weights_path, names=names, image_shape=image_shape, **detector_params)
    elif detector_type.startswith("rfdetr"):
        return RFDETR(weights_path=weights_path, model_name=detector_type, names=names, resolution=max(image_shape), **detector_params)
    else:
        raise ValueError("Detector name not found.")
