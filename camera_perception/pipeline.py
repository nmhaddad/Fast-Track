""" Processing pipeline with detection and tracking """

from __future__ import annotations

import logging
from typing import Optional, Dict, Any
from types import TracebackType

import cv2
import numpy as np

from .object_detection import YOLOv7
from .object_tracking import BYTETracker
from .utils import save_video


class Pipeline:
    """ Class that represents a camera-based detection and tracking pipeline
    
    Attributes:
        data_path
        camera
        detector
        tracker
        frame_count
        frames
        results
        outfile
    """

    def __init__(self, 
                 data_path: str, 
                 detector: Dict[str, Any], 
                 tracker: Dict[str, Any], 
                 outfile: Optional[str] = 'video.avi'):
        self.data_path = data_path
        self.camera = cv2.VideoCapture(self.data_path)
        self.detector = YOLOv7(**detector, image_shape = (self.camera.get(3), self.camera.get(4)))
        self.tracker = BYTETracker(**tracker)
        self.frame_count = 0
        self.frames = []
        self.results = []
        self.outfile = outfile

    def __enter__(self):
        return self

    def __exit__(self, 
                 type: Optional[type[BaseException]] = None, 
                 value: Optional[BaseException] = None, 
                 traceback: Optional[TracebackType] = None) -> None:
        if type or value or traceback:
            logging.info(type)
            logging.info(value)
            logging.info(traceback)
        self.camera.release()
        cv2.destroyAllWindows()

    def run(self) -> None:

        while True:
            ret, frame = self.camera.read()

            if not ret:
                break

            # detection
            class_ids, scores, boxes = self.detector.detect(frame)
            self.detector.visualize_detections(frame, class_ids, scores, boxes)

            # tracking
            online_targets = self.tracker.update(np.array(boxes), np.array(scores))
            self.tracker.visualize_tracks(online_targets, frame, self.frame_count)
        
            # logging.info(track_str)
            # self.results.append(track_str)
            self.frames.append(frame)
            self.frame_count += 1

        if self.frames:
            logging.info(f"saving output to {self.outfile}")
            save_video(self.frames, self.outfile, fps=self.camera.get(cv2.CAP_PROP_FPS))
