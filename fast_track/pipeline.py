""" Processing pipeline with detection and tracking """

import logging
import os
import tempfile
from threading import Thread
from typing import Optional
from types import TracebackType

import cv2

from .detectors import ObjectDetector
from .trackers import ObjectTracker
from .database import Database

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Pipeline:
    """ Class that represents a camera-based detection and tracking pipeline

    Attributes:
        camera: opencv-python camera for reading video data.
        detector: object detector.
        tracker: object tracker.
        database: database to store information about tracks and detections.
        frames: list containing processed frames.
        outfile: cv2.VideoWriter object to write processed frames to.
    """

    def __init__(self,
                 camera: cv2.VideoCapture,
                 detector: ObjectDetector,
                 tracker: Optional[ObjectTracker] = None,
                 database: Optional[Database] = None,
                 outfile: Optional[str] = None):
        """ Inits Pipeline class with a given object detector and tracker.

        Args:
            data_path: path to camera file or stream.
            detector: object detector.
            tracker: object tracker.
            database: database to store information about tracks and detections.
            outfile: path to write processed frames to.
        """
        self.camera = camera
        self.detector = detector
        self.tracker = tracker
        self.database = database
        # Output settings
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        fps = int(self.camera.get(cv2.CAP_PROP_FPS))
        w = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if outfile is None:
            temp_dir = tempfile.mkdtemp()
            self.outfile_path = os.path.join(temp_dir, "output.mp4")
        else:
            self.outfile_path = outfile
        self.outfile = cv2.VideoWriter(self.outfile_path, fourcc, fps, (w, h))
        # Frame List
        self.interval_seconds = 60
        self.interval_frames = int(fps * self.interval_seconds)
        self.frame_count = 0
        self.frames = []

    def __enter__(self):
        """ Context manager enter. """
        logger.info("__enter__ | Starting fast_track pipeline...")
        return self

    def __exit__(self,
                 exc_type: Optional[type[BaseException]] = None,
                 exc_value: Optional[BaseException] = None,
                 exc_traceback: Optional[TracebackType] = None) -> None:
        """ Context manager exit. """
        if exc_type or exc_value or exc_traceback:
            logger.error(type)
            logger.error(exc_value)
            logger.error(exc_traceback)
        logger.info("__exit__ | Releasing camera and closing output file.")
        self.camera.release()
        self.outfile.release()
        cv2.destroyAllWindows()
        logger.info("__exit__ | Camera and output file released.")
        if self.database:
            logger.info("__exit__ | Closing database connection.")
            self.database.commit()
            self.database.close()
            logger.info("__exit__ | Database connection closed.")

    def run(self) -> None:
        """ Runs object tracking pipeline. """
        while True:
            ret, frame = self.camera.read()

            if not ret:
                break

            # add frame to database
            if self.frame_count % self.interval_frames == 0 and self.database:
                logger.info(f"run | Adding frame {self.frame_count} to database.")
                t = Thread(target=self.database.add_frame, args=(frame, self.frame_count)).start()

            # detection
            class_ids, scores, boxes = self.detector(frame)

            # tracking
            if self.tracker:
                track_messages = self.tracker.update(boxes, scores, class_ids, frame)
                if self.database:
                    self.database.update(track_messages)

            # write processed frame to output file
            self.outfile.write(frame)

            # update frame count
            self.frame_count += 1

        return self.outfile_path
