"""Database class to store information about tracks and detections."""

import datetime
import logging
from typing import Any, Dict, List

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .schemas import Base, Detection, Frame, Job, Track
from .utils import encode_image, generate_frame_caption

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SQLDatabase:
    """Database class to store information about tracks and detections."""

    def __init__(self, db_uri: str, class_names: List[str], create_image_captions: bool = False):
        """Inits Database class with a given database URI.

        Args:
            db_uri: database URI.
            create_image_captions: bool to use VLM to generate image captions (OpenAI).
        """
        self.db_uri = db_uri
        self.class_names = class_names
        self.create_image_captions = create_image_captions

        self.db = None
        self.job_id = None

        self._connect_db()

    def _connect_db(self) -> None:
        """Connects to the database and creates tables if they don't exist."""
        engine = create_engine(self.db_uri)
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()
        self.db = session
        job = Job(job_name=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.db.add(job)
        self.db.commit()
        self.job_id = job.job_id

    def close(self) -> None:
        """Closes the database connection."""
        self.db.close()

    def commit(self) -> None:
        """Commits the database session."""
        self.db.commit()

    def update_detections(self, detections: List[Dict[str, Any]]) -> None:
        """Updates the database with detections.

        Args:
            detections: list of detections to update.
        """
        for detection in detections:
            detection["job_id"] = self.job_id
            self.db.add(Detection(**detection))
        self.db.commit()

    def update_tracks(self, track_messages: List[Dict[str, Any]]) -> None:
        """Updates the database with tracks."""
        for track_message in track_messages:
            detection_ids = track_message.pop("detection_ids")
            track_message["class_name"] = self.class_names[track_message.pop("class_id")]
            # check to see if track_id already exists, if so, update it, else add it
            existing_track = self.db.query(Track).filter(Track.track_id == track_message["track_id"]).first()
            if existing_track:
                existing_track.count = track_message["count"]
                existing_track.is_activated = track_message["is_activated"]
                existing_track.state = track_message["state"]
                existing_track.score = track_message["score"]
                existing_track.curr_frame_number = track_message["curr_frame_number"]
                existing_track.time_since_update = track_message["time_since_update"]
                existing_track.location = track_message["location"]
                existing_track.class_name = track_message["class_name"]
            else:
                self.db.add(Track(**track_message, job_id=self.job_id))

            self.db.flush()

            for detection_id in detection_ids:
                detection = self.db.query(Detection).filter(Detection.detection_id == detection_id).first()
                if detection:
                    detection.track_id = track_message["track_id"]

        self.db.commit()

    def add_frame(self, frame: np.ndarray, frame_number: int) -> None:
        """Adds a frame to the database Frames table.

        Args:
            frame: frame to add.
        """
        image_base64 = encode_image(frame)
        self.db.add(
            Frame(
                frame_number=frame_number,
                image_base64=image_base64,
                timestamp=datetime.timezone.utc,
                image_caption=generate_frame_caption(image_base64) if self.create_image_captions else None,
                job_id=self.job_id,
            )
        )
        try:
            self.db.commit()
        except Exception:
            logger.warning("add_frame | Failed to add frame to database.")
