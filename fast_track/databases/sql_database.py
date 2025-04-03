"""Database class to store information about tracks and detections."""

import datetime
import logging
from typing import Any, Dict, List

import numpy as np
from sqlalchemy import create_engine, select
from sqlalchemy.orm import scoped_session, sessionmaker

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
        self.job_id = None

        # Connect to the database and create tables if they don't exist
        engine = create_engine(db_uri)
        Base.metadata.create_all(engine)
        # Create a SessionFactory for thread-safe sessions
        self.SessionFactory = scoped_session(sessionmaker(bind=engine))

        self._create_new_job()

    def _create_new_job(self) -> None:
        """Connects to the database and creates tables if they don't exist."""
        session = self.SessionFactory
        job = Job(job_name=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        session.add(job)
        session.commit()
        self.job_id = job.job_id
        session.close()

    def update_detections(self, detections: List[Dict[str, Any]]) -> None:
        """Updates the database with detections.

        Args:
            detections: list of detections to update.
        """
        session = self.SessionFactory
        detection_objects = [
            Detection(
                job_id=self.job_id,
                detection_id=detection["detection_id"],
                class_id=int(detection["class_id"]),
                class_name=detection["class_name"],
                bbox=detection["bbox"].astype(np.int32).tolist(),
                confidence=float(detection["confidence"]),
                frame_number=int(detection["frame_number"]),
                timestamp=detection["timestamp"],
            )
            for detection in detections
        ]
        session.add_all(detection_objects)
        session.commit()
        session.close()

    def update_tracks(self, track_messages: List[Dict[str, Any]]) -> None:
        """Efficiently updates the database with tracks."""
        session = self.SessionFactory

        # Extract track IDs and detection IDs for bulk querying
        track_ids = {tm["track_id"] for tm in track_messages}
        detection_map = {tm["track_id"]: tm.pop("detection_ids") for tm in track_messages}

        # Pre-fetch existing tracks in one query
        existing_tracks = {
            t.track_id: t for t in session.execute(select(Track).where(Track.track_id.in_(track_ids))).scalars().all()
        }  # Converts list to a dictionary for O(1) lookups

        # Batch insert/update tracks
        new_tracks = []
        for track_message in track_messages:
            track_id = track_message["track_id"]
            track_message["class_name"] = self.class_names[track_message.pop("class_id")]

            if track_id in existing_tracks:
                track = existing_tracks[track_id]
                track.count = track_message["count"]
                track.is_activated = track_message["is_activated"]
                track.state = track_message["state"]
                track.score = track_message["score"]
                track.curr_frame_number = track_message["curr_frame_number"]
                track.time_since_update = track_message["time_since_update"]
                track.location = track_message["location"]
                track.class_name = track_message["class_name"]
            else:
                new_tracks.append(Track(**track_message, job_id=self.job_id))

        if new_tracks:
            session.add_all(new_tracks)  # Bulk insert new tracks

        # Pre-fetch detections in one query
        detection_ids = {d_id for d_list in detection_map.values() for d_id in d_list}
        existing_detections = {
            d.detection_id: d
            for d in session.execute(select(Detection).where(Detection.detection_id.in_(detection_ids))).scalars().all()
        }  # Converts list to a dictionary for O(1) lookups

        # Update detection track IDs in bulk
        for track_id, detection_ids in detection_map.items():
            for detection_id in detection_ids:
                if detection_id in existing_detections:
                    existing_detections[detection_id].track_id = track_id

        session.commit()
        session.close()

    def add_frame(self, frame: np.ndarray, frame_number: int) -> None:
        """Adds a frame to the database Frames table.

        Args:
            frame: frame to add.
        """
        session = self.SessionFactory
        image_base64 = encode_image(frame)
        session.add(
            Frame(
                frame_number=frame_number,
                timestamp=datetime.datetime.now(datetime.timezone.utc),
                image_base64=image_base64,
                image_caption=generate_frame_caption(image_base64) if self.create_image_captions else None,
                job_id=self.job_id,
            )
        )
        session.commit()
        session.close()
