"""Detection schemas"""

import datetime

from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, relationship
from sqlalchemy.sql.sqltypes import JSON, DateTime

from . import Base, Job


class Detection(Base):
    """Detection schema"""

    __tablename__ = "detections"
    id = Column(Integer, primary_key=True)
    detection_id = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.timezone.utc)
    class_id = Column(Integer, nullable=False)
    class_name = Column(String, nullable=False)
    bbox = Column(JSON, nullable=False)
    confidence = Column(Integer, nullable=False)
    frame_number = Column(Integer, nullable=False)
    track_id = Column(Integer, ForeignKey("tracks.id"), nullable=True)
    track = relationship("Track", back_populates="detections")
    job_id = Column(Integer, ForeignKey("jobs.job_id"))
    job: Mapped["Job"] = relationship(back_populates="frames")
