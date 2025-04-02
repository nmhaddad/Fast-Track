"""Job schema"""

from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship

from .base import Base


class Job(Base):
    """Job schema"""

    __tablename__ = "jobs"
    job_id = Column(Integer, primary_key=True)
    job_name = Column(String, nullable=False)
    detections = relationship("Detection", back_populates="job", cascade="all, delete, delete-orphan")
    tracks = relationship("Track", back_populates="job", cascade="all, delete, delete-orphan")
    frames = relationship("Frame", back_populates="job", cascade="all, delete, delete-orphan")
