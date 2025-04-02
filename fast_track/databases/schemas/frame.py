"""Frame schema"""

import datetime

from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, relationship
from sqlalchemy.sql.sqltypes import DateTime

from . import Base, Job


class Frame(Base):
    """Frame schema"""

    __tablename__ = "frames"
    id = Column(Integer, primary_key=True)
    frame_id = Column(Integer, primary_key=True)
    frame_number = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.timezone.utc)
    image_base64 = Column(String, nullable=False)
    image_caption = Column(String, nullable=True)
    job_id = Column(Integer, ForeignKey("jobs.job_id"))
    job: Mapped["Job"] = relationship(back_populates="frames")
