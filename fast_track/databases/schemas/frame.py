"""Frame schema"""

from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship, Mapped

from . import Base, Job


class Frame(Base):
    """Frame schema"""

    __tablename__ = "frames"
    frame_id = Column(Integer, primary_key=True)
    frame_number = Column(Integer, nullable=False)
    time_created = Column(String, nullable=False)
    frame_base64 = Column(String, nullable=False)
    image_caption = Column(String, nullable=True)
    job_id = Column(Integer, ForeignKey("jobs.job_id"))
    job: Mapped["Job"] = relationship(back_populates="frames")
