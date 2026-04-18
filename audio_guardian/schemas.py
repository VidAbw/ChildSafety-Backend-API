from pydantic import BaseModel, Field


class AudioFrameSchema(BaseModel):
    user_id: str = "default_user"
    time: str | float | int | None = None
    mfcc: list[list[float]] = Field(default_factory=list)
    trauma_threshold: float = 0.8
    duration_threshold: float = 0.6
