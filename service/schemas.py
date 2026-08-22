"""Pydantic request/response models matching contracts/openapi_notes.md exactly.
No undocumented params -- /docs must render exactly these four routes' shapes."""

from typing import Literal

from pydantic import BaseModel

Action = Literal["completed", "liked", "playlist", "skip", "replay"]


class RecommendRequest(BaseModel):
    user_id: str


class RecommendResponse(BaseModel):
    track_id: str


class FeedbackRequest(BaseModel):
    user_id: str
    track_id: str
    action: Action


class FeedbackResponse(BaseModel):
    status: str = "ok"


class HealthResponse(BaseModel):
    status: str = "ok"


class MetricsResponse(BaseModel):
    ctr: float
    avg_reward: float
    cumulative_regret: float
    total_recommendations: int
