from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, StringConstraints
from typing import Annotated

Label = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]


class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: FiniteFloat = Field(ge=0)
    screen_hours: FiniteFloat = Field(ge=0)
    apps_installed: int = Field(ge=0)
    plan_gb_limit: FiniteFloat = Field(ge=0)
    video_minutes: FiniteFloat = Field(ge=0)
    wifi_hours: FiniteFloat = Field(ge=0)
    phone_tier: Label
    circle: Label


class PredictionOutput(BaseModel):
    gb_per_day: FiniteFloat
    model_version: str
