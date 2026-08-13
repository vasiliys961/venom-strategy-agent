"""
Pydantic-модель состояния VENOM Canvas и состояния диалога.
Каждый этап метода добавляет свои поля; финальная сборка использует их все.
"""
from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field

Stage = Literal[
    "vision", "evaluation", "gaps", "objectives", "management", "assembly", "done"
]


class SmartObjective(BaseModel):
    title: str
    horizon: str  # напр. "2027 Q4"
    metric: str
    first_step: str


class VenomCanvas(BaseModel):
    user_id: int

    # V — Vision
    vision_10y: Optional[str] = None
    desired_future: Optional[str] = None
    core_values: list[str] = Field(default_factory=list)

    # E — Evaluation
    life_spheres: dict[str, str] = Field(default_factory=dict)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    environment_notes: Optional[str] = None

    # N — Narrow gaps (стратегические разрывы)
    stability_gaps: list[str] = Field(default_factory=list)
    growth_gaps: list[str] = Field(default_factory=list)
    root_causes: dict[str, str] = Field(default_factory=dict)

    # O — Objectives
    strategic_goals: list[str] = Field(default_factory=list)
    smart_objectives: list[SmartObjective] = Field(default_factory=list)

    # M — Management
    habits_to_build: list[str] = Field(default_factory=list)
    retrospective_cadence: Optional[str] = None
    management_system: Optional[str] = None

    # meta
    stage: Stage = "vision"
    retrospective_notes: list[str] = Field(default_factory=list)
    needs_revision_of: Optional[Stage] = None

    def is_stage_complete(self, stage: Stage) -> bool:
        checks = {
            "vision": bool(self.vision_10y and self.desired_future),
            "evaluation": bool(self.life_spheres and (self.strengths or self.weaknesses)),
            "gaps": bool(self.stability_gaps or self.growth_gaps),
            "objectives": bool(self.smart_objectives),
            "management": bool(self.management_system),
        }
        return checks.get(stage, True)
