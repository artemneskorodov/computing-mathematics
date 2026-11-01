"""Geometrical objects for the room thermal editor (distances in metres)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path


@dataclass
class Window:
    """Window section along the parent wall, in normalized wall coordinates."""
    centre: float
    length: float  # metres along wall


@dataclass
class Wall:
    x1: float
    y1: float
    x2: float
    y2: float
    thickness: float = 0.25
    windows: list[Window] = field(default_factory=list)


@dataclass
class Heater:
    x: float
    y: float
    radius: float = 0.35
    temperature: float = 40.0


@dataclass
class Project:
    width: float = 8.0
    height: float = 6.0
    step: float = 0.10
    outdoor_temperature: float = -10.0
    # This is deliberately an effective 2D conductivity, not a CFD model.
    air_conductivity: float = 0.10
    concrete_conductivity: float = 1.40
    window_conductivity: float = 0.80
    walls: list[Wall] = field(default_factory=list)
    heaters: list[Heater] = field(default_factory=list)
    street_seed: tuple[float, float] | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> Project:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        project = cls(
            **{key: data[key] for key in (
                "width", "height", "step", "outdoor_temperature",
                "air_conductivity", "concrete_conductivity", "window_conductivity",
            ) if key in data}
        )
        project.walls = [
            Wall(**{**wall, "windows": [Window(**w) for w in wall.get("windows", [])]})
            for wall in data.get("walls", [])
        ]
        project.heaters = [Heater(**heater) for heater in data.get("heaters", [])]
        project.street_seed = (
            tuple(data["street_seed"]) if data.get("street_seed") is not None else None
        )
        return project


def example_project() -> Project:
    project = Project()
    project.walls = [
        Wall(1.2, 1.0, 6.8, 1.0, windows=[Window(0.5, 1.5)]),
        Wall(6.8, 1.0, 6.8, 5.0, windows=[Window(0.55, 1.0)]),
        Wall(6.8, 5.0, 1.2, 5.0),
        Wall(1.2, 5.0, 1.2, 1.0),
    ]
    project.heaters = [Heater(3.2, 3.0, 0.55, 40.0)]
    project.street_seed = (0.4, 0.4)
    return project
