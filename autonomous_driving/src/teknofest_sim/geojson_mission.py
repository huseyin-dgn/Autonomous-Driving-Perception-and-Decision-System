import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class MissionPoint:
    name: str
    description: str
    nokta_id: int
    lon: float
    lat: float
    yaw: Optional[float] = None


@dataclass
class MissionSpec:
    round_name: str
    start: MissionPoint
    task_points: List[MissionPoint]
    park_entry: MissionPoint
    raw_points: Dict[str, MissionPoint]


def _point_from_feature(feature: dict) -> MissionPoint:
    props = feature.get("properties", {})
    geom = feature.get("geometry", {})
    coords = geom.get("coordinates", [])

    if geom.get("type") != "Point":
        raise ValueError(f"Sadece Point geometry destekleniyor: {geom.get('type')}")

    if len(coords) < 2:
        raise ValueError("GEOJSON Point coordinates [lon, lat] formatında olmalı.")

    return MissionPoint(
        name=str(props.get("name", "")).strip(),
        description=str(props.get("description", "")).strip(),
        nokta_id=int(props.get("nokta_id", 0)),
        lon=float(coords[0]),
        lat=float(coords[1]),
        yaw=float(props["yaw"]) if "yaw" in props and props["yaw"] is not None else None,
    )


def load_mission_geojson(path: str, round_name: str = "round_3") -> MissionSpec:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("type") != "FeatureCollection":
        raise ValueError("GEOJSON kökü FeatureCollection olmalı.")

    points: Dict[str, MissionPoint] = {}

    for feature in data.get("features", []):
        point = _point_from_feature(feature)
        if not point.name:
            raise ValueError("Her feature properties.name içermeli.")
        points[point.name] = point

    if "start" not in points:
        raise ValueError("GEOJSON içinde name=start bulunmalı.")

    if "park_giris" not in points:
        raise ValueError("GEOJSON içinde name=park_giris bulunmalı.")

    task_points = [
        p for name, p in sorted(points.items(), key=lambda item: item[1].nokta_id)
        if name.startswith("gorev_") or name.startswith("passenger_")
    ]

    if not task_points:
        task_points = [
            p for name, p in sorted(points.items(), key=lambda item: item[1].nokta_id)
            if name not in {"start", "park_giris"}
        ]

    return MissionSpec(
        round_name=round_name,
        start=points["start"],
        task_points=task_points,
        park_entry=points["park_giris"],
        raw_points=points,
    )


def haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371000.0

    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)

    a = (
        math.sin(dp / 2.0) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    )

    return 2.0 * radius * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def mission_to_dict(mission: MissionSpec) -> dict:
    def p2d(p: MissionPoint):
        return {
            "name": p.name,
            "description": p.description,
            "nokta_id": p.nokta_id,
            "lat": p.lat,
            "lon": p.lon,
            "yaw": p.yaw,
        }

    return {
        "round_name": mission.round_name,
        "start": p2d(mission.start),
        "task_points": [p2d(p) for p in mission.task_points],
        "park_entry": p2d(mission.park_entry),
    }