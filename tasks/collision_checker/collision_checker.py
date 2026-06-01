#!/usr/bin/env python3
"""
Assignment 6: Collision Checker

Reads circles and rectangles from a text file, checks all unique pairs for
collision, visualizes the scene, and prints colliding pairs (no duplicates).

Rectangle geometry (assignment spec):
  - (x, y) is the top-left corner before rotation
  - width extends to the right (+x), height extends downward (-y in math coords)
  - rotation is counter-clockwise in degrees about the top-left corner
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.patches import Circle as MplCircle
from matplotlib.patches import Polygon as MplPolygon

Point = Tuple[float, float]
Object = Union["Circle", "Rectangle"]


@dataclass
class Circle:
    name: str
    x: float
    y: float
    radius: float

    @property
    def kind(self) -> str:
        return "circle"


@dataclass
class Rectangle:
    name: str
    x: float
    y: float
    width: float
    height: float
    angle_deg: float

    @property
    def kind(self) -> str:
        return "rectangle"


def parse_objects(path: str) -> List[Object]:
    """Parse object definitions from the assignment file format."""
    objects: List[Object] = []

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("!"):
                continue

            parts = line.split()
            kind = parts[0].lower()

            if kind == "circle":
                if len(parts) != 5:
                    raise ValueError(f"Invalid circle line: {raw_line!r}")
                _, name, x, y, radius = parts
                objects.append(
                    Circle(name=name, x=float(x), y=float(y), radius=float(radius))
                )
            elif kind == "rectangle":
                if len(parts) != 7:
                    raise ValueError(f"Invalid rectangle line: {raw_line!r}")
                _, name, x, y, w, h, angle = parts
                objects.append(
                    Rectangle(
                        name=name,
                        x=float(x),
                        y=float(y),
                        width=float(w),
                        height=float(h),
                        angle_deg=float(angle),
                    )
                )
            else:
                raise ValueError(f"Unknown object type: {kind}")

    return objects


def rectangle_vertices(rect: Rectangle) -> List[Point]:
    """
    World-space vertices of a rotated rectangle.
    Local corners relative to top-left pivot (0,0), (w,0), (w,-h), (0,-h).
    """
    theta = math.radians(rect.angle_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    local_corners = [
        (0.0, 0.0),
        (rect.width, 0.0),
        (rect.width, -rect.height),
        (0.0, -rect.height),
    ]

    vertices: List[Point] = []
    for lx, ly in local_corners:
        # rotation matrix: [cos_t, -sin_t; sin_t, cos_t]
        wx = rect.x + lx * cos_t - ly * sin_t
        wy = rect.y + lx * sin_t + ly * cos_t
        vertices.append((wx, wy))
    return vertices


def _normalize(axis: Point) -> Point:
    length = math.hypot(axis[0], axis[1])
    if length < 1e-12:
        return (0.0, 0.0)
    return (axis[0] / length, axis[1] / length)


#returns interval of projected vertices on the axis
def _project_polygon(vertices: Sequence[Point], axis: Point) -> Tuple[float, float]:
    dots = [v[0] * axis[0] + v[1] * axis[1] for v in vertices]
    return min(dots), max(dots)


def _intervals_overlap(a_min: float, a_max: float, b_min: float, b_max: float) -> bool:
    return not (a_max < b_min or b_max < a_min)


#returns list of axes for the polygon, which are the normals of the edges
def _sat_axes(vertices: Sequence[Point]) -> List[Point]:
    axes: List[Point] = []
    n = len(vertices)
    for i in range(n):
        x1, y1 = vertices[i]
        #% because the polygon is closed, we need to wrap around to the first vertex
        x2, y2 = vertices[(i + 1) % n]
        edge = (x2 - x1, y2 - y1)
        normal = (-edge[1], edge[0])
        axis = _normalize(normal)
        if axis != (0.0, 0.0):
            axes.append(axis)
    return axes


def collides_circle_circle(a: Circle, b: Circle) -> bool:
    """Two circles collide when center distance <= sum of radii."""
    dx = a.x - b.x
    dy = a.y - b.y
    reach = a.radius + b.radius
    return dx * dx + dy * dy <= reach * reach


def _point_in_polygon(x: float, y: float, vertices: Sequence[Point]) -> bool:
    """Ray-casting point-in-polygon test."""
    inside = False
    n = len(vertices)
    j = n - 1
    for i in range(n):
        xi, yi = vertices[i]
        xj, yj = vertices[j]
        #ray-casting test to check if the point is on the same side of the edge
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def _closest_point_on_segment(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> Point:
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    #squared length of the edge/segment
    ab_len_sq = abx * abx + aby * aby
    if ab_len_sq < 1e-12:   
        #if the segment is a point, return the point
        return (ax, ay)
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab_len_sq))
    return (ax + t * abx, ay + t * aby)


def collides_circle_rectangle(circle: Circle, rect: Rectangle) -> bool:
    """
    Circle vs rotated rectangle:
    collide if circle center is inside rectangle OR closest edge distance <= radius.
    """
    verts = rectangle_vertices(rect)

    if _point_in_polygon(circle.x, circle.y, verts):
        return True

    #find the closest point on the edges of the rectangle to the circle center
    min_dist_sq = float("inf")
    n = len(verts)
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        qx, qy = _closest_point_on_segment(circle.x, circle.y, x1, y1, x2, y2)
        dist_sq = (circle.x - qx) ** 2 + (circle.y - qy) ** 2
        min_dist_sq = min(min_dist_sq, dist_sq)

    return min_dist_sq <= circle.radius * circle.radius


def collides_rectangle_rectangle(a: Rectangle, b: Rectangle) -> bool:
    """Rotated rectangle vs rotated rectangle using Separating Axis Theorem (SAT)."""
    verts_a = rectangle_vertices(a)
    verts_b = rectangle_vertices(b)

    axes = _sat_axes(verts_a) + _sat_axes(verts_b)
    for axis in axes:
        min_a, max_a = _project_polygon(verts_a, axis)
        min_b, max_b = _project_polygon(verts_b, axis)
        if not _intervals_overlap(min_a, max_a, min_b, max_b):
            return False
    return True


def objects_collide(a: Object, b: Object) -> bool:
    if isinstance(a, Circle) and isinstance(b, Circle):
        return collides_circle_circle(a, b)
    if isinstance(a, Circle) and isinstance(b, Rectangle):
        return collides_circle_rectangle(a, b)
    if isinstance(a, Rectangle) and isinstance(b, Circle):
        return collides_circle_rectangle(b, a)
    if isinstance(a, Rectangle) and isinstance(b, Rectangle):
        return collides_rectangle_rectangle(a, b)
    raise TypeError("Unsupported object types")


def find_collisions(objects: Sequence[Object]) -> List[Tuple[str, str]]:
    """Return unique colliding pairs (i < j), sorted by name."""
    pairs: List[Tuple[str, str]] = []
    n = len(objects)
    for i in range(n):
        for j in range(i + 1, n):
            if objects_collide(objects[i], objects[j]):
                pairs.append((objects[i].name, objects[j].name))
    return pairs


def visualize(objects: Sequence[Object], colliding_names: set[str]) -> None:
    """Draw all objects; colliding ones are highlighted in red."""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_title("Collision Visualization")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)

    for obj in objects:
        colliding = obj.name in colliding_names
        if isinstance(obj, Circle):
            color = "red" if colliding else "blue"
            patch = MplCircle(
                (obj.x, obj.y),
                obj.radius,
                facecolor=color,
                edgecolor="black",
                alpha=0.45,
                linewidth=1.2,
            )
            ax.add_patch(patch)
            ax.text(obj.x, obj.y, obj.name, ha="center", va="center", fontsize=8)
        else:
            color = "red" if colliding else "gold"
            verts = rectangle_vertices(obj)
            patch = MplPolygon(
                verts,
                closed=True,
                facecolor=color,
                edgecolor="black",
                alpha=0.45,
                linewidth=1.2,
            )
            ax.add_patch(patch)
            cx = sum(v[0] for v in verts) / 4.0
            cy = sum(v[1] for v in verts) / 4.0
            ax.text(cx, cy, obj.name, ha="center", va="center", fontsize=8)

    all_x: List[float] = []
    all_y: List[float] = []
    for obj in objects:
        if isinstance(obj, Circle):
            all_x.extend([obj.x - obj.radius, obj.x + obj.radius])
            all_y.extend([obj.y - obj.radius, obj.y + obj.radius])
        else:
            for vx, vy in rectangle_vertices(obj):
                all_x.append(vx)
                all_y.append(vy)

    margin = 30.0
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    plt.show()


def main() -> None:
    input_path = os.path.join(os.path.dirname(__file__), "objects.txt")
    objects = parse_objects(input_path)

    pairs = find_collisions(objects)
    colliding_names = {name for pair in pairs for name in pair}

    visualize(objects, colliding_names)

    for a, b in pairs:
        print(f"{a} {b}")


if __name__ == "__main__":
    main()
