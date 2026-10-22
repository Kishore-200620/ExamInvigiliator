"""Small, dependency-free helpers shared by the live video overlay and tests."""


def count_active_violations(students: list[dict]) -> int:
    """Count active rule labels from the current per-student detection payload."""
    count = 0
    for student in students:
        if not isinstance(student, dict):
            continue
        active = student.get("active_violations") or []
        active_count = len(active) if isinstance(active, (list, tuple, set)) else 0
        count += max(active_count, int(bool(student.get("violation_active"))))
    return count
