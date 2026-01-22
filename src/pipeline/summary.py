from typing import Dict, List


def _best_view(view_preds: List[Dict]) -> str:
    if not view_preds:
        return "Unknown view"
    best = max(view_preds, key=lambda x: x.get("confidence", 0.0))
    return best.get("view", "Unknown view")


def make_summary(view_preds: List[Dict], damage: str, severity: str) -> str:
    if damage == "unknown" and severity == "unknown":
        return "Insufficient evidence to assess damage."
    if damage == "none" and severity == "none":
        return "No visible damage detected."
    view = _best_view(view_preds)
    if severity == "unknown":
        return f"{view.title()} view shows {damage} damage."
    return f"{view.title()} view shows {severity} {damage} damage."
