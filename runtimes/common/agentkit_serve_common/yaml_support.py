"""Lossless safe YAML loading for JSON-compatible numeric configuration."""

from __future__ import annotations

import math
from decimal import Decimal, InvalidOperation
from typing import Any

import yaml

_MAX_EXACT_YAML_FLOAT_INTEGER_DIGITS = 4096
_MAX_EXACT_JSON_FLOAT_INTEGER = (1 << 53) - 1


class _LosslessSafeLoader(yaml.SafeLoader):
    """SafeLoader variant that never silently rounds YAML float scalars."""


def _construct_lossless_float(loader: yaml.SafeLoader, node: yaml.Node) -> int | float:
    raw = loader.construct_scalar(node).replace("_", "").lower()
    if ":" in raw:
        raise ValueError(f"YAML float literal {raw!r} uses unsupported sexagesimal notation")
    try:
        decimal = Decimal(raw)
    except InvalidOperation as exc:
        raise ValueError(f"YAML float literal {raw!r} is invalid") from exc
    if not decimal.is_finite():
        raise ValueError(f"YAML float literal {raw!r} must be finite")
    if decimal.is_zero() and decimal.is_signed():
        return -0.0
    if decimal == decimal.to_integral_value():
        digits = max(decimal.adjusted() + 1, len(decimal.as_tuple().digits)) if decimal else 1
        if digits > _MAX_EXACT_YAML_FLOAT_INTEGER_DIGITS:
            raise ValueError(f"YAML float literal {raw!r} expands to an integer that is too large")
        if abs(decimal) <= _MAX_EXACT_JSON_FLOAT_INTEGER:
            return float(decimal)
        return int(decimal)
    try:
        candidate = float(decimal)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"YAML float literal {raw!r} cannot be represented exactly") from exc
    if not math.isfinite(candidate) or Decimal(str(candidate)) != decimal:
        raise ValueError(f"YAML float literal {raw!r} cannot be represented exactly")
    return candidate


_LosslessSafeLoader.add_constructor("tag:yaml.org,2002:float", _construct_lossless_float)


def safe_load_lossless(raw: str) -> Any:
    """Load one YAML document without lossy float coercion."""

    return yaml.load(raw, Loader=_LosslessSafeLoader)
