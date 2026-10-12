from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SolverMethod:
    """Paper-derived metadata for one family-scoped solver-bank entry.

    ``codename`` preserves the bare name printed in the paper. Some bare names occur in
    more than one PDE family, so ``method_id`` is the unique catalog key. This record does
    not assert that a clean-room implementation exists.
    """

    codename: str
    family_id: str
    scheme: str
    integrator: str
    stability_control: str
    citation_keys: tuple[str, ...]
    expected_order: str
    fallback: str | None
    implementation_version: str
    ambiguity_flags: tuple[str, ...]

    def __post_init__(self) -> None:
        required = {
            "codename": self.codename,
            "family_id": self.family_id,
            "scheme": self.scheme,
            "integrator": self.integrator,
            "stability_control": self.stability_control,
            "expected_order": self.expected_order,
            "implementation_version": self.implementation_version,
        }
        for field_name, value in required.items():
            if not value.strip():
                raise ValueError(f"{field_name} must not be empty")
        if not self.citation_keys or any(not key.strip() for key in self.citation_keys):
            raise ValueError("citation_keys must contain at least one non-empty key")
        if len(set(self.citation_keys)) != len(self.citation_keys):
            raise ValueError("citation_keys must not contain duplicates")
        if len(set(self.ambiguity_flags)) != len(self.ambiguity_flags):
            raise ValueError("ambiguity_flags must not contain duplicates")

    @property
    def method_id(self) -> str:
        """Return the unique family-scoped catalog identifier."""

        return f"{self.family_id}/{self.codename}"

    @property
    def family(self) -> str:
        """Return the PDE family using the short field name from the method contract."""

        return self.family_id
