"""
ML policy helpers for flight rebooking.

This module provides:
- deterministic expert policy used for dataset generation,
- fixed-length feature extraction for supervised learning,
- safe action construction from ranked action-type preferences.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from environment import ActionType, CabinClass, PriorityTier


ACTION_TYPE_ORDER: List[str] = [
    ActionType.REBOOK_PASSENGER.value,
    ActionType.OFFER_DOWNGRADE.value,
    ActionType.REBOOK_ON_PARTNER.value,
    ActionType.BOOK_HOTEL.value,
    ActionType.MARK_NO_SOLUTION.value,
    ActionType.FINALIZE.value,
]


def _tier_weight(tier: str) -> int:
    return {
        PriorityTier.PLATINUM.value: 4,
        PriorityTier.GOLD.value: 3,
        PriorityTier.SILVER.value: 2,
        PriorityTier.STANDARD.value: 1,
    }.get(tier, 1)


def _deadline_sort_value(deadline_hrs: Optional[float]) -> float:
    return float(deadline_hrs) if deadline_hrs is not None else 10**9


def _has_seat(flight: Dict[str, Any], cabin_class: str) -> bool:
    if cabin_class == CabinClass.BUSINESS.value:
        return int(flight["business_seats"]) > 0
    return int(flight["economy_seats"]) > 0


def _sorted_pending_passengers(observation: Dict[str, Any]) -> List[Dict[str, Any]]:
    pending = list(observation.get("pending_passengers", []))
    pending.sort(
        key=lambda p: (
            -_tier_weight(str(p.get("priority_tier", ""))),
            _deadline_sort_value(p.get("connection_deadline_hrs")),
        )
    )
    return pending


def _sorted_flights(observation: Dict[str, Any]) -> List[Dict[str, Any]]:
    flights = list(observation.get("available_flights", []))
    flights.sort(key=lambda f: float(f.get("departure_hrs", 10**9)))
    return flights


def heuristic_action(observation: Dict[str, Any]) -> Dict[str, Any]:
    pending = _sorted_pending_passengers(observation)
    if not pending:
        return {"action_type": ActionType.FINALIZE.value}

    passenger = pending[0]
    flights = _sorted_flights(observation)

    for flight in flights:
        if flight.get("is_partner", False):
            continue
        if _has_seat(flight, str(passenger["cabin_class"])):
            return {
                "action_type": ActionType.REBOOK_PASSENGER.value,
                "passenger_id": passenger["id"],
                "flight_id": flight["id"],
            }

    if passenger.get("cabin_class") == CabinClass.BUSINESS.value:
        for flight in flights:
            if flight.get("is_partner", False):
                continue
            if int(flight.get("economy_seats", 0)) > 0 and float(observation.get("budget_remaining", 0.0)) >= 500.0:
                return {
                    "action_type": ActionType.OFFER_DOWNGRADE.value,
                    "passenger_id": passenger["id"],
                    "flight_id": flight["id"],
                }

    for flight in flights:
        if not flight.get("is_partner", False):
            continue
        if _has_seat(flight, str(passenger["cabin_class"])) and float(observation.get("budget_remaining", 0.0)) >= 800.0:
            return {
                "action_type": ActionType.REBOOK_ON_PARTNER.value,
                "passenger_id": passenger["id"],
                "flight_id": flight["id"],
            }

    if float(observation.get("budget_remaining", 0.0)) >= 250.0:
        return {
            "action_type": ActionType.BOOK_HOTEL.value,
            "passenger_id": passenger["id"],
        }

    return {
        "action_type": ActionType.MARK_NO_SOLUTION.value,
        "passenger_id": passenger["id"],
    }


def observation_to_features(
    observation: Dict[str, Any],
    max_pending: int = 5,
    max_flights: int = 6,
) -> List[float]:
    pending = _sorted_pending_passengers(observation)
    flights = _sorted_flights(observation)

    processed_count = float(observation.get("processed_count", 0))
    total_passengers = max(float(observation.get("total_passengers", 1)), 1.0)
    budget_remaining = max(float(observation.get("budget_remaining", 0.0)), 0.0)
    budget_spent = max(float(observation.get("budget_spent", 0.0)), 0.0)
    budget_total = max(budget_remaining + budget_spent, 1.0)

    features: List[float] = []

    features.extend(
        [
            min(len(pending), 20) / 20.0,
            min(len(flights), 20) / 20.0,
            budget_remaining / budget_total,
            budget_spent / budget_total,
            processed_count / total_passengers,
            min(float(observation.get("invalid_actions", 0)), 20.0) / 20.0,
            min(float(observation.get("step_count", 0)), 120.0) / 120.0,
            1.0 if pending else 0.0,
        ]
    )

    for passenger in pending[:max_pending]:
        deadline = passenger.get("connection_deadline_hrs")
        has_deadline = 1.0 if deadline is not None else 0.0
        deadline_norm = (min(float(deadline), 12.0) / 12.0) if deadline is not None else 1.0
        features.extend(
            [
                _tier_weight(str(passenger.get("priority_tier", ""))) / 4.0,
                1.0 if passenger.get("cabin_class") == CabinClass.BUSINESS.value else 0.0,
                has_deadline,
                deadline_norm,
            ]
        )

    for _ in range(max_pending - len(pending[:max_pending])):
        features.extend([0.0, 0.0, 0.0, 0.0])

    for flight in flights[:max_flights]:
        features.extend(
            [
                1.0 if flight.get("is_partner", False) else 0.0,
                min(float(flight.get("departure_hrs", 12.0)), 12.0) / 12.0,
                min(float(flight.get("economy_seats", 0.0)), 12.0) / 12.0,
                min(float(flight.get("business_seats", 0.0)), 6.0) / 6.0,
            ]
        )

    for _ in range(max_flights - len(flights[:max_flights])):
        features.extend([0.0, 0.0, 0.0, 0.0])

    same_econ = 0.0
    same_bus = 0.0
    partner_econ = 0.0
    partner_bus = 0.0
    for flight in flights:
        if flight.get("is_partner", False):
            partner_econ += float(flight.get("economy_seats", 0.0))
            partner_bus += float(flight.get("business_seats", 0.0))
        else:
            same_econ += float(flight.get("economy_seats", 0.0))
            same_bus += float(flight.get("business_seats", 0.0))

    features.extend(
        [
            min(same_econ, 30.0) / 30.0,
            min(same_bus, 20.0) / 20.0,
            min(partner_econ, 30.0) / 30.0,
            min(partner_bus, 20.0) / 20.0,
        ]
    )

    return features


def build_feasible_action_for_type(observation: Dict[str, Any], action_type: str) -> Optional[Dict[str, Any]]:
    pending = _sorted_pending_passengers(observation)
    flights = _sorted_flights(observation)
    budget_remaining = float(observation.get("budget_remaining", 0.0))

    if not pending:
        return {"action_type": ActionType.FINALIZE.value}

    if action_type == ActionType.FINALIZE.value:
        return None

    if action_type == ActionType.BOOK_HOTEL.value:
        if budget_remaining >= 250.0:
            return {
                "action_type": ActionType.BOOK_HOTEL.value,
                "passenger_id": pending[0]["id"],
            }
        return None

    if action_type == ActionType.MARK_NO_SOLUTION.value:
        return {
            "action_type": ActionType.MARK_NO_SOLUTION.value,
            "passenger_id": pending[0]["id"],
        }

    if action_type == ActionType.REBOOK_PASSENGER.value:
        for passenger in pending:
            for flight in flights:
                if flight.get("is_partner", False):
                    continue
                if _has_seat(flight, str(passenger["cabin_class"])):
                    return {
                        "action_type": ActionType.REBOOK_PASSENGER.value,
                        "passenger_id": passenger["id"],
                        "flight_id": flight["id"],
                    }
        return None

    if action_type == ActionType.OFFER_DOWNGRADE.value:
        if budget_remaining < 500.0:
            return None
        business_pending = [p for p in pending if p.get("cabin_class") == CabinClass.BUSINESS.value]
        for passenger in business_pending:
            for flight in flights:
                if flight.get("is_partner", False):
                    continue
                if int(flight.get("economy_seats", 0)) > 0:
                    return {
                        "action_type": ActionType.OFFER_DOWNGRADE.value,
                        "passenger_id": passenger["id"],
                        "flight_id": flight["id"],
                    }
        return None

    if action_type == ActionType.REBOOK_ON_PARTNER.value:
        if budget_remaining < 800.0:
            return None
        for passenger in pending:
            for flight in flights:
                if not flight.get("is_partner", False):
                    continue
                if _has_seat(flight, str(passenger["cabin_class"])):
                    return {
                        "action_type": ActionType.REBOOK_ON_PARTNER.value,
                        "passenger_id": passenger["id"],
                        "flight_id": flight["id"],
                    }
        return None

    return None


def choose_action_from_ranked_types(observation: Dict[str, Any], ranked_types: Iterable[str]) -> Dict[str, Any]:
    for action_type in ranked_types:
        if not action_type:
            continue
        candidate = build_feasible_action_for_type(observation, str(action_type))
        if candidate is not None:
            return candidate

    return heuristic_action(observation)
