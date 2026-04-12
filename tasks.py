"""
Task Definitions and Deterministic Graders
===========================================

Provides three progressively difficult real-world disruption tasks and
task-specific grading functions that return normalized scores in [0.0, 1.0].
"""

from typing import Dict, List

from environment import EnvState, PassengerStatus, PriorityTier


EASY_TASK = {
    "task_id": "easy_minor_disruption",
    "difficulty": "easy",
    "objective": "Rebook all passengers on same-airline flights while preserving premium service and minimizing spend.",
    "max_budget": 3000,
    "max_steps": 40,
    "passengers": [
        {
            "id": "P1",
            "name": "Alice Johnson",
            "priority_tier": "Platinum",
            "original_flight": "FL-100",
            "cabin_class": "Business",
            "connection_deadline_hrs": None,
        },
        {
            "id": "P2",
            "name": "Bob Smith",
            "priority_tier": "Gold",
            "original_flight": "FL-100",
            "cabin_class": "Economy",
            "connection_deadline_hrs": None,
        },
        {
            "id": "P3",
            "name": "Carol Davis",
            "priority_tier": "Standard",
            "original_flight": "FL-100",
            "cabin_class": "Economy",
            "connection_deadline_hrs": None,
        },
    ],
    "flights": [
        {
            "id": "FL-102",
            "destination": "New York",
            "departure_hrs": 3.0,
            "economy_seats": 5,
            "business_seats": 2,
            "is_partner": False,
        },
        {
            "id": "FL-104",
            "destination": "New York",
            "departure_hrs": 6.0,
            "economy_seats": 10,
            "business_seats": 3,
            "is_partner": False,
        },
        {
            "id": "FL-201",
            "destination": "New York",
            "departure_hrs": 4.0,
            "economy_seats": 3,
            "business_seats": 1,
            "is_partner": True,
        },
    ],
}


MEDIUM_TASK = {
    "task_id": "medium_connection_crisis",
    "difficulty": "medium",
    "objective": "Prioritize high-tier passengers with tight deadlines under constrained seats and budget.",
    "max_budget": 5000,
    "max_steps": 60,
    "passengers": [
        {
            "id": "P1",
            "name": "David Lee",
            "priority_tier": "Platinum",
            "original_flight": "FL-300",
            "cabin_class": "Business",
            "connection_deadline_hrs": 4.0,
        },
        {
            "id": "P2",
            "name": "Emma Wilson",
            "priority_tier": "Gold",
            "original_flight": "FL-300",
            "cabin_class": "Economy",
            "connection_deadline_hrs": 2.5,
        },
        {
            "id": "P3",
            "name": "Frank Brown",
            "priority_tier": "Silver",
            "original_flight": "FL-300",
            "cabin_class": "Economy",
            "connection_deadline_hrs": None,
        },
        {
            "id": "P4",
            "name": "Grace Kim",
            "priority_tier": "Standard",
            "original_flight": "FL-300",
            "cabin_class": "Business",
            "connection_deadline_hrs": 5.0,
        },
        {
            "id": "P5",
            "name": "Henry Park",
            "priority_tier": "Gold",
            "original_flight": "FL-300",
            "cabin_class": "Economy",
            "connection_deadline_hrs": 3.0,
        },
    ],
    "flights": [
        {
            "id": "FL-302",
            "destination": "Chicago",
            "departure_hrs": 2.0,
            "economy_seats": 2,
            "business_seats": 1,
            "is_partner": False,
        },
        {
            "id": "FL-304",
            "destination": "Chicago",
            "departure_hrs": 5.0,
            "economy_seats": 4,
            "business_seats": 0,
            "is_partner": False,
        },
        {
            "id": "FL-401",
            "destination": "Chicago",
            "departure_hrs": 3.5,
            "economy_seats": 2,
            "business_seats": 1,
            "is_partner": True,
        },
    ],
}


HARD_TASK = {
    "task_id": "hard_multi_wave_disruption",
    "difficulty": "hard",
    "objective": "Handle mixed loyalty tiers, scarce seats, and multiple urgent connections while staying under budget.",
    "max_budget": 7000,
    "max_steps": 90,
    "passengers": [
        {
            "id": "P1",
            "name": "Iris Patel",
            "priority_tier": "Platinum",
            "original_flight": "FL-500",
            "cabin_class": "Business",
            "connection_deadline_hrs": 2.5,
        },
        {
            "id": "P2",
            "name": "Jack Rivera",
            "priority_tier": "Gold",
            "original_flight": "FL-500",
            "cabin_class": "Economy",
            "connection_deadline_hrs": 2.0,
        },
        {
            "id": "P3",
            "name": "Karen Novak",
            "priority_tier": "Gold",
            "original_flight": "FL-500",
            "cabin_class": "Business",
            "connection_deadline_hrs": 4.0,
        },
        {
            "id": "P4",
            "name": "Liam Chen",
            "priority_tier": "Silver",
            "original_flight": "FL-500",
            "cabin_class": "Economy",
            "connection_deadline_hrs": 3.0,
        },
        {
            "id": "P5",
            "name": "Maya Brooks",
            "priority_tier": "Standard",
            "original_flight": "FL-500",
            "cabin_class": "Economy",
            "connection_deadline_hrs": None,
        },
        {
            "id": "P6",
            "name": "Noah Singh",
            "priority_tier": "Platinum",
            "original_flight": "FL-500",
            "cabin_class": "Business",
            "connection_deadline_hrs": 3.5,
        },
        {
            "id": "P7",
            "name": "Olivia Green",
            "priority_tier": "Silver",
            "original_flight": "FL-500",
            "cabin_class": "Economy",
            "connection_deadline_hrs": 5.0,
        },
        {
            "id": "P8",
            "name": "Peter Hall",
            "priority_tier": "Standard",
            "original_flight": "FL-500",
            "cabin_class": "Economy",
            "connection_deadline_hrs": 2.8,
        },
    ],
    "flights": [
        {
            "id": "FL-502",
            "destination": "San Francisco",
            "departure_hrs": 1.8,
            "economy_seats": 2,
            "business_seats": 1,
            "is_partner": False,
        },
        {
            "id": "FL-504",
            "destination": "San Francisco",
            "departure_hrs": 3.0,
            "economy_seats": 2,
            "business_seats": 1,
            "is_partner": False,
        },
        {
            "id": "FL-506",
            "destination": "San Francisco",
            "departure_hrs": 5.5,
            "economy_seats": 3,
            "business_seats": 0,
            "is_partner": False,
        },
        {
            "id": "FL-701",
            "destination": "San Francisco",
            "departure_hrs": 2.2,
            "economy_seats": 2,
            "business_seats": 1,
            "is_partner": True,
        },
        {
            "id": "FL-703",
            "destination": "San Francisco",
            "departure_hrs": 4.4,
            "economy_seats": 2,
            "business_seats": 1,
            "is_partner": True,
        },
    ],
}


TASKS = {
    "easy": EASY_TASK,
    "medium": MEDIUM_TASK,
    "hard": HARD_TASK,
}


_OUTCOME_SCORES = {
    PassengerStatus.REBOOKED: 1.00,
    PassengerStatus.PARTNER_REBOOKED: 0.85,
    PassengerStatus.DOWNGRADED: 0.65,
    PassengerStatus.HOTEL_BOOKED: 0.40,
    PassengerStatus.NO_SOLUTION: 0.00,
    PassengerStatus.PENDING: 0.00,
}


_TIER_WEIGHTS = {
    PriorityTier.PLATINUM: 4,
    PriorityTier.GOLD: 3,
    PriorityTier.SILVER: 2,
    PriorityTier.STANDARD: 1,
}


_GRADING_PROFILES = {
    "easy": {
        "quality": 0.45,
        "coverage": 0.20,
        "connection": 0.10,
        "budget": 0.15,
        "policy": 0.10,
    },
    "medium": {
        "quality": 0.38,
        "coverage": 0.17,
        "connection": 0.22,
        "budget": 0.13,
        "policy": 0.10,
    },
    "hard": {
        "quality": 0.30,
        "coverage": 0.15,
        "connection": 0.30,
        "budget": 0.15,
        "policy": 0.10,
    },
}


def _clamp(value: float) -> float:
    return max(0.0001, min(0.9999, value))


def _resolve_tier_weight(tier: PriorityTier) -> int:
    if isinstance(tier, str):
        tier = PriorityTier(tier)
    return _TIER_WEIGHTS.get(tier, 1)


def _resolve_outcome_score(status: PassengerStatus) -> float:
    if isinstance(status, str):
        status = PassengerStatus(status)
    return _OUTCOME_SCORES.get(status, 0.0)


def _connection_score(state: EnvState) -> float:
    deadline_passengers = [p for p in state.passengers if p.connection_deadline_hrs is not None]
    if not deadline_passengers:
        return 0.9999

    weighted_hits = 0.0
    weighted_total = 0.0

    flights_by_id = {f.id: f for f in state.flights}
    for passenger in deadline_passengers:
        weight = _resolve_tier_weight(passenger.priority_tier)
        weighted_total += weight

        if passenger.assigned_flight is None:
            continue

        flight = flights_by_id.get(passenger.assigned_flight)
        if flight is None:
            continue

        if flight.departure_hrs <= passenger.connection_deadline_hrs:
            weighted_hits += weight
        else:
            weighted_hits += weight * 0.2

    if weighted_total <= 0:
        return 0.0001

    return _clamp(weighted_hits / weighted_total)


def _coverage_score(state: EnvState) -> float:
    if not state.passengers:
        return 0.0001
    resolved = sum(1 for p in state.passengers if p.status != PassengerStatus.PENDING)
    return _clamp(resolved / len(state.passengers))


def _quality_score(state: EnvState) -> float:
    weighted_sum = 0.0
    weighted_total = 0.0

    for passenger in state.passengers:
        weight = _resolve_tier_weight(passenger.priority_tier)
        weighted_total += weight
        weighted_sum += weight * _resolve_outcome_score(passenger.status)

    if weighted_total <= 0:
        return 0.0001

    return _clamp(weighted_sum / weighted_total)


def _budget_score(state: EnvState, max_budget: float) -> float:
    if max_budget <= 0:
        return 0.9999
    return _clamp(1.0 - (state.budget_spent / max_budget))


def _policy_score(state: EnvState) -> float:
    invalid_actions = max(getattr(state, "invalid_actions", 0), 0)
    invalid_penalty = min(invalid_actions * 0.03, 0.3)

    order: Dict[str, int] = {}
    step = 0
    for event in state.actions_taken:
        if not event.get("success", False):
            continue
        action = event.get("action", {})
        passenger_id = action.get("passenger_id")
        if passenger_id and passenger_id not in order:
            order[passenger_id] = step
            step += 1

    inversion_pairs = 0
    total_pairs = 0
    passengers = list(state.passengers)
    for i in range(len(passengers)):
        for j in range(i + 1, len(passengers)):
            p_i = passengers[i]
            p_j = passengers[j]
            w_i = _resolve_tier_weight(p_i.priority_tier)
            w_j = _resolve_tier_weight(p_j.priority_tier)
            if w_i == w_j:
                continue

            if p_i.id not in order or p_j.id not in order:
                continue

            total_pairs += 1
            if w_i > w_j and order[p_i.id] > order[p_j.id]:
                inversion_pairs += 1
            if w_j > w_i and order[p_j.id] > order[p_i.id]:
                inversion_pairs += 1

    inversion_penalty = (inversion_pairs / total_pairs) if total_pairs > 0 else 0.0
    return _clamp(1.0 - invalid_penalty - inversion_penalty)


def _grade_with_profile(state: EnvState, max_budget: float, profile_name: str) -> float:
    profile = _GRADING_PROFILES[profile_name]
    quality = _quality_score(state)
    coverage = _coverage_score(state)
    connection = _connection_score(state)
    budget = _budget_score(state, max_budget)
    policy = _policy_score(state)

    final = (
        profile["quality"] * quality
        + profile["coverage"] * coverage
        + profile["connection"] * connection
        + profile["budget"] * budget
        + profile["policy"] * policy
    )
    return _clamp(final)


def grade_easy_episode(state: EnvState, max_budget: float) -> float:
    return _grade_with_profile(state, max_budget, "easy")


def grade_medium_episode(state: EnvState, max_budget: float) -> float:
    return _grade_with_profile(state, max_budget, "medium")


def grade_hard_episode(state: EnvState, max_budget: float) -> float:
    return _grade_with_profile(state, max_budget, "hard")


TASK_GRADERS = {
    "easy": grade_easy_episode,
    "medium": grade_medium_episode,
    "hard": grade_hard_episode,
}


def grade_task(task_key: str, state: EnvState, max_budget: float) -> float:
    grader = TASK_GRADERS[task_key]
    score = grader(state, max_budget)
    # Enforce strict (0, 1) bounds required by the validator
    return max(0.0001, min(0.9999, float(score)))


def grade_episode(state: EnvState, max_budget: float) -> float:
    """Backward-compatible default grader, mapped to medium difficulty."""
    score = grade_medium_episode(state, max_budget)
    return max(0.0001, min(0.9999, float(score)))
