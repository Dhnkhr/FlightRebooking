"""
Task Definitions & Grading
============================
Contains scenario data for the flight rebooking environment and
the grading function that scores agent performance.
"""

from environment import EnvState, PassengerStatus, PriorityTier


# ============================================================
# TASK: EASY — Minor Disruption
# ============================================================
# Scenario: Flight FL-100 to New York is cancelled.
# 3 passengers, plenty of seats, generous budget.

EASY_TASK = {
    "max_budget": 3000,
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


# ============================================================
# TASK: MEDIUM — Connection Crisis
# ============================================================
# Scenario: Flight FL-300 to Chicago is cancelled.
# 5 passengers, tight connections, limited seats, tighter budget.

MEDIUM_TASK = {
    "max_budget": 5000,
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


# ============================================================
# GRADING FUNCTION
# ============================================================

# Outcome weights determine how much credit each resolution gets
_OUTCOME_SCORES = {
    PassengerStatus.REBOOKED: 1.0,           # Full credit
    PassengerStatus.PARTNER_REBOOKED: 0.95,   # Slightly less (expensive)
    PassengerStatus.DOWNGRADED: 0.70,         # Partial — comfort lost
    PassengerStatus.HOTEL_BOOKED: 0.30,       # Minimal — not at destination
    PassengerStatus.NO_SOLUTION: -0.20,       # Penalty
    PassengerStatus.PENDING: 0.0,             # Forgotten passengers
}

# Priority tier importance weights
_TIER_WEIGHTS = {
    PriorityTier.PLATINUM: 4,
    PriorityTier.GOLD: 3,
    PriorityTier.SILVER: 2,
    PriorityTier.STANDARD: 1,
}


def grade_episode(state: EnvState, max_budget: float) -> float:
    """
    Grade the agent's performance on a single episode.

    Scoring formula:
      1. Each passenger earns `tier_weight * outcome_score`.
      2. A connection bonus/penalty is applied for passengers with deadlines.
      3. A budget-efficiency multiplier rewards frugal spending.
      4. Final score is clamped to [0.0, 1.0].

    Args:
        state: The final EnvState after the episode.
        max_budget: The maximum allowed budget for the task.

    Returns:
        A float score between 0.0 and 1.0.
    """
    total_score = 0.0
    max_possible = 0.0

    for passenger in state.passengers:
        # Resolve tier (handle both enum and string)
        tier = passenger.priority_tier
        if isinstance(tier, str):
            tier = PriorityTier(tier)

        tier_weight = _TIER_WEIGHTS.get(tier, 1)
        max_possible += tier_weight

        # Base outcome score
        status = passenger.status
        if isinstance(status, str):
            status = PassengerStatus(status)

        outcome = _OUTCOME_SCORES.get(status, 0.0)
        passenger_score = tier_weight * outcome

        # Connection-deadline bonus/penalty
        if passenger.connection_deadline_hrs is not None:
            if passenger.assigned_flight is not None:
                # Find the assigned flight's departure time
                assigned = next(
                    (f for f in state.flights if f.id == passenger.assigned_flight),
                    None,
                )
                if assigned and assigned.departure_hrs <= passenger.connection_deadline_hrs:
                    passenger_score += tier_weight * 0.15  # Connection saved bonus
                elif assigned:
                    passenger_score -= tier_weight * 0.10  # Connection missed penalty
            elif status in (PassengerStatus.NO_SOLUTION, PassengerStatus.PENDING):
                passenger_score -= tier_weight * 0.10  # Missed connection entirely

        total_score += passenger_score

    if max_possible <= 0:
        return 0.0

    # Normalize to [0, 1] range
    raw = total_score / max_possible

    # Budget efficiency multiplier: spending less is better
    # Full budget usage gets 0.9x, spending nothing gets 1.0x
    budget_ratio = state.budget_spent / max_budget if max_budget > 0 else 0
    budget_multiplier = 1.0 - (budget_ratio * 0.1)

    final = raw * budget_multiplier
    return max(0.0, min(1.0, final))
