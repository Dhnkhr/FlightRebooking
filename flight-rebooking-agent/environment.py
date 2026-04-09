"""
Flight Rebooking Environment Engine
===================================

Real-world simulation of airline disruption recovery where an agent must
rebook stranded passengers under strict business constraints.

OpenEnv interface:
  - reset() -> Observation
  - step(Action) -> tuple[Observation, Reward, bool, dict]
  - state() -> EnvState
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class PriorityTier(str, Enum):
    PLATINUM = "Platinum"
    GOLD = "Gold"
    SILVER = "Silver"
    STANDARD = "Standard"


class CabinClass(str, Enum):
    BUSINESS = "Business"
    ECONOMY = "Economy"


class PassengerStatus(str, Enum):
    PENDING = "pending"
    REBOOKED = "rebooked"
    DOWNGRADED = "downgraded"
    HOTEL_BOOKED = "hotel_booked"
    PARTNER_REBOOKED = "partner_rebooked"
    NO_SOLUTION = "no_solution"


class ActionType(str, Enum):
    REBOOK_PASSENGER = "rebook_passenger"
    OFFER_DOWNGRADE = "offer_downgrade"
    BOOK_HOTEL = "book_hotel"
    REBOOK_ON_PARTNER = "rebook_on_partner"
    MARK_NO_SOLUTION = "mark_no_solution"
    FINALIZE = "finalize"


class Passenger(BaseModel):
    """A stranded passenger awaiting re-accommodation."""

    id: str
    name: str
    priority_tier: PriorityTier
    original_flight: str
    cabin_class: CabinClass
    connection_deadline_hrs: Optional[float] = None
    status: PassengerStatus = PassengerStatus.PENDING
    assigned_flight: Optional[str] = None


class Flight(BaseModel):
    """A candidate replacement flight."""

    id: str
    destination: str
    departure_hrs: float
    economy_seats: int
    business_seats: int
    is_partner: bool = False


class Action(BaseModel):
    """Action model consumed by step()."""

    action_type: ActionType
    passenger_id: Optional[str] = None
    flight_id: Optional[str] = None


class Reward(BaseModel):
    """Typed reward payload in the [0.0, 1.0] range."""

    value: float = Field(ge=0.0, le=1.0)
    components: Dict[str, float] = Field(default_factory=dict)
    notes: List[str] = Field(default_factory=list)


class Observation(BaseModel):
    """Agent-visible state after each transition."""

    pending_passengers: List[Dict[str, Any]]
    available_flights: List[Dict[str, Any]]
    budget_remaining: float
    budget_spent: float
    processed_count: int
    total_passengers: int
    invalid_actions: int
    step_count: int


class EnvState(BaseModel):
    """Full simulator state for graders and debugging."""

    passengers: List[Passenger] = Field(default_factory=list)
    flights: List[Flight] = Field(default_factory=list)
    budget_spent: float = 0.0
    max_budget: float = 0.0
    actions_taken: List[Dict[str, Any]] = Field(default_factory=list)
    invalid_actions: int = 0
    finalized: bool = False
    step_count: int = 0


ACTION_COSTS = {
    ActionType.REBOOK_PASSENGER: 0.0,
    ActionType.OFFER_DOWNGRADE: 500.0,
    ActionType.BOOK_HOTEL: 250.0,
    ActionType.REBOOK_ON_PARTNER: 800.0,
    ActionType.MARK_NO_SOLUTION: 0.0,
    ActionType.FINALIZE: 0.0,
}


PRIORITY_WEIGHTS = {
    PriorityTier.PLATINUM: 4,
    PriorityTier.GOLD: 3,
    PriorityTier.SILVER: 2,
    PriorityTier.STANDARD: 1,
}


OUTCOME_QUALITY = {
    PassengerStatus.REBOOKED: 1.00,
    PassengerStatus.PARTNER_REBOOKED: 0.85,
    PassengerStatus.DOWNGRADED: 0.65,
    PassengerStatus.HOTEL_BOOKED: 0.45,
    PassengerStatus.NO_SOLUTION: 0.05,
}


class FlightRebookingEnv:
    """OpenEnv-compatible flight rebooking simulator."""

    def __init__(self, task_data: dict):
        self.task_data = task_data
        self._state: Optional[EnvState] = None
        self._step_count = 0
        self._max_steps = int(task_data.get("max_steps", 80))

    def reset(self) -> Observation:
        passengers = [Passenger(**p) for p in self.task_data["passengers"]]
        flights = [Flight(**f) for f in self.task_data["flights"]]

        self._state = EnvState(
            passengers=passengers,
            flights=flights,
            budget_spent=0.0,
            max_budget=self.task_data["max_budget"],
            actions_taken=[],
            invalid_actions=0,
            finalized=False,
            step_count=0,
        )
        self._step_count = 0
        return self._get_observation()

    def state(self) -> EnvState:
        if self._state is None:
            raise RuntimeError("Environment is not initialized. Call reset() first.")
        return self._state

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        if self._state is None:
            raise RuntimeError("Environment is not initialized. Call reset() first.")

        self._step_count += 1
        self._state.step_count = self._step_count
        info: Dict[str, Any] = {}

        if self._state.finalized:
            reward = Reward(value=0.0, components={"terminal": 1.0}, notes=["episode_already_finalized"])
            return self._get_observation(), reward, True, {"warning": "Episode already finalized."}

        if self._step_count > self._max_steps:
            self._state.finalized = True
            reward = Reward(
                value=0.0,
                components={
                    "progress": self._completion_ratio(),
                    "budget_efficiency": self._budget_efficiency(),
                    "max_step_exceeded": 1.0,
                },
                notes=["forced_finalize_max_steps"],
            )
            self._record_action(action, reward, success=False, done=True, info={"warning": "Max steps reached."})
            return self._get_observation(), reward, True, {"warning": "Max steps reached, forcing finalize."}

        if action.action_type == ActionType.FINALIZE:
            reward = self._build_finalize_reward()
            self._state.finalized = True
            unresolved = [p.id for p in self._state.passengers if p.status == PassengerStatus.PENDING]
            if unresolved:
                info["unresolved_passengers"] = unresolved
            self._record_action(action, reward, success=(len(unresolved) == 0), done=True, info=info)
            return self._get_observation(), reward, True, info

        passenger = self._find_passenger(action.passenger_id)
        if passenger is None:
            reward = self._invalid_reward("passenger_not_found")
            info["error"] = f"Passenger not found: {action.passenger_id}"
            self._record_action(action, reward, success=False, done=False, info=info)
            return self._get_observation(), reward, False, info

        if passenger.status != PassengerStatus.PENDING:
            reward = self._invalid_reward("passenger_already_processed")
            info["error"] = f"Passenger {action.passenger_id} already processed ({passenger.status.value})."
            self._record_action(action, reward, success=False, done=False, info=info)
            return self._get_observation(), reward, False, info

        priority_inversion = self._has_higher_priority_pending(passenger)

        handler = {
            ActionType.REBOOK_PASSENGER: self._handle_rebook,
            ActionType.OFFER_DOWNGRADE: self._handle_downgrade,
            ActionType.BOOK_HOTEL: self._handle_hotel,
            ActionType.REBOOK_ON_PARTNER: self._handle_partner,
            ActionType.MARK_NO_SOLUTION: self._handle_no_solution,
        }[action.action_type]

        success, action_info = handler(passenger, action)
        info.update(action_info)

        if not success:
            reward = self._invalid_reward(info.get("error", "invalid_action"))
            self._record_action(action, reward, success=False, done=False, info=info)
            return self._get_observation(), reward, False, info

        repeat_penalty = self._repeat_failure_penalty(action)
        reward = self._build_resolution_reward(
            passenger=passenger,
            flight=self._find_flight(passenger.assigned_flight),
            action_cost=ACTION_COSTS[action.action_type],
            priority_inversion=priority_inversion,
            repeat_penalty=repeat_penalty,
        )

        done = all(p.status != PassengerStatus.PENDING for p in self._state.passengers)
        if done:
            self._state.finalized = True
            reward = self._add_terminal_bonus(reward)
            info["auto_finalized"] = True

        self._record_action(action, reward, success=True, done=done, info=info)
        return self._get_observation(), reward, done, info

    def _handle_rebook(self, passenger: Passenger, action: Action) -> Tuple[bool, Dict[str, Any]]:
        flight = self._find_flight(action.flight_id)
        if flight is None:
            return False, {"error": f"Flight not found: {action.flight_id}"}

        if flight.is_partner:
            return False, {"error": "Use rebook_on_partner for partner flights."}

        ok, msg = self._consume_seat(flight, passenger.cabin_class)
        if not ok:
            return False, {"error": msg}

        passenger.status = PassengerStatus.REBOOKED
        passenger.assigned_flight = flight.id
        return True, {"resolved_status": passenger.status.value}

    def _handle_downgrade(self, passenger: Passenger, action: Action) -> Tuple[bool, Dict[str, Any]]:
        if passenger.cabin_class != CabinClass.BUSINESS:
            return False, {"error": "Can only downgrade Business passengers."}

        cost = ACTION_COSTS[ActionType.OFFER_DOWNGRADE]
        if not self._spend(cost):
            return False, {"error": f"Insufficient budget. Need ${cost:.0f}, have ${self._budget_remaining():.0f}."}

        flight = self._find_flight(action.flight_id)
        if flight is None:
            self._refund(cost)
            return False, {"error": f"Flight not found: {action.flight_id}"}

        ok, msg = self._consume_seat(flight, CabinClass.ECONOMY)
        if not ok:
            self._refund(cost)
            return False, {"error": msg}

        passenger.status = PassengerStatus.DOWNGRADED
        passenger.assigned_flight = flight.id
        return True, {"resolved_status": passenger.status.value}

    def _handle_hotel(self, passenger: Passenger, action: Action) -> Tuple[bool, Dict[str, Any]]:
        cost = ACTION_COSTS[ActionType.BOOK_HOTEL]
        if not self._spend(cost):
            return False, {"error": f"Insufficient budget. Need ${cost:.0f}, have ${self._budget_remaining():.0f}."}

        passenger.status = PassengerStatus.HOTEL_BOOKED
        passenger.assigned_flight = None
        return True, {"resolved_status": passenger.status.value}

    def _handle_partner(self, passenger: Passenger, action: Action) -> Tuple[bool, Dict[str, Any]]:
        cost = ACTION_COSTS[ActionType.REBOOK_ON_PARTNER]
        if not self._spend(cost):
            return False, {"error": f"Insufficient budget. Need ${cost:.0f}, have ${self._budget_remaining():.0f}."}

        flight = self._find_flight(action.flight_id)
        if flight is None:
            self._refund(cost)
            return False, {"error": f"Flight not found: {action.flight_id}"}

        if not flight.is_partner:
            self._refund(cost)
            return False, {"error": f"Flight {action.flight_id} is not a partner flight."}

        ok, msg = self._consume_seat(flight, passenger.cabin_class)
        if not ok:
            self._refund(cost)
            return False, {"error": msg}

        passenger.status = PassengerStatus.PARTNER_REBOOKED
        passenger.assigned_flight = flight.id
        return True, {"resolved_status": passenger.status.value}

    def _handle_no_solution(self, passenger: Passenger, action: Action) -> Tuple[bool, Dict[str, Any]]:
        passenger.status = PassengerStatus.NO_SOLUTION
        passenger.assigned_flight = None
        return True, {"resolved_status": passenger.status.value}

    def _invalid_reward(self, reason: str) -> Reward:
        self._state.invalid_actions += 1
        penalty = min(0.08 * self._state.invalid_actions, 0.5)
        return Reward(
            value=max(0.0, 0.05 - penalty),
            components={
                "progress": self._completion_ratio(),
                "budget_efficiency": self._budget_efficiency(),
                "invalid_action_penalty": penalty,
            },
            notes=[reason, "invalid_action"],
        )

    def _build_resolution_reward(
        self,
        passenger: Passenger,
        flight: Optional[Flight],
        action_cost: float,
        priority_inversion: bool,
        repeat_penalty: float,
    ) -> Reward:
        progress = self._completion_ratio()
        outcome_quality = OUTCOME_QUALITY.get(passenger.status, 0.0)
        priority_score = PRIORITY_WEIGHTS[passenger.priority_tier] / 4.0
        deadline_score = self._deadline_score(passenger, flight)
        budget_efficiency = self._budget_efficiency()

        penalty = 0.0
        notes: List[str] = []

        if priority_inversion:
            penalty += 0.15
            notes.append("priority_inversion")

        if repeat_penalty > 0:
            penalty += repeat_penalty
            notes.append("repeated_failed_action_pattern")

        if passenger.status == PassengerStatus.NO_SOLUTION:
            penalty += 0.2
            notes.append("no_solution_penalty")

        if action_cost > 0:
            # Costly actions are valid but receive a mild regularization penalty.
            penalty += min(action_cost / max(self._state.max_budget, 1.0), 0.15)

        base = (
            (0.30 * outcome_quality)
            + (0.25 * progress)
            + (0.15 * priority_score)
            + (0.15 * deadline_score)
            + (0.15 * budget_efficiency)
        )

        value = self._clamp(base - penalty)
        return Reward(
            value=value,
            components={
                "progress": progress,
                "outcome_quality": outcome_quality,
                "priority_score": priority_score,
                "deadline_score": deadline_score,
                "budget_efficiency": budget_efficiency,
                "penalty": penalty,
            },
            notes=notes,
        )

    def _build_finalize_reward(self) -> Reward:
        pending_count = sum(1 for p in self._state.passengers if p.status == PassengerStatus.PENDING)
        total = max(len(self._state.passengers), 1)
        completion = self._completion_ratio()
        budget_efficiency = self._budget_efficiency()

        if pending_count == 0:
            value = self._clamp((0.85 * completion) + (0.15 * budget_efficiency))
            notes = ["clean_finalize"]
        else:
            unresolved_penalty = pending_count / total
            value = self._clamp(0.20 * completion - 0.30 * unresolved_penalty)
            notes = ["early_finalize_penalty"]

        return Reward(
            value=value,
            components={
                "completion": completion,
                "budget_efficiency": budget_efficiency,
                "pending_ratio": pending_count / total,
            },
            notes=notes,
        )

    def _add_terminal_bonus(self, reward: Reward) -> Reward:
        bonus = 0.1 * max(0.0, 1.0 - (self._state.invalid_actions * 0.05))
        merged = dict(reward.components)
        merged["terminal_bonus"] = bonus
        return Reward(
            value=self._clamp(reward.value + bonus),
            components=merged,
            notes=reward.notes + ["all_passengers_processed"],
        )

    def _deadline_score(self, passenger: Passenger, flight: Optional[Flight]) -> float:
        if passenger.connection_deadline_hrs is None:
            return 1.0

        if flight is None:
            return 0.0

        if flight.departure_hrs <= passenger.connection_deadline_hrs:
            return 1.0

        return 0.2

    def _has_higher_priority_pending(self, passenger: Passenger) -> bool:
        current_weight = PRIORITY_WEIGHTS[passenger.priority_tier]
        for other in self._state.passengers:
            if other.id == passenger.id or other.status != PassengerStatus.PENDING:
                continue

            other_weight = PRIORITY_WEIGHTS[other.priority_tier]
            if other_weight > current_weight:
                return True

            if (
                other_weight == current_weight
                and other.connection_deadline_hrs is not None
                and passenger.connection_deadline_hrs is not None
                and other.connection_deadline_hrs < passenger.connection_deadline_hrs
            ):
                return True

            if (
                other_weight == current_weight
                and other.connection_deadline_hrs is not None
                and passenger.connection_deadline_hrs is None
            ):
                return True

        return False

    def _repeat_failure_penalty(self, action: Action) -> float:
        if len(self._state.actions_taken) < 2:
            return 0.0

        signature = self._signature(action)
        recent = self._state.actions_taken[-2:]
        repeated_failures = all(
            (not item.get("success", True)) and tuple(item.get("signature", ())) == signature
            for item in recent
        )
        return 0.1 if repeated_failures else 0.0

    def _completion_ratio(self) -> float:
        total = max(len(self._state.passengers), 1)
        processed = sum(1 for p in self._state.passengers if p.status != PassengerStatus.PENDING)
        return processed / total

    def _budget_efficiency(self) -> float:
        if self._state.max_budget <= 0:
            return 1.0
        return self._clamp(1.0 - (self._state.budget_spent / self._state.max_budget))

    def _spend(self, cost: float) -> bool:
        if (self._state.budget_spent + cost) > self._state.max_budget:
            return False
        self._state.budget_spent += cost
        return True

    def _refund(self, cost: float) -> None:
        self._state.budget_spent = max(0.0, self._state.budget_spent - cost)

    def _find_passenger(self, passenger_id: Optional[str]) -> Optional[Passenger]:
        if passenger_id is None:
            return None
        for passenger in self._state.passengers:
            if passenger.id == passenger_id:
                return passenger
        return None

    def _find_flight(self, flight_id: Optional[str]) -> Optional[Flight]:
        if flight_id is None:
            return None
        for flight in self._state.flights:
            if flight.id == flight_id:
                return flight
        return None

    def _consume_seat(self, flight: Flight, cabin: CabinClass) -> Tuple[bool, str]:
        if cabin == CabinClass.BUSINESS:
            if flight.business_seats <= 0:
                return False, f"No Business seats on {flight.id}."
            flight.business_seats -= 1
            return True, ""

        if flight.economy_seats <= 0:
            return False, f"No Economy seats on {flight.id}."
        flight.economy_seats -= 1
        return True, ""

    def _budget_remaining(self) -> float:
        return self._state.max_budget - self._state.budget_spent

    def _signature(self, action: Action) -> Tuple[str, Optional[str], Optional[str]]:
        return action.action_type.value, action.passenger_id, action.flight_id

    def _record_action(self, action: Action, reward: Reward, success: bool, done: bool, info: Dict[str, Any]) -> None:
        self._state.actions_taken.append(
            {
                "step": self._step_count,
                "signature": self._signature(action),
                "action": action.model_dump(mode="json"),
                "reward": reward.model_dump(mode="json"),
                "success": success,
                "done": done,
                "info": info,
            }
        )

    def _clamp(self, value: float) -> float:
        return max(0.0, min(1.0, value))

    def _get_observation(self) -> Observation:
        pending_passengers: List[Dict[str, Any]] = []
        for passenger in self._state.passengers:
            if passenger.status != PassengerStatus.PENDING:
                continue
            pending_passengers.append(
                {
                    "id": passenger.id,
                    "name": passenger.name,
                    "priority_tier": passenger.priority_tier.value,
                    "original_flight": passenger.original_flight,
                    "cabin_class": passenger.cabin_class.value,
                    "connection_deadline_hrs": passenger.connection_deadline_hrs,
                }
            )

        pending_passengers.sort(
            key=lambda p: (
                -PRIORITY_WEIGHTS[PriorityTier(p["priority_tier"])],
                p["connection_deadline_hrs"] if p["connection_deadline_hrs"] is not None else 1e9,
            )
        )

        available_flights: List[Dict[str, Any]] = []
        for flight in self._state.flights:
            available_flights.append(
                {
                    "id": flight.id,
                    "destination": flight.destination,
                    "departure_hrs": flight.departure_hrs,
                    "economy_seats": flight.economy_seats,
                    "business_seats": flight.business_seats,
                    "is_partner": flight.is_partner,
                }
            )

        processed = sum(1 for p in self._state.passengers if p.status != PassengerStatus.PENDING)

        return Observation(
            pending_passengers=pending_passengers,
            available_flights=available_flights,
            budget_remaining=self._budget_remaining(),
            budget_spent=self._state.budget_spent,
            processed_count=processed,
            total_passengers=len(self._state.passengers),
            invalid_actions=self._state.invalid_actions,
            step_count=self._step_count,
        )
