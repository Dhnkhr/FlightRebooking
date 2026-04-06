"""
Flight Rebooking Environment Engine
====================================
A Pydantic-based simulation environment for rebooking airline passengers
during mass flight disruptions (e.g., storms, cancellations).

Follows an OpenAI Gym-style interface: reset() -> observe, step(action) -> (obs, reward, done, info)
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum


# ============================================================
# ENUMS
# ============================================================

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


# ============================================================
# PYDANTIC MODELS
# ============================================================

class Passenger(BaseModel):
    """Represents a stranded passenger who needs rebooking."""
    id: str
    name: str
    priority_tier: PriorityTier
    original_flight: str
    cabin_class: CabinClass
    connection_deadline_hrs: Optional[float] = None
    status: PassengerStatus = PassengerStatus.PENDING
    assigned_flight: Optional[str] = None


class Flight(BaseModel):
    """Represents an available replacement flight."""
    id: str
    destination: str
    departure_hrs: float  # hours from now
    economy_seats: int
    business_seats: int
    is_partner: bool = False


class Action(BaseModel):
    """An action the agent can take."""
    action_type: str
    passenger_id: Optional[str] = None
    flight_id: Optional[str] = None


class Observation(BaseModel):
    """The current state visible to the agent."""
    pending_passengers: List[Dict[str, Any]]
    available_flights: List[Dict[str, Any]]
    budget_remaining: float
    budget_spent: float
    processed_count: int
    total_passengers: int


class EnvState(BaseModel):
    """Full internal state of the environment."""
    passengers: List[Passenger] = Field(default_factory=list)
    flights: List[Flight] = Field(default_factory=list)
    budget_spent: float = 0.0
    max_budget: float = 0.0
    actions_taken: List[Dict[str, Any]] = Field(default_factory=list)


# ============================================================
# ACTION COSTS
# ============================================================

ACTION_COSTS = {
    "rebook_passenger": 0.0,
    "offer_downgrade": 500.0,
    "book_hotel": 250.0,
    "rebook_on_partner": 800.0,
    "mark_no_solution": 0.0,
    "finalize": 0.0,
}

# Priority-tier ordering (higher value = higher priority)
PRIORITY_WEIGHTS = {
    PriorityTier.PLATINUM: 4,
    PriorityTier.GOLD: 3,
    PriorityTier.SILVER: 2,
    PriorityTier.STANDARD: 1,
}


# ============================================================
# ENVIRONMENT
# ============================================================

class FlightRebookingEnv:
    """
    Flight Rebooking Environment.

    Simulates an airline operations desk during a mass disruption.
    The agent must process stranded passengers by rebooking them onto
    available flights while respecting priority tiers, cabin classes,
    connection deadlines, and a limited budget.

    Interface:
        env = FlightRebookingEnv(task_data=...)
        obs = env.reset()
        obs, reward, done, info = env.step(action)
        state = env.state()
    """

    def __init__(self, task_data: dict):
        self.task_data = task_data
        self._state: Optional[EnvState] = None
        self._step_count = 0
        self._max_steps = 50  # Safety limit

    def reset(self) -> Observation:
        """Initialize the environment and return the first observation."""
        passengers = [Passenger(**p) for p in self.task_data["passengers"]]
        flights = [Flight(**f) for f in self.task_data["flights"]]

        self._state = EnvState(
            passengers=passengers,
            flights=flights,
            budget_spent=0.0,
            max_budget=self.task_data["max_budget"],
        )
        self._step_count = 0
        return self._get_observation()

    def state(self) -> EnvState:
        """Return the full internal state (used for grading)."""
        return self._state

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """
        Execute an action in the environment.

        Returns:
            observation: Updated state visible to the agent
            reward: Immediate reward for this action
            done: Whether the episode is finished
            info: Additional metadata (errors, etc.)
        """
        self._step_count += 1
        info: Dict[str, Any] = {}

        # Safety: force finalize after too many steps
        if self._step_count >= self._max_steps:
            info["warning"] = "Max steps reached, forcing finalize."
            return self._get_observation(), 0.0, True, info

        # --- FINALIZE ---
        if action.action_type == "finalize":
            return self._get_observation(), 0.0, True, info

        # --- VALIDATE PASSENGER ---
        passenger = self._find_passenger(action.passenger_id)
        if passenger is None:
            info["error"] = f"Passenger not found: {action.passenger_id}"
            return self._get_observation(), -0.1, False, info

        if passenger.status != PassengerStatus.PENDING:
            info["error"] = f"Passenger {action.passenger_id} already processed ({passenger.status.value})"
            return self._get_observation(), -0.1, False, info

        # --- DISPATCH ACTION ---
        handler = {
            "rebook_passenger": self._handle_rebook,
            "offer_downgrade": self._handle_downgrade,
            "book_hotel": self._handle_hotel,
            "rebook_on_partner": self._handle_partner,
            "mark_no_solution": self._handle_no_solution,
        }.get(action.action_type)

        if handler is None:
            info["error"] = f"Unknown action type: {action.action_type}"
            return self._get_observation(), -0.1, False, info

        reward, action_info = handler(passenger, action)
        info.update(action_info)

        # Record the action
        self._state.actions_taken.append(action.model_dump())

        # Check if all passengers are processed
        done = all(p.status != PassengerStatus.PENDING for p in self._state.passengers)
        if done:
            info["auto_finalized"] = True

        return self._get_observation(), reward, done, info

    # ----------------------------------------------------------
    # ACTION HANDLERS
    # ----------------------------------------------------------

    def _handle_rebook(self, passenger: Passenger, action: Action) -> Tuple[float, Dict]:
        """Rebook passenger on a same-airline flight in their original cabin."""
        info: Dict[str, Any] = {}
        flight = self._find_flight(action.flight_id)

        if flight is None:
            return -0.1, {"error": f"Flight not found: {action.flight_id}"}

        if flight.is_partner:
            return -0.1, {"error": "Cannot use rebook_passenger for partner flights. Use rebook_on_partner."}

        # Check seat in original cabin
        ok, msg = self._consume_seat(flight, passenger.cabin_class)
        if not ok:
            return -0.1, {"error": msg}

        passenger.status = PassengerStatus.REBOOKED
        passenger.assigned_flight = flight.id

        reward = self._satisfaction_reward(passenger, flight)
        return reward, info

    def _handle_downgrade(self, passenger: Passenger, action: Action) -> Tuple[float, Dict]:
        """Downgrade a Business passenger to Economy with $500 compensation."""
        info: Dict[str, Any] = {}

        if passenger.cabin_class != CabinClass.BUSINESS:
            return -0.1, {"error": "Can only downgrade Business-class passengers."}

        cost = ACTION_COSTS["offer_downgrade"]
        if not self._can_afford(cost):
            return -0.1, {"error": f"Insufficient budget. Need ${cost}, have ${self._budget_remaining():.0f}."}

        flight = self._find_flight(action.flight_id)
        if flight is None:
            return -0.1, {"error": f"Flight not found: {action.flight_id}"}

        # Downgrade means they sit in Economy
        ok, msg = self._consume_seat(flight, CabinClass.ECONOMY)
        if not ok:
            return -0.1, {"error": msg}

        self._state.budget_spent += cost
        passenger.status = PassengerStatus.DOWNGRADED
        passenger.assigned_flight = flight.id

        # Reduced satisfaction due to downgrade
        reward = self._satisfaction_reward(passenger, flight) * 0.7
        return reward, info

    def _handle_hotel(self, passenger: Passenger, action: Action) -> Tuple[float, Dict]:
        """Book a hotel for the passenger ($250). Does NOT rebook them on a flight."""
        info: Dict[str, Any] = {}

        cost = ACTION_COSTS["book_hotel"]
        if not self._can_afford(cost):
            return -0.1, {"error": f"Insufficient budget. Need ${cost}, have ${self._budget_remaining():.0f}."}

        self._state.budget_spent += cost
        passenger.status = PassengerStatus.HOTEL_BOOKED

        # Partial credit: passenger is safe but not at their destination
        reward = 0.3
        return reward, info

    def _handle_partner(self, passenger: Passenger, action: Action) -> Tuple[float, Dict]:
        """Rebook on a partner airline flight ($800)."""
        info: Dict[str, Any] = {}

        cost = ACTION_COSTS["rebook_on_partner"]
        if not self._can_afford(cost):
            return -0.1, {"error": f"Insufficient budget. Need ${cost}, have ${self._budget_remaining():.0f}."}

        flight = self._find_flight(action.flight_id)
        if flight is None:
            return -0.1, {"error": f"Flight not found: {action.flight_id}"}

        if not flight.is_partner:
            return -0.1, {"error": f"Flight {action.flight_id} is not a partner flight. Use rebook_passenger."}

        ok, msg = self._consume_seat(flight, passenger.cabin_class)
        if not ok:
            return -0.1, {"error": msg}

        self._state.budget_spent += cost
        passenger.status = PassengerStatus.PARTNER_REBOOKED
        passenger.assigned_flight = flight.id

        # Slight penalty for using partner (cost concern)
        reward = self._satisfaction_reward(passenger, flight) * 0.9
        return reward, info

    def _handle_no_solution(self, passenger: Passenger, action: Action) -> Tuple[float, Dict]:
        """Mark passenger as having no viable solution (heavy penalty)."""
        passenger.status = PassengerStatus.NO_SOLUTION
        return -0.5, {}

    # ----------------------------------------------------------
    # HELPERS
    # ----------------------------------------------------------

    def _find_passenger(self, passenger_id: Optional[str]) -> Optional[Passenger]:
        if passenger_id is None:
            return None
        for p in self._state.passengers:
            if p.id == passenger_id:
                return p
        return None

    def _find_flight(self, flight_id: Optional[str]) -> Optional[Flight]:
        if flight_id is None:
            return None
        for f in self._state.flights:
            if f.id == flight_id:
                return f
        return None

    def _consume_seat(self, flight: Flight, cabin: CabinClass) -> Tuple[bool, str]:
        """Try to consume a seat on the given flight. Returns (success, error_msg)."""
        if cabin == CabinClass.BUSINESS:
            if flight.business_seats <= 0:
                return False, f"No Business seats on {flight.id}."
            flight.business_seats -= 1
        else:
            if flight.economy_seats <= 0:
                return False, f"No Economy seats on {flight.id}."
            flight.economy_seats -= 1
        return True, ""

    def _can_afford(self, cost: float) -> bool:
        return (self._state.budget_spent + cost) <= self._state.max_budget

    def _budget_remaining(self) -> float:
        return self._state.max_budget - self._state.budget_spent

    def _satisfaction_reward(self, passenger: Passenger, flight: Flight) -> float:
        """
        Calculate reward based on:
        - Passenger priority tier (Platinum most valuable)
        - Whether the connection deadline was met
        """
        tier = passenger.priority_tier
        multiplier = {
            PriorityTier.PLATINUM: 1.5,
            PriorityTier.GOLD: 1.3,
            PriorityTier.SILVER: 1.1,
            PriorityTier.STANDARD: 1.0,
        }.get(tier, 1.0)

        base_reward = multiplier

        # Connection-deadline bonus/penalty
        if passenger.connection_deadline_hrs is not None:
            if flight.departure_hrs <= passenger.connection_deadline_hrs:
                base_reward += 0.5  # Connection saved!
            else:
                base_reward -= 0.3  # Connection missed

        return base_reward

    def _get_observation(self) -> Observation:
        """Build the agent-visible observation from internal state."""
        pending = []
        for p in self._state.passengers:
            if p.status == PassengerStatus.PENDING:
                pending.append({
                    "id": p.id,
                    "name": p.name,
                    "priority_tier": p.priority_tier.value,
                    "original_flight": p.original_flight,
                    "cabin_class": p.cabin_class.value,
                    "connection_deadline_hrs": p.connection_deadline_hrs,
                })

        flights = []
        for f in self._state.flights:
            flights.append({
                "id": f.id,
                "destination": f.destination,
                "departure_hrs": f.departure_hrs,
                "economy_seats": f.economy_seats,
                "business_seats": f.business_seats,
                "is_partner": f.is_partner,
            })

        processed = sum(
            1 for p in self._state.passengers
            if p.status != PassengerStatus.PENDING
        )

        return Observation(
            pending_passengers=pending,
            available_flights=flights,
            budget_remaining=self._budget_remaining(),
            budget_spent=self._state.budget_spent,
            processed_count=processed,
            total_passengers=len(self._state.passengers),
        )
