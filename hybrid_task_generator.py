"""
Hybrid Task Generator — Real flights + Synthetic passengers
============================================================

Samples real flights from the nycflights13 CSV dataset for realistic routes,
carriers, and departure times, then layers synthetic seat counts and
passenger manifests on top.

Produces task dicts fully compatible with FlightRebookingEnv and the
existing grading system (easy / medium / hard profiles).

Usage:
    # Standalone validation — generate & test 10 tasks
    python hybrid_task_generator.py --count 10 --seed 42

    # As a library
    from hybrid_task_generator import HybridTaskGenerator
    gen = HybridTaskGenerator(csv_path="data/flights.csv", seed=42)
    task = gen.generate_task(difficulty="medium")
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


# ── Constants ──

ORIGINS = ["EWR", "JFK", "LGA"]

# Dominant carrier at each origin (treated as "home airline")
_HOME_CARRIERS = {
    "EWR": "UA",
    "JFK": "B6",
    "LGA": "DL",
}

# Realistic name pool for synthetic passengers
_FIRST_NAMES = [
    "Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry",
    "Iris", "Jack", "Karen", "Liam", "Maya", "Noah", "Olivia", "Peter",
    "Quinn", "Rachel", "Sam", "Tara", "Uma", "Victor", "Wendy", "Xavier",
    "Yara", "Zach", "Amara", "Bryan", "Chloe", "Derek", "Elena", "Finn",
    "Gina", "Hugo", "Ines", "James", "Kira", "Leo", "Mila", "Nate",
    "Opal", "Priya", "Ravi", "Sana", "Troy", "Vera", "Will", "Xena",
]

_LAST_NAMES = [
    "Johnson", "Smith", "Davis", "Lee", "Wilson", "Brown", "Kim", "Park",
    "Patel", "Rivera", "Novak", "Chen", "Brooks", "Singh", "Green", "Hall",
    "Lopez", "Martinez", "Anderson", "Thomas", "Taylor", "Moore", "Jackson",
    "White", "Harris", "Clark", "Lewis", "Young", "Walker", "Allen",
    "Wright", "Scott", "Torres", "Nguyen", "Hill", "Adams", "Baker",
    "Cruz", "Diaz", "Evans", "Flores", "Garcia", "Hayes", "Ito", "James",
]

# Difficulty profiles
_DIFFICULTY_PROFILES = {
    "easy": {
        "passengers_range": (2, 3),
        "flights_range": (3, 4),
        "partner_ratio": 0.20,
        "deadline_ratio": 0.25,       # fraction of passengers with deadlines
        "business_ratio": 0.20,       # fraction of business class passengers
        "budget_multiplier": 1.50,    # generous budget
        "max_steps_range": (30, 50),
        "seat_scale": 1.3,            # more seats available
        "tier_weights": {             # probability distribution for tiers
            "Standard": 0.50,
            "Silver": 0.25,
            "Gold": 0.15,
            "Platinum": 0.10,
        },
    },
    "medium": {
        "passengers_range": (4, 6),
        "flights_range": (3, 5),
        "partner_ratio": 0.30,
        "deadline_ratio": 0.50,
        "business_ratio": 0.30,
        "budget_multiplier": 1.10,
        "max_steps_range": (45, 70),
        "seat_scale": 1.0,
        "tier_weights": {
            "Standard": 0.35,
            "Silver": 0.25,
            "Gold": 0.25,
            "Platinum": 0.15,
        },
    },
    "hard": {
        "passengers_range": (6, 9),
        "flights_range": (4, 6),
        "partner_ratio": 0.40,
        "deadline_ratio": 0.70,
        "business_ratio": 0.40,
        "budget_multiplier": 0.85,
        "max_steps_range": (60, 100),
        "seat_scale": 0.7,            # scarce seats
        "tier_weights": {
            "Standard": 0.25,
            "Silver": 0.25,
            "Gold": 0.30,
            "Platinum": 0.20,
        },
    },
}


class HybridTaskGenerator:
    """Generates flight rebooking tasks using real CSV flight data."""

    def __init__(self, csv_path: str = "data/flights.csv", seed: int = 42):
        self.rng = random.Random(seed)
        self._load_csv(csv_path)
        self._task_counter = 0

    def _load_csv(self, csv_path: str) -> None:
        """Load and index the flights CSV."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"flights.csv not found at '{csv_path}'. "
                f"Copy it to data/flights.csv inside the project."
            )

        self._all_flights: List[Dict[str, Any]] = []
        self._route_index: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        self._cancelled_index: Dict[str, List[int]] = defaultdict(list)
        self._carriers_at_origin: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row_idx, row in enumerate(reader):
                origin = row.get("origin", "").strip()
                dest = row.get("dest", "").strip()
                carrier = row.get("carrier", "").strip()
                dep_time = row.get("dep_time", "").strip()
                sched_dep = row.get("sched_dep_time", "").strip()

                if not origin or not dest or not carrier or not sched_dep:
                    continue

                flight_record = {
                    "row_idx": row_idx,
                    "origin": origin,
                    "dest": dest,
                    "carrier": carrier,
                    "flight_num": row.get("flight", "").strip(),
                    "sched_dep_time": int(sched_dep),
                    "dep_time": float(dep_time) if dep_time else None,
                    "dep_delay": float(row["dep_delay"]) if row.get("dep_delay", "").strip() else None,
                    "distance": int(row.get("distance", "0") or "0"),
                    "airline_name": row.get("name", carrier).strip(),
                }

                idx = len(self._all_flights)
                self._all_flights.append(flight_record)
                self._route_index[(origin, dest)].append(idx)
                self._carriers_at_origin[origin][carrier] += 1

                # Track cancelled / heavily delayed flights as disruption triggers
                if dep_time == "" or (flight_record["dep_delay"] is not None and flight_record["dep_delay"] > 60):
                    self._cancelled_index[origin].append(idx)

        # Precompute dominant carrier per origin
        self._home_carrier: Dict[str, str] = {}
        for origin, carrier_counts in self._carriers_at_origin.items():
            self._home_carrier[origin] = max(carrier_counts, key=carrier_counts.get)

        # Collect routes with enough flights for sampling
        self._viable_routes: List[Tuple[str, str]] = [
            route for route, indices in self._route_index.items()
            if len(indices) >= 10  # need enough variety
        ]

        if not self._viable_routes:
            raise ValueError("No viable routes found in CSV (need ≥10 flights per route).")

        print(
            f"[HybridTaskGen] Loaded {len(self._all_flights):,} flights, "
            f"{len(self._viable_routes)} viable routes, "
            f"{sum(len(v) for v in self._cancelled_index.values()):,} disruption candidates"
        )

    def set_seed(self, seed: int) -> None:
        """Reset the RNG with a new seed."""
        self.rng = random.Random(seed)

    def generate_task(self, difficulty: str = "medium") -> Dict[str, Any]:
        """Generate a single task dict compatible with FlightRebookingEnv.

        Args:
            difficulty: "easy", "medium", or "hard"

        Returns:
            Task dict with keys: task_id, difficulty, objective, max_budget,
            max_steps, passengers, flights
        """
        profile = _DIFFICULTY_PROFILES[difficulty]
        self._task_counter += 1

        # ── Step 1: Pick a route ──
        route = self.rng.choice(self._viable_routes)
        origin, dest = route
        home_carrier = self._home_carrier.get(origin, _HOME_CARRIERS.get(origin, "UA"))

        # ── Step 2: Pick the disrupted flight ──
        disrupted_flight_id = self._pick_disrupted_flight(origin, dest, home_carrier)

        # ── Step 3: Sample replacement flights ──
        num_flights = self.rng.randint(*profile["flights_range"])
        flights = self._sample_flights(
            origin=origin,
            dest=dest,
            home_carrier=home_carrier,
            num_flights=num_flights,
            partner_ratio=profile["partner_ratio"],
            seat_scale=profile["seat_scale"],
        )

        # ── Step 4: Generate passengers ──
        num_passengers = self.rng.randint(*profile["passengers_range"])
        passengers = self._generate_passengers(
            num_passengers=num_passengers,
            disrupted_flight_id=disrupted_flight_id,
            flights=flights,
            profile=profile,
        )

        # ── Step 5: Calculate budget ──
        base_budget = num_passengers * 800
        budget = max(
            1000,
            int(round(base_budget * profile["budget_multiplier"] / 250.0) * 250),
        )

        # ── Step 6: Determine max_steps ──
        max_steps = self.rng.randint(*profile["max_steps_range"])

        task_id = f"hybrid_{origin}_{dest}_{self._task_counter}"

        return {
            "task_id": task_id,
            "difficulty": difficulty,
            "objective": (
                f"Handle disruption on route {origin}→{dest}. "
                f"Rebook {num_passengers} stranded passengers using "
                f"{num_flights} available flights while respecting tiers, "
                f"deadlines, and budget constraints."
            ),
            "max_budget": budget,
            "max_steps": max_steps,
            "passengers": passengers,
            "flights": flights,
        }

    def generate_batch(
        self,
        count: int,
        difficulty_weights: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """Generate a batch of tasks with mixed difficulties.

        Args:
            count: Number of tasks to generate
            difficulty_weights: Optional weights, e.g. {"easy": 0.3, "medium": 0.4, "hard": 0.3}
                Defaults to uniform distribution.

        Returns:
            List of task dicts
        """
        if difficulty_weights is None:
            difficulty_weights = {"easy": 1.0, "medium": 1.0, "hard": 1.0}

        difficulties = list(difficulty_weights.keys())
        weights = list(difficulty_weights.values())

        tasks = []
        for _ in range(count):
            difficulty = self.rng.choices(difficulties, weights=weights, k=1)[0]
            tasks.append(self.generate_task(difficulty=difficulty))
        return tasks

    # ── Private helpers ──

    def _pick_disrupted_flight(self, origin: str, dest: str, home_carrier: str) -> str:
        """Pick a disrupted flight ID from real data."""
        # Try to find a cancelled/delayed flight on this route with the home carrier
        route_indices = set(self._route_index.get((origin, dest), []))
        cancelled_at_origin = self._cancelled_index.get(origin, [])

        # Cancelled flights on this exact route with home carrier
        home_cancelled = [
            idx for idx in cancelled_at_origin
            if idx in route_indices and self._all_flights[idx]["carrier"] == home_carrier
        ]

        if home_cancelled:
            chosen = self._all_flights[self.rng.choice(home_cancelled)]
        elif cancelled_at_origin:
            # Any cancelled flight from this origin
            chosen = self._all_flights[self.rng.choice(cancelled_at_origin)]
        else:
            # Fallback: pick any flight on the route
            chosen = self._all_flights[self.rng.choice(list(route_indices))]

        return f"FL-{chosen['carrier']}{chosen['flight_num']}"

    def _sched_dep_to_hours(self, sched_dep_time: int, disruption_hour: float) -> float:
        """Convert HHMM scheduled departure to hours-from-disruption.

        Args:
            sched_dep_time: Scheduled departure in HHMM format (e.g. 1430 = 2:30 PM)
            disruption_hour: Hour of the disruption event (0-24)

        Returns:
            Float hours from disruption, clamped to [0.5, 12.0]
        """
        hours = sched_dep_time // 100
        minutes = sched_dep_time % 100
        flight_hour = hours + minutes / 60.0

        delta = flight_hour - disruption_hour
        if delta < 0:
            delta += 24.0  # next day

        return max(0.5, min(12.0, round(delta, 1)))

    def _synthesize_seats(self, distance: int, seat_scale: float) -> Tuple[int, int]:
        """Generate realistic seat counts based on route distance.

        Longer routes → larger aircraft → more total seats but similar availability.
        Shorter routes → smaller regional jets → fewer seats.
        """
        if distance < 500:
            # Regional jet (CRJ/ERJ)
            base_economy = self.rng.randint(1, 5)
            base_business = self.rng.randint(0, 1)
        elif distance < 1200:
            # Narrowbody (A320/737)
            base_economy = self.rng.randint(2, 8)
            base_business = self.rng.randint(0, 2)
        else:
            # Widebody or large narrowbody
            base_economy = self.rng.randint(3, 10)
            base_business = self.rng.randint(0, 3)

        economy = max(0, int(round(base_economy * seat_scale)))
        business = max(0, int(round(base_business * seat_scale)))
        return economy, business

    def _sample_flights(
        self,
        origin: str,
        dest: str,
        home_carrier: str,
        num_flights: int,
        partner_ratio: float,
        seat_scale: float,
    ) -> List[Dict[str, Any]]:
        """Sample replacement flights from CSV data."""
        route_key = (origin, dest)
        available_indices = list(self._route_index.get(route_key, []))

        if len(available_indices) < num_flights:
            # If not enough flights on exact route, supplement with nearby routes
            for alt_dest in self._get_nearby_destinations(origin, dest):
                available_indices.extend(self._route_index.get((origin, alt_dest), []))
                if len(available_indices) >= num_flights * 3:
                    break

        # Sample more than needed and deduplicate by carrier+time
        sample_size = min(len(available_indices), num_flights * 4)
        sampled_indices = self.rng.sample(available_indices, sample_size)

        # Pick a disruption time (when the cancellation happened)
        disruption_hour = self.rng.uniform(6.0, 18.0)

        # Build flight candidates
        candidates = []
        seen_signatures = set()

        for idx in sampled_indices:
            record = self._all_flights[idx]

            # Skip cancelled flights (those are disruptions, not replacements)
            if record["dep_time"] is None:
                continue

            departure_hrs = self._sched_dep_to_hours(record["sched_dep_time"], disruption_hour)

            # Determine if partner
            is_partner = record["carrier"] != home_carrier

            # Deduplicate by (partner_status, approximate_departure)
            dep_bucket = round(departure_hrs * 2) / 2  # 30-min buckets
            sig = (is_partner, dep_bucket)
            if sig in seen_signatures:
                continue
            seen_signatures.add(sig)

            distance = record["distance"] or 1000
            economy_seats, business_seats = self._synthesize_seats(distance, seat_scale)

            flight_id = f"FL-{record['carrier']}{record['flight_num']}"

            candidates.append({
                "id": flight_id,
                "destination": dest,
                "departure_hrs": departure_hrs,
                "economy_seats": economy_seats,
                "business_seats": business_seats,
                "is_partner": is_partner,
                "_carrier": record["carrier"],
                "_distance": distance,
            })

        # Ensure we have the right partner ratio
        home_flights = [f for f in candidates if not f["is_partner"]]
        partner_flights = [f for f in candidates if f["is_partner"]]

        target_partner_count = max(1, int(round(num_flights * partner_ratio)))
        target_home_count = num_flights - target_partner_count

        # Select flights to meet targets
        selected = []

        if len(home_flights) >= target_home_count:
            selected.extend(self.rng.sample(home_flights, target_home_count))
        else:
            selected.extend(home_flights)

        if len(partner_flights) >= target_partner_count:
            selected.extend(self.rng.sample(partner_flights, target_partner_count))
        else:
            selected.extend(partner_flights)

        # Fill remaining if under target
        remaining = [f for f in candidates if f not in selected]
        while len(selected) < num_flights and remaining:
            selected.append(remaining.pop(self.rng.randrange(len(remaining))))

        # If still not enough, synthesize fallback flights
        while len(selected) < num_flights:
            fallback_departure = round(self.rng.uniform(1.0, 10.0), 1)
            is_partner = self.rng.random() < partner_ratio
            economy, business = self._synthesize_seats(1000, seat_scale)
            selected.append({
                "id": f"FL-SYN{self._task_counter}{len(selected)}",
                "destination": dest,
                "departure_hrs": fallback_departure,
                "economy_seats": economy,
                "business_seats": business,
                "is_partner": is_partner,
                "_carrier": "SYN",
                "_distance": 1000,
            })

        # Sort by departure time
        selected.sort(key=lambda f: f["departure_hrs"])

        # Clean up internal fields
        cleaned = []
        for flight in selected:
            cleaned.append({
                "id": flight["id"],
                "destination": flight["destination"],
                "departure_hrs": flight["departure_hrs"],
                "economy_seats": flight["economy_seats"],
                "business_seats": flight["business_seats"],
                "is_partner": flight["is_partner"],
            })

        return cleaned

    def _get_nearby_destinations(self, origin: str, target_dest: str) -> List[str]:
        """Get destination airports that share routes from the same origin."""
        all_dests = set()
        for (o, d) in self._viable_routes:
            if o == origin and d != target_dest:
                all_dests.add(d)
        return list(all_dests)

    def _generate_passengers(
        self,
        num_passengers: int,
        disrupted_flight_id: str,
        flights: List[Dict[str, Any]],
        profile: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Generate synthetic passenger manifest."""
        tier_labels = list(profile["tier_weights"].keys())
        tier_probs = list(profile["tier_weights"].values())

        # Collect available departure times for deadline generation
        departure_times = sorted(f["departure_hrs"] for f in flights)
        max_departure = max(departure_times) if departure_times else 8.0

        passengers = []
        used_names = set()

        for i in range(num_passengers):
            passenger_id = f"P{i + 1}"

            # Pick a unique name
            for _ in range(50):
                first = self.rng.choice(_FIRST_NAMES)
                last = self.rng.choice(_LAST_NAMES)
                name = f"{first} {last}"
                if name not in used_names:
                    used_names.add(name)
                    break

            # Priority tier
            tier = self.rng.choices(tier_labels, weights=tier_probs, k=1)[0]

            # Cabin class
            cabin_class = "Business" if self.rng.random() < profile["business_ratio"] else "Economy"

            # Connection deadline
            connection_deadline = None
            if self.rng.random() < profile["deadline_ratio"]:
                # Derive from actual flight departure gaps
                if len(departure_times) >= 2:
                    # Pick a deadline near one of the middle departure times
                    target_dep = self.rng.choice(departure_times[:-1])
                    connection_deadline = round(
                        target_dep + self.rng.uniform(-0.5, 1.5), 1
                    )
                else:
                    connection_deadline = round(self.rng.uniform(2.0, max_departure), 1)

                connection_deadline = max(1.0, min(12.0, connection_deadline))

            passengers.append({
                "id": passenger_id,
                "name": name,
                "priority_tier": tier,
                "original_flight": disrupted_flight_id,
                "cabin_class": cabin_class,
                "connection_deadline_hrs": connection_deadline,
            })

        return passengers


# ── Standalone validation ──

def _validate_task(task: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate a generated task against the environment schema."""
    try:
        from environment import FlightRebookingEnv
        env = FlightRebookingEnv(task_data=task)
        obs = env.reset()

        # Basic checks
        assert len(task["passengers"]) > 0, "No passengers"
        assert len(task["flights"]) > 0, "No flights"
        assert task["max_budget"] > 0, "Zero budget"
        assert task["max_steps"] > 0, "Zero steps"

        # Check observation is valid
        assert obs.total_passengers == len(task["passengers"])
        assert len(obs.pending_passengers) == len(task["passengers"])
        assert obs.budget_remaining == task["max_budget"]

        return True, "OK"
    except Exception as exc:
        return False, str(exc)


def main():
    parser = argparse.ArgumentParser(description="Generate and validate hybrid tasks.")
    parser.add_argument("--csv-path", default="data/flights.csv")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard", "mixed"], default="mixed")
    args = parser.parse_args()

    gen = HybridTaskGenerator(csv_path=args.csv_path, seed=args.seed)

    print(f"\nGenerating {args.count} tasks (difficulty={args.difficulty})...\n")

    passed = 0
    failed = 0
    difficulty_counts = {"easy": 0, "medium": 0, "hard": 0}
    action_type_counts = {"partner": 0, "home": 0}

    for i in range(args.count):
        if args.difficulty == "mixed":
            difficulty = gen.rng.choice(["easy", "medium", "hard"])
        else:
            difficulty = args.difficulty

        task = gen.generate_task(difficulty=difficulty)
        ok, msg = _validate_task(task)

        difficulty_counts[difficulty] += 1
        num_partner = sum(1 for f in task["flights"] if f["is_partner"])
        num_home = len(task["flights"]) - num_partner
        action_type_counts["partner"] += num_partner
        action_type_counts["home"] += num_home

        status = "PASS" if ok else "FAIL"
        print(
            f"  {status} Task {i+1:3d}: {task['task_id']:30s} "
            f"difficulty={difficulty:6s} "
            f"passengers={len(task['passengers'])} "
            f"flights={len(task['flights'])} (home={num_home}, partner={num_partner}) "
            f"budget=${task['max_budget']:,} "
            f"steps={task['max_steps']}"
        )

        if not ok:
            print(f"       Error: {msg}")
            failed += 1
        else:
            passed += 1

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{args.count} passed, {failed}/{args.count} failed")
    print(f"Difficulty distribution: {dict(difficulty_counts)}")
    print(
        f"Flight types: {action_type_counts['home']} home, "
        f"{action_type_counts['partner']} partner "
        f"({action_type_counts['partner'] / max(sum(action_type_counts.values()), 1) * 100:.0f}% partner)"
    )

    if failed > 0:
        sys.exit(1)

    # Print one sample task for inspection
    print(f"\n{'=' * 60}")
    print("Sample task (last generated):")
    print(f"{'=' * 60}")
    import json
    print(json.dumps(task, indent=2))


if __name__ == "__main__":
    main()
