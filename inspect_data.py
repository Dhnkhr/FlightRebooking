import json

data = []
with open("artifacts/flight_rebooking_sft_real_large.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

print(f"Total examples: {len(data)}")
print(f"First example message count: {len(data[0]['messages'])}")

# Check message roles distribution
for i, msg in enumerate(data[0]["messages"][:8]):
    role = msg["role"]
    content = msg["content"][:200]
    print(f"\n--- Message {i} (role={role}) ---")
    print(content)

# Check how many examples have few vs many turns
turn_counts = [len(d["messages"]) for d in data]
print(f"\n\nTurn count stats:")
print(f"  Min turns: {min(turn_counts)}")
print(f"  Max turns: {max(turn_counts)}")
print(f"  Avg turns: {sum(turn_counts)/len(turn_counts):.1f}")

# Check content length stats
total_chars = [sum(len(m["content"]) for m in d["messages"]) for d in data]
print(f"\nContent length stats (chars per example):")
print(f"  Min: {min(total_chars)}")
print(f"  Max: {max(total_chars)}")
print(f"  Avg: {sum(total_chars)/len(total_chars):.0f}")

# Check if any assistant responses are just 'finalize'
finalize_only = 0
for d in data:
    assistant_msgs = [m for m in d["messages"] if m["role"] == "assistant"]
    if all("finalize" in m["content"] for m in assistant_msgs):
        finalize_only += 1
print(f"\nExamples where ALL assistant turns are finalize: {finalize_only}/{len(data)}")
