class_name Suspects
extends RefCounted
## Roster of cover identities. Pure data — no UI. Safe to reuse on a future mobile port.

const TRAIT_KEYS := ["zone", "manner", "tool"]

const LIST := [
	{
		"id": "aya",
		"name": "Aya",
		"color": Color(0.86, 0.28, 0.32),
		"zone": "Lab",
		"manner": "Quiet",
		"tool": "Notebook",
	},
	{
		"id": "ren",
		"name": "Ren",
		"color": Color(0.25, 0.55, 0.92),
		"zone": "Hall",
		"manner": "Bold",
		"tool": "Watch",
	},
	{
		"id": "kai",
		"name": "Kai",
		"color": Color(0.28, 0.72, 0.42),
		"zone": "Lab",
		"manner": "Bold",
		"tool": "Glasses",
	},
	{
		"id": "mio",
		"name": "Mio",
		"color": Color(0.92, 0.72, 0.22),
		"zone": "Hall",
		"manner": "Quiet",
		"tool": "Notebook",
	},
]


static func by_id(suspect_id: String) -> Dictionary:
	for s in LIST:
		if s["id"] == suspect_id:
			return s
	return {}


static func trait_label(key: String) -> String:
	match key:
		"zone":
			return "Zone"
		"manner":
			return "Manner"
		"tool":
			return "Tool"
		_:
			return key.capitalize()


static func matching(clues: Array) -> Array:
	var out: Array = []
	for s in LIST:
		var ok := true
		for c in clues:
			if str(s[c["key"]]) != str(c["value"]):
				ok = false
				break
		if ok:
			out.append(s)
	return out
