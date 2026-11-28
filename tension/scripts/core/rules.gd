class_name Rules
extends RefCounted
## Win / progress rules adapted from Killer Within (logic only).
## Team L: arrest the cover identity holding the Note, or fill Investigation bar.
## Team Kira: survive arrests + fill New World bar via judged actions.

const MAX_ROUNDS := 3
const INVESTIGATION_WIN := 100
const NEW_WORLD_WIN := 100

## L searching the impostor's true zone yields stronger investigation progress.
const L_SEARCH_HIT := 34
const L_SEARCH_MISS := 18
## Kira leaking a true trait advances New World (they must act each day).
const KIRA_ACT_PROGRESS := 28
## Wrong arrest hands Kira a large New World spike (KW: failed arrest / exposure risk).
const WRONG_ARREST_BONUS := 40


static func investigation_gain(searched_zone: String, impostor_zone: String) -> int:
	if searched_zone == impostor_zone:
		return L_SEARCH_HIT
	return L_SEARCH_MISS


static func kira_act_gain() -> int:
	return KIRA_ACT_PROGRESS


static func wrong_arrest_gain() -> int:
	return WRONG_ARREST_BONUS
