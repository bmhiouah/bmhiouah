extends Node
## Autoload session state. UI scenes only read/write through this API.
## Structured so a later mobile shell can reuse the same core without scene changes.

signal phase_changed(phase: PhaseMachine.Phase)
signal progress_changed
signal clues_changed
signal game_ended(l_won: bool, reason: String)

var phase: PhaseMachine.Phase = PhaseMachine.Phase.SETUP
var kira_cover_id: String = ""
var clues: Array = [] ## { key, value, round }
var used_trait_keys: Array[String] = []
var current_round: int = 1
var investigation_progress: int = 0
var new_world_progress: int = 0
var last_search_zone: String = ""
var last_kira_trait: String = ""
var arrest_id: String = ""
var l_won: bool = false
var end_reason: String = ""
var game_over: bool = false
## Failed arrests this match (tension / history on deduce board).
var failed_arrests: Array[String] = []


func reset() -> void:
	phase = PhaseMachine.Phase.SETUP
	kira_cover_id = ""
	clues.clear()
	used_trait_keys.clear()
	current_round = 1
	investigation_progress = 0
	new_world_progress = 0
	last_search_zone = ""
	last_kira_trait = ""
	arrest_id = ""
	l_won = false
	end_reason = ""
	game_over = false
	failed_arrests.clear()
	phase_changed.emit(phase)
	progress_changed.emit()
	clues_changed.emit()


func set_phase(next: PhaseMachine.Phase) -> void:
	phase = next
	phase_changed.emit(phase)


func set_kira_cover(suspect_id: String) -> void:
	kira_cover_id = suspect_id


func kira_cover() -> Dictionary:
	return Suspects.by_id(kira_cover_id)


func remaining_trait_keys() -> Array[String]:
	var left: Array[String] = []
	for key in Suspects.TRAIT_KEYS:
		if key not in used_trait_keys:
			left.append(key)
	return left


func begin_match() -> void:
	current_round = 1
	set_phase(PhaseMachine.Phase.INVESTIGATE_KIRA)


## --- Investigate (Kira half): leave one true trait trace (Note activity). ---
func kira_leave_trace(trait_key: String) -> Dictionary:
	var cover := kira_cover()
	if cover.is_empty() or trait_key in used_trait_keys:
		return {}
	var clue := {
		"key": trait_key,
		"value": str(cover[trait_key]),
		"round": current_round,
	}
	clues.append(clue)
	used_trait_keys.append(trait_key)
	last_kira_trait = trait_key
	_add_new_world(Rules.kira_act_gain())
	clues_changed.emit()
	if _check_bar_wins():
		return clue
	set_phase(PhaseMachine.Phase.INVESTIGATE_L)
	return clue


## --- Investigate (L half): search a zone for observations. ---
func l_search_zone(zone: String) -> void:
	last_search_zone = zone
	var cover := kira_cover()
	var gain := Rules.investigation_gain(zone, str(cover.get("zone", "")))
	_add_investigation(gain)
	if _check_bar_wins():
		return
	set_phase(PhaseMachine.Phase.DEDUCE)


## --- Deduce → Eliminate transition. ---
func finish_deduce() -> void:
	set_phase(PhaseMachine.Phase.ELIMINATE)


## --- Eliminate: L arrests a cover identity. ---
func arrest(suspect_id: String) -> void:
	if game_over:
		return
	arrest_id = suspect_id
	if suspect_id == kira_cover_id:
		_end_game(true, "L arrested the cover holding the Note.")
		return
	failed_arrests.append(suspect_id)
	_add_new_world(Rules.wrong_arrest_gain())
	if _check_bar_wins():
		return
	## Wrong arrest: next investigation day, or force result if rounds exhausted.
	if current_round >= Rules.MAX_ROUNDS:
		_end_game(false, "Rounds over — Kira's New World holds.")
		return
	current_round += 1
	set_phase(PhaseMachine.Phase.INVESTIGATE_KIRA)


func skip_arrest() -> void:
	## Like a failed majority / skip vote — day advances, tension rises via Kira bar only.
	if game_over:
		return
	if current_round >= Rules.MAX_ROUNDS:
		_end_game(false, "No arrest. Kira slips away.")
		return
	current_round += 1
	set_phase(PhaseMachine.Phase.INVESTIGATE_KIRA)


func possible_suspects() -> Array:
	return Suspects.matching(clues)


func tension_ratio() -> float:
	## Builds each round + with failed arrests.
	var round_t := float(current_round - 1) / float(Rules.MAX_ROUNDS)
	var bar_t := float(maxi(investigation_progress, new_world_progress)) / 100.0
	var fail_t := float(failed_arrests.size()) * 0.12
	return clampf(maxi(round_t, bar_t) + fail_t, 0.0, 1.0)


func _add_investigation(amount: int) -> void:
	investigation_progress = mini(Rules.INVESTIGATION_WIN, investigation_progress + amount)
	progress_changed.emit()


func _add_new_world(amount: int) -> void:
	new_world_progress = mini(Rules.NEW_WORLD_WIN, new_world_progress + amount)
	progress_changed.emit()


func _check_bar_wins() -> bool:
	if investigation_progress >= Rules.INVESTIGATION_WIN:
		_end_game(true, "Investigation Progress filled — Team L wins.")
		return true
	if new_world_progress >= Rules.NEW_WORLD_WIN:
		_end_game(false, "New World Progress filled — Team Kira wins.")
		return true
	return false


func _end_game(team_l_won: bool, reason: String) -> void:
	game_over = true
	l_won = team_l_won
	end_reason = reason
	set_phase(PhaseMachine.Phase.RESULT)
	game_ended.emit(l_won, end_reason)
