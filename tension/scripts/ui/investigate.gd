extends Control
## INVESTIGATE phase — Killer Within day/task analogue.
## Kira acts first (L looks away), then L searches a zone.

@onready var _phase_label: Label = $Margin/VBox/PhaseLabel
@onready var _hint: Label = $Margin/VBox/Hint
@onready var _actions: VBoxContainer = $Margin/VBox/Actions
@onready var _progress: Label = $Margin/VBox/Progress
@onready var _cover: ColorRect = $Cover
@onready var _cover_label: Label = $Cover/CoverLabel
@onready var _cover_btn: Button = $Cover/CoverButton


func _ready() -> void:
	GameState.phase_changed.connect(_on_phase)
	GameState.progress_changed.connect(_refresh_progress)
	GameState.game_ended.connect(_on_ended)
	_cover_btn.text = "Ready — continue"
	UiTheme.apply_button(_cover_btn)
	_cover_btn.pressed.connect(_hide_cover)
	_refresh_progress()
	_on_phase(GameState.phase)


func _on_ended(_l_won: bool, _reason: String) -> void:
	get_tree().change_scene_to_file("res://scenes/result.tscn")


func _on_phase(phase: PhaseMachine.Phase) -> void:
	match phase:
		PhaseMachine.Phase.INVESTIGATE_KIRA:
			_show_cover("Pass to Kira.\nL: look away.")
			_setup_kira_actions()
		PhaseMachine.Phase.INVESTIGATE_L:
			_show_cover("Pass to L.\nKira: look away / step back.")
			_setup_l_actions()
		PhaseMachine.Phase.DEDUCE:
			get_tree().change_scene_to_file("res://scenes/deduce.tscn")
		PhaseMachine.Phase.RESULT:
			get_tree().change_scene_to_file("res://scenes/result.tscn")
		_:
			pass


func _show_cover(text: String) -> void:
	_cover.visible = true
	_cover_label.text = text


func _hide_cover() -> void:
	_cover.visible = false


func _refresh_progress() -> void:
	_progress.text = "Round %d/%d    Investigation %d%%    New World %d%%    Tension %d%%" % [
		GameState.current_round,
		Rules.MAX_ROUNDS,
		GameState.investigation_progress,
		GameState.new_world_progress,
		int(GameState.tension_ratio() * 100.0),
	]


func _clear_actions() -> void:
	for c in _actions.get_children():
		c.queue_free()


func _setup_kira_actions() -> void:
	_phase_label.text = "INVESTIGATE — Kira"
	_hint.text = "Use the Note carefully: leak ONE true trait about your cover.\nPick a trait that keeps multiple suspects possible."
	_clear_actions()
	var left := GameState.remaining_trait_keys()
	if left.is_empty():
		## Should not happen within MAX_ROUNDS == trait count, but guard anyway.
		GameState.set_phase(PhaseMachine.Phase.INVESTIGATE_L)
		return
	for key in left:
		var cover := GameState.kira_cover()
		var btn := Button.new()
		btn.text = "Leave trace: %s = %s" % [Suspects.trait_label(key), cover[key]]
		UiTheme.apply_button(btn, true)
		btn.pressed.connect(_on_kira_trace.bind(key))
		_actions.add_child(btn)


func _on_kira_trace(key: String) -> void:
	GameState.kira_leave_trace(key)


func _setup_l_actions() -> void:
	_phase_label.text = "INVESTIGATE — L"
	_hint.text = "Search a zone. Hitting Kira's true zone yields stronger Investigation Progress.\nWatch for patterns across rounds — no dice rolls."
	_clear_actions()
	for zone in ["Lab", "Hall"]:
		var btn := Button.new()
		btn.text = "Search %s" % zone
		UiTheme.apply_button(btn)
		btn.pressed.connect(_on_l_search.bind(zone))
		_actions.add_child(btn)


func _on_l_search(zone: String) -> void:
	GameState.l_search_zone(zone)
