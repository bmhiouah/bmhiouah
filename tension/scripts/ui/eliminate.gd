extends Control
## ELIMINATE phase — Killer Within vote / arrest analogue.
## L chooses who to arrest. Correct cover = L wins. Wrong = New World spike + next day.

@onready var _title: Label = $Margin/VBox/Title
@onready var _hint: Label = $Margin/VBox/Hint
@onready var _grid: GridContainer = $Margin/VBox/Grid
@onready var _skip: Button = $Margin/VBox/SkipButton
@onready var _status: Label = $Margin/VBox/Status


func _ready() -> void:
	_title.text = "ELIMINATE — Arrest vote"
	_hint.text = "L: arrest the cover you believe holds the Note.\nWrong arrest boosts Kira's New World Progress."
	_skip.text = "Skip arrest (advance day)"
	UiTheme.apply_button(_skip)
	_skip.pressed.connect(_on_skip)
	GameState.game_ended.connect(_on_ended)
	GameState.phase_changed.connect(_on_phase)
	_build()


func _on_ended(_l_won: bool, _reason: String) -> void:
	get_tree().change_scene_to_file("res://scenes/result.tscn")


func _on_phase(phase: PhaseMachine.Phase) -> void:
	match phase:
		PhaseMachine.Phase.INVESTIGATE_KIRA:
			get_tree().change_scene_to_file("res://scenes/investigate.tscn")
		PhaseMachine.Phase.RESULT:
			get_tree().change_scene_to_file("res://scenes/result.tscn")
		_:
			pass


func _build() -> void:
	for c in _grid.get_children():
		c.queue_free()
	var possibles := GameState.possible_suspects()
	var possible_ids: Array = []
	for p in possibles:
		possible_ids.append(p["id"])

	for s in Suspects.LIST:
		if s["id"] in GameState.failed_arrests:
			continue
		var btn := Button.new()
		var tag := " (matches clues)" if s["id"] in possible_ids else " (clues disagree)"
		btn.text = "Arrest %s%s" % [s["name"], tag]
		btn.custom_minimum_size = Vector2(240, 72)
		var style := StyleBoxFlat.new()
		style.bg_color = s["color"].darkened(0.3)
		style.corner_radius_top_left = 8
		style.corner_radius_top_right = 8
		style.corner_radius_bottom_left = 8
		style.corner_radius_bottom_right = 8
		btn.add_theme_stylebox_override("normal", style)
		var hover := style.duplicate()
		hover.bg_color = s["color"].darkened(0.1)
		btn.add_theme_stylebox_override("hover", hover)
		btn.add_theme_color_override("font_color", UiTheme.TEXT)
		btn.pressed.connect(_on_arrest.bind(s["id"]))
		_grid.add_child(btn)

	_status.text = "Investigation %d%% · New World %d%% · Round %d/%d" % [
		GameState.investigation_progress,
		GameState.new_world_progress,
		GameState.current_round,
		Rules.MAX_ROUNDS,
	]


func _on_arrest(suspect_id: String) -> void:
	GameState.arrest(suspect_id)


func _on_skip() -> void:
	GameState.skip_arrest()
