extends Control
## DEDUCE phase — Killer Within discussion / evidence report analogue.
## Shared screen: both players review clues, possibles, and bars before arrest.

@onready var _title: Label = $Margin/VBox/Title
@onready var _clues: RichTextLabel = $Margin/VBox/Clues
@onready var _board: GridContainer = $Margin/VBox/Board
@onready var _status: Label = $Margin/VBox/Status
@onready var _continue: Button = $Margin/VBox/ContinueButton


func _ready() -> void:
	_title.text = "DEDUCE — Evidence board"
	_continue.text = "Proceed to Eliminate (arrest vote)"
	UiTheme.apply_button(_continue)
	_continue.pressed.connect(_on_continue)
	_render()


func _render() -> void:
	var lines: PackedStringArray = []
	lines.append("[b]Public traces[/b] (Kira Note activity — always true):")
	if GameState.clues.is_empty():
		lines.append("  (none yet)")
	else:
		for c in GameState.clues:
			lines.append(
				"  Round %d — %s: [color=#2BB8CC]%s[/color]"
				% [c["round"], Suspects.trait_label(c["key"]), c["value"]]
			)
	if not GameState.last_search_zone.is_empty():
		lines.append("")
		lines.append("L last searched: [b]%s[/b]" % GameState.last_search_zone)
	if GameState.failed_arrests.size() > 0:
		lines.append("")
		lines.append("Failed arrests: %s" % ", ".join(PackedStringArray(GameState.failed_arrests)))
	_clues.text = "\n".join(lines)

	for child in _board.get_children():
		child.queue_free()
	var possibles := GameState.possible_suspects()
	var possible_ids: Array = []
	for p in possibles:
		possible_ids.append(p["id"])

	for s in Suspects.LIST:
		var panel := PanelContainer.new()
		var style := StyleBoxFlat.new()
		var eliminated := s["id"] in GameState.failed_arrests
		var still_in := s["id"] in possible_ids and not eliminated
		style.bg_color = s["color"].darkened(0.45 if still_in else 0.7)
		style.corner_radius_top_left = 8
		style.corner_radius_top_right = 8
		style.corner_radius_bottom_left = 8
		style.corner_radius_bottom_right = 8
		style.content_margin_left = 10
		style.content_margin_right = 10
		style.content_margin_top = 10
		style.content_margin_bottom = 10
		panel.add_theme_stylebox_override("panel", style)
		var lbl := Label.new()
		var mark := "SUSPECT" if still_in else ("CLEARED" if eliminated else "ELIMINATED BY CLUES")
		lbl.text = "%s\n%s / %s / %s\n[%s]" % [s["name"], s["zone"], s["manner"], s["tool"], mark]
		lbl.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
		panel.add_child(lbl)
		_board.add_child(panel)

	_status.text = "Round %d/%d | Investigation %d%% | New World %d%% | Tension %d%% | %d still possible" % [
		GameState.current_round,
		Rules.MAX_ROUNDS,
		GameState.investigation_progress,
		GameState.new_world_progress,
		int(GameState.tension_ratio() * 100.0),
		possible_ids.size(),
	]


func _on_continue() -> void:
	GameState.finish_deduce()
	get_tree().change_scene_to_file("res://scenes/eliminate.tscn")
