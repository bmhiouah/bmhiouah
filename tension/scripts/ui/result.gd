extends Control

@onready var _title: Label = $Margin/VBox/Title
@onready var _reason: Label = $Margin/VBox/Reason
@onready var _detail: Label = $Margin/VBox/Detail
@onready var _again: Button = $Margin/VBox/AgainButton
@onready var _menu: Button = $Margin/VBox/MenuButton


func _ready() -> void:
	var cover := GameState.kira_cover()
	if GameState.l_won:
		_title.text = "TEAM L WINS"
		_title.add_theme_color_override("font_color", UiTheme.ACCENT)
	else:
		_title.text = "TEAM KIRA WINS"
		_title.add_theme_color_override("font_color", UiTheme.DANGER)
	_reason.text = GameState.end_reason
	_detail.text = "Kira's cover was %s (%s / %s / %s).\nArrest attempted: %s\nFinal bars — Investigation %d%% · New World %d%%" % [
		cover.get("name", "?"),
		cover.get("zone", "?"),
		cover.get("manner", "?"),
		cover.get("tool", "?"),
		GameState.arrest_id if not GameState.arrest_id.is_empty() else "(none)",
		GameState.investigation_progress,
		GameState.new_world_progress,
	]
	_again.text = "Play again"
	_menu.text = "Main menu"
	UiTheme.apply_button(_again)
	UiTheme.apply_button(_menu)
	_again.pressed.connect(func() -> void:
		GameState.reset()
		get_tree().change_scene_to_file("res://scenes/role_select.tscn")
	)
	_menu.pressed.connect(func() -> void:
		GameState.reset()
		get_tree().change_scene_to_file("res://scenes/main_menu.tscn")
	)
