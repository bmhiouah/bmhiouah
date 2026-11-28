extends Control
## Secret cover selection — Investigator (L) looks away. Deterministic, player-chosen.

@onready var _prompt: Label = $Margin/VBox/Prompt
@onready var _grid: GridContainer = $Margin/VBox/Grid
@onready var _confirm_wrap: VBoxContainer = $Margin/VBox/ConfirmWrap
@onready var _confirm_label: Label = $Margin/VBox/ConfirmWrap/ConfirmLabel
@onready var _hand_off: Button = $Margin/VBox/ConfirmWrap/HandOffButton
@onready var _back: Button = $Margin/VBox/BackButton

var _selected: String = ""
var _locked: bool = false


func _ready() -> void:
	_prompt.text = "L: look away.\nKira: secretly choose your cover identity among the four."
	_confirm_wrap.visible = false
	_hand_off.text = "Cover chosen — hand off to L"
	UiTheme.apply_button(_hand_off)
	UiTheme.apply_button(_back)
	_back.text = "Back"
	_hand_off.pressed.connect(_on_hand_off)
	_back.pressed.connect(func() -> void:
		get_tree().change_scene_to_file("res://scenes/main_menu.tscn")
	)
	_build_grid()


func _build_grid() -> void:
	for child in _grid.get_children():
		child.queue_free()
	for s in Suspects.LIST:
		var btn := Button.new()
		btn.text = "%s\n%s · %s · %s" % [s["name"], s["zone"], s["manner"], s["tool"]]
		btn.custom_minimum_size = Vector2(220, 100)
		var style := StyleBoxFlat.new()
		style.bg_color = s["color"].darkened(0.35)
		style.corner_radius_top_left = 8
		style.corner_radius_top_right = 8
		style.corner_radius_bottom_left = 8
		style.corner_radius_bottom_right = 8
		style.content_margin_left = 12
		style.content_margin_right = 12
		style.content_margin_top = 12
		style.content_margin_bottom = 12
		btn.add_theme_stylebox_override("normal", style)
		var hover := style.duplicate()
		hover.bg_color = s["color"].darkened(0.15)
		btn.add_theme_stylebox_override("hover", hover)
		btn.add_theme_color_override("font_color", UiTheme.TEXT)
		var id: String = s["id"]
		btn.pressed.connect(_on_pick.bind(id, s["name"]))
		_grid.add_child(btn)


func _on_pick(suspect_id: String, suspect_name: String) -> void:
	if _locked:
		return
	_selected = suspect_id
	_locked = true
	_grid.visible = false
	_prompt.visible = false
	_confirm_wrap.visible = true
	_confirm_label.text = "Cover locked: %s\n(Keep this secret. Clear the screen before L returns.)" % suspect_name


func _on_hand_off() -> void:
	if _selected.is_empty():
		return
	GameState.set_kira_cover(_selected)
	GameState.begin_match()
	get_tree().change_scene_to_file("res://scenes/investigate.tscn")
