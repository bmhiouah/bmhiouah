extends Control

@onready var _title: Label = $Center/VBox/Title
@onready var _subtitle: Label = $Center/VBox/Subtitle
@onready var _start: Button = $Center/VBox/StartButton
@onready var _how: Label = $Center/VBox/HowTo


func _ready() -> void:
	_title.text = "TENSION"
	_subtitle.text = "Local 2-player deduction — Investigate · Deduce · Eliminate"
	_how.text = "PC pass-and-play. One player is Kira (secret cover), the other is L.\nNo networking. Skill and observation — no random roles."
	_start.text = "New Match"
	UiTheme.apply_button(_start)
	_start.pressed.connect(_on_start)
	GameState.reset()


func _on_start() -> void:
	get_tree().change_scene_to_file("res://scenes/role_select.tscn")
