class_name UiTheme
extends RefCounted
## Shared visual tokens for PC UI. Swap or restyle later for mobile without touching core.

const BG := Color(0.07, 0.08, 0.1)
const PANEL := Color(0.12, 0.14, 0.18)
const ACCENT := Color(0.17, 0.72, 0.8)
const DANGER := Color(0.76, 0.23, 0.23)
const TEXT := Color(0.91, 0.94, 0.96)
const MUTED := Color(0.55, 0.62, 0.68)


static func apply_button(btn: Button, danger: bool = false) -> void:
	btn.custom_minimum_size = Vector2(160, 44)
	var style := StyleBoxFlat.new()
	style.bg_color = DANGER if danger else ACCENT
	style.corner_radius_top_left = 6
	style.corner_radius_top_right = 6
	style.corner_radius_bottom_left = 6
	style.corner_radius_bottom_right = 6
	style.content_margin_left = 16
	style.content_margin_right = 16
	style.content_margin_top = 10
	style.content_margin_bottom = 10
	btn.add_theme_stylebox_override("normal", style)
	var hover := style.duplicate()
	hover.bg_color = style.bg_color.lightened(0.12)
	btn.add_theme_stylebox_override("hover", hover)
	btn.add_theme_color_override("font_color", Color(0.05, 0.07, 0.09) if not danger else TEXT)
