class_name PhaseMachine
extends RefCounted
## Killer Within–mapped phase cycle (logic only).
## Investigate → Deduce → Eliminate → (next round | result)

enum Phase {
	SETUP,
	INVESTIGATE_KIRA,
	INVESTIGATE_L,
	DEDUCE,
	ELIMINATE,
	RESULT,
}

const PHASE_NAMES := {
	Phase.SETUP: "Setup",
	Phase.INVESTIGATE_KIRA: "Investigate (Kira)",
	Phase.INVESTIGATE_L: "Investigate (L)",
	Phase.DEDUCE: "Deduce",
	Phase.ELIMINATE: "Eliminate",
	Phase.RESULT: "Result",
}


static func display_name(phase: Phase) -> String:
	return PHASE_NAMES.get(phase, "Unknown")
