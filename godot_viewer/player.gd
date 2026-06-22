extends CharacterBody3D
# Inspection controller with two modes:
#   FLY (default): noclip free-flight, move along the look direction. Best for
#                  judging a rough Phase-1 shell without fighting blob collision.
#   WALK:          first-person CharacterBody with gravity + collision.
# Controls: mouse = look, WASD = move, Space/Shift = up/down (fly),
#           F = toggle fly/walk, Esc = release mouse.

@export var speed: float = 4.0          # walk speed
@export var fly_speed: float = 8.0      # fly speed
@export var mouse_sensitivity: float = 0.0025
@export var gravity: float = 12.0

var _pitch: float = 0.0
var _fly: bool = true                   # start in fly so blobs don't trap you

func _ready() -> void:
	Input.mouse_mode = Input.MOUSE_MODE_CAPTURED

func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventMouseMotion and Input.mouse_mode == Input.MOUSE_MODE_CAPTURED:
		rotate_y(-event.relative.x * mouse_sensitivity)
		_pitch = clampf(_pitch - event.relative.y * mouse_sensitivity, -1.4, 1.4)
		$Camera3D.rotation.x = _pitch
	elif event.is_action_pressed("ui_cancel"):
		Input.mouse_mode = (Input.MOUSE_MODE_VISIBLE
			if Input.mouse_mode == Input.MOUSE_MODE_CAPTURED
			else Input.MOUSE_MODE_CAPTURED)
	elif event is InputEventKey and event.pressed and event.keycode == KEY_F:
		_fly = not _fly
		velocity = Vector3.ZERO

func _physics_process(delta: float) -> void:
	if _fly:
		_fly_move(delta)
	else:
		_walk_move(delta)

func _fly_move(delta: float) -> void:
	# Noclip: move freely along the camera's look direction, no gravity/collision.
	var cam := $Camera3D
	var input := Input.get_vector("move_left", "move_right", "move_forward", "move_back")
	var dir := Vector3.ZERO
	dir += cam.global_transform.basis.x * input.x          # strafe
	dir += cam.global_transform.basis.z * input.y          # forward/back (basis.z = backward)
	if Input.is_key_pressed(KEY_SPACE):
		dir += Vector3.UP
	if Input.is_key_pressed(KEY_SHIFT):
		dir += Vector3.DOWN
	if dir.length() > 0.0:
		global_position += dir.normalized() * fly_speed * delta

func _walk_move(delta: float) -> void:
	if not is_on_floor():
		velocity.y -= gravity * delta
	var input_dir := Input.get_vector("move_left", "move_right", "move_forward", "move_back")
	var dir := (transform.basis * Vector3(input_dir.x, 0, input_dir.y)).normalized()
	if dir.length() > 0.0:
		velocity.x = dir.x * speed
		velocity.z = dir.z * speed
	else:
		velocity.x = move_toward(velocity.x, 0, speed)
		velocity.z = move_toward(velocity.z, 0, speed)
	move_and_slide()
