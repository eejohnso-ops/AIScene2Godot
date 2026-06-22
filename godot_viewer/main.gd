extends Node3D
# Builds the whole viewer at runtime so the .tscn files stay tiny and robust:
#   - input actions (WASD)            - environment + ambient light
#   - loads cargo_bay.glb             - trimesh static collision off the room
#   - makes room materials unshaded + double-sided (visible from INSIDE)
#   - a temporary "catch floor" so you can't fall through a distorted nadir
#   - spawns the first-person player at the room's computed center
#
# Spawn point and catch-floor are derived from the room's bounding box, so this
# works unchanged for any room GLB you drop in (just rename it or edit ROOM_PATH).

var _room_min := Vector3.INF
var _room_max := -Vector3.INF
var _spawn_pos := Vector3.ZERO
var _have_spawn := false

func _ready() -> void:
	_setup_input()
	_setup_environment()
	var room := _load_room()
	if room:
		_compute_bounds(room)
	_add_catch_floor()
	_spawn_player()

func _setup_input() -> void:
	var keys := {"move_forward": KEY_W, "move_back": KEY_S,
				 "move_left": KEY_A, "move_right": KEY_D}
	for action in keys:
		if not InputMap.has_action(action):
			InputMap.add_action(action)
			var ev := InputEventKey.new()
			ev.physical_keycode = keys[action]
			InputMap.action_add_event(action, ev)

func _setup_environment() -> void:
	var env := Environment.new()
	env.background_mode = Environment.BG_COLOR
	env.background_color = Color(0.45, 0.50, 0.58)   # soft neutral, not black
	env.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	env.ambient_light_color = Color(1, 1, 1)
	env.ambient_light_energy = 0.5                    # lower, so the sun adds contrast
	var we := WorldEnvironment.new()
	we.environment = env
	add_child(we)
	# A sun so untextured (e.g. MIDI) geometry shows form instead of flat grey.
	# (Unshaded/baked-texture meshes ignore this and keep showing their texture.)
	var sun := DirectionalLight3D.new()
	sun.rotation_degrees = Vector3(-50, -35, 0)
	sun.light_energy = 1.0
	add_child(sun)

func _find_newest_glb() -> String:
	# Load whichever .glb in the project folder was built most recently, so
	# build_room.py's auto-copy "just works" with no renaming.
	var dir := DirAccess.open("res://")
	var newest := ""
	var newest_time := -1
	if dir:
		dir.list_dir_begin()
		var f := dir.get_next()
		while f != "":
			if not dir.current_is_dir() and f.to_lower().ends_with(".glb"):
				var t := FileAccess.get_modified_time("res://" + f)
				if t > newest_time:
					newest_time = t
					newest = "res://" + f
			f = dir.get_next()
		dir.list_dir_end()
	return newest

func _load_room() -> Node:
	var room_path := _find_newest_glb()
	if room_path == "":
		push_error("No .glb found in the project folder. Run build_room.py first.")
		return null
	print("Loading room: ", room_path)
	var packed := load(room_path) as PackedScene
	if packed == null:
		push_error("Could not load " + room_path + " (Godot may still be importing it).")
		return null
	var room := packed.instantiate()
	add_child(room)
	_process_meshes(room)
	return room

# Recursively: add trimesh collision + force materials unshaded & double-sided.
func _process_meshes(node: Node) -> void:
	if node is MeshInstance3D and node.mesh != null:
		node.create_trimesh_collision()
		for i in range(node.mesh.get_surface_count()):
			var m = node.mesh.surface_get_material(i)
			if m is BaseMaterial3D:
				m.cull_mode = BaseMaterial3D.CULL_DISABLED
				m.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	for c in node.get_children():
		_process_meshes(c)

func _compute_bounds(node: Node) -> void:
	# Collect all world-space vertices, then take the per-axis MEDIAN as the spawn
	# point. MoGe floor-stretch makes the mesh lopsided, so the bbox center can
	# land outside the walls (you end up looking at the room as an object). The
	# median sits in the dense room core, which is where you actually want to be.
	var xs := PackedFloat32Array()
	var ys := PackedFloat32Array()
	var zs := PackedFloat32Array()
	_gather(node, xs, ys, zs)
	if xs.is_empty():
		return
	xs.sort(); ys.sort(); zs.sort()
	var n := xs.size()
	var mid := n >> 1
	_room_min = Vector3(xs[0], ys[0], zs[0])
	_room_max = Vector3(xs[n - 1], ys[n - 1], zs[n - 1])
	_spawn_pos = Vector3(xs[mid], ys[mid], zs[mid])
	_have_spawn = true

func _gather(node: Node, xs: PackedFloat32Array, ys: PackedFloat32Array, zs: PackedFloat32Array) -> void:
	if node is MeshInstance3D and node.mesh != null:
		var t: Transform3D = (node as Node3D).global_transform
		for s in range(node.mesh.get_surface_count()):
			var arr = node.mesh.surface_get_arrays(s)
			var verts: PackedVector3Array = arr[Mesh.ARRAY_VERTEX]
			for v in verts:
				var w: Vector3 = t * v
				xs.append(w.x)
				ys.append(w.y)
				zs.append(w.z)
	for c in node.get_children():
		_gather(c, xs, ys, zs)

func _floor_y() -> float:
	return _room_min.y if _have_spawn else -2.0

func _add_catch_floor() -> void:
	var fy := _floor_y()
	# Collision plane at floor level so you can stand/walk.
	var body := StaticBody3D.new()
	var col := CollisionShape3D.new()
	var plane := WorldBoundaryShape3D.new()
	plane.plane = Plane(Vector3.UP, fy)
	col.shape = plane
	body.add_child(col)
	add_child(body)
	# Visible ground so objects rest on something instead of floating in void.
	var vis := MeshInstance3D.new()
	var pm := PlaneMesh.new()
	pm.size = Vector2(40, 40)
	vis.mesh = pm
	var mat := StandardMaterial3D.new()
	mat.albedo_color = Color(0.30, 0.32, 0.36)
	vis.material_override = mat
	vis.position = Vector3(0, fy, 0)
	add_child(vis)

func _spawn_player() -> void:
	# Spawn at the WORLD ORIGIN. The reconstruction is camera-relative, so (0,0,0)
	# is exactly where the panorama was captured -- guaranteed inside the room,
	# unlike the bbox/median center which drifts with the floor-stretch smear.
	# Start a little above origin and let gravity settle onto the floor.
	var player := preload("res://player.tscn").instantiate()
	player.position = Vector3(0, 0.5, 0)
	add_child(player)
