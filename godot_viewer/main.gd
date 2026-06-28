extends Node3D
# Builds the whole viewer at runtime so the .tscn files stay tiny and robust:
#   - input actions (WASD)            - environment + ambient light
#   - loads ALL .glb files            - trimesh static collision per object
#   - makes materials unshaded + double-sided (visible from INSIDE)
#   - a temporary "catch floor" so you can't fall through a distorted nadir
#   - spawns the first-person player at the room's computed center
#
# Spawn point and catch-floor are derived from the combined bounding box of all
# loaded scenes. Drop any number of .glb files in the project folder and press F5.

var _room_min := Vector3.INF
var _room_max := -Vector3.INF
var _spawn_pos := Vector3.ZERO
var _have_spawn := false
var _has_room_shell := false

func _ready() -> void:
	_setup_input()
	_setup_environment()
	var scenes := _load_scenes()
	for s in scenes:
		_compute_bounds(s)
	if not _has_room_shell:
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
	env.background_color = Color(0.45, 0.50, 0.58)
	env.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	env.ambient_light_color = Color(1, 1, 1)
	env.ambient_light_energy = 0.5
	var we := WorldEnvironment.new()
	we.environment = env
	add_child(we)
	var sun := DirectionalLight3D.new()
	sun.rotation_degrees = Vector3(-50, -35, 0)
	sun.light_energy = 1.0
	add_child(sun)

func _find_all_glbs() -> Array[String]:
	# Find the newest project subfolder (each build_scene.py run creates one).
	# Falls back to root-level .glb files for backward compat with to_godot.py.
	var dir := DirAccess.open("res://")
	if not dir:
		return []

	# Scan subdirectories for the newest .glb
	var best_folder := ""
	var best_time := -1
	var root_glbs: Array[String] = []

	dir.list_dir_begin()
	var f := dir.get_next()
	while f != "":
		if dir.current_is_dir() and not f.begins_with("."):
			var sub := DirAccess.open("res://" + f)
			if sub:
				sub.list_dir_begin()
				var sf := sub.get_next()
				while sf != "":
					if not sub.current_is_dir() and sf.to_lower().ends_with(".glb"):
						var full := "res://" + f + "/" + sf
						var t := FileAccess.get_modified_time(full)
						if t > best_time:
							best_time = t
							best_folder = f
					sf = sub.get_next()
				sub.list_dir_end()
		elif not dir.current_is_dir() and f.to_lower().ends_with(".glb"):
			root_glbs.append("res://" + f)
		f = dir.get_next()
	dir.list_dir_end()

	# Prefer the newest project subfolder over root-level files
	if best_folder != "":
		var glbs: Array[String] = []
		var sub := DirAccess.open("res://" + best_folder)
		if sub:
			sub.list_dir_begin()
			var sf := sub.get_next()
			while sf != "":
				if not sub.current_is_dir() and sf.to_lower().ends_with(".glb"):
					glbs.append("res://" + best_folder + "/" + sf)
				sf = sub.get_next()
			sub.list_dir_end()
		# Check if any root-level glb is newer (e.g. a to_godot.py drop)
		for rg in root_glbs:
			if FileAccess.get_modified_time(rg) > best_time:
				glbs.append(rg)
		print("Project: ", best_folder)
		glbs.sort()
		return glbs

	root_glbs.sort()
	return root_glbs

func _load_scenes() -> Array[Node]:
	var paths := _find_all_glbs()
	if paths.is_empty():
		push_error("No .glb found in the project folder. Run to_godot.py or room_from_image.py first.")
		return []
	var nodes: Array[Node] = []
	for path in paths:
		print("Loading: ", path)
		if path.get_file().to_lower().contains("room"):
			_has_room_shell = true
		var packed := load(path) as PackedScene
		if packed == null:
			push_warning("Could not load " + path + " (Godot may still be importing it).")
			continue
		var node := packed.instantiate()
		add_child(node)
		_process_meshes(node)
		nodes.append(node)
	return nodes

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
	# Accumulates min/max across all loaded scenes. Uses per-axis median of all
	# vertices as the spawn point (robust when geometry is lopsided).
	var xs := PackedFloat32Array()
	var ys := PackedFloat32Array()
	var zs := PackedFloat32Array()
	_gather(node, xs, ys, zs)
	if xs.is_empty():
		return
	xs.sort(); ys.sort(); zs.sort()
	var n := xs.size()
	var mid := n >> 1
	var node_min := Vector3(xs[0], ys[0], zs[0])
	var node_max := Vector3(xs[n - 1], ys[n - 1], zs[n - 1])
	_room_min = Vector3(min(_room_min.x, node_min.x),
						min(_room_min.y, node_min.y),
						min(_room_min.z, node_min.z))
	_room_max = Vector3(max(_room_max.x, node_max.x),
						max(_room_max.y, node_max.y),
						max(_room_max.z, node_max.z))
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
	var body := StaticBody3D.new()
	var col := CollisionShape3D.new()
	var plane := WorldBoundaryShape3D.new()
	plane.plane = Plane(Vector3.UP, fy)
	col.shape = plane
	body.add_child(col)
	add_child(body)
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
	# Spawn at the room entrance (max Z = where the source camera was), facing
	# inward (-Z). This makes the scene look like the original photo on load.
	var player := preload("res://player.tscn").instantiate()
	if _have_spawn:
		var eye_y := _room_min.y + 1.6
		var entrance_z := _room_max.z - 0.8
		player.position = Vector3(0.0, eye_y, entrance_z)
		# rotation.y = 0: Godot's default forward is -Z, which points into the room
	else:
		player.position = Vector3(0.0, 0.5, 0.0)
	add_child(player)
