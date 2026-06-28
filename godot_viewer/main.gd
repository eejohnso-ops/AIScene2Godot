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
	if _has_room_shell:
		_bake_navmesh(scenes)

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
		# Unshaded + double-sided: surfaces show their (flat, de-lit) albedo from
		# inside the room. Dynamic shading of these thin double-sided shells drops
		# out back/exterior faces, so we keep the simple unshaded look.
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

# --- Navigation -------------------------------------------------------------
# Bake one NavigationMesh across the whole loaded dwelling: floors become
# walkable, walls carve borders, and the sized doorways connect the rooms into a
# single walkable surface. Runs at F5 like the rest of the viewer (no editor
# bake step), draws the result as a translucent overlay, and prints a self-check
# that paths corner-to-corner to prove the rooms connect through the doorways.

const NAV_AGENT_RADIUS := 0.2   # << half the narrowest (0.8m) doorway, so it fits

func _bake_navmesh(scenes: Array[Node]) -> void:
	var nav := NavigationMesh.new()
	nav.cell_size = 0.1
	nav.cell_height = 0.1
	nav.agent_radius = NAV_AGENT_RADIUS
	nav.agent_height = 1.8
	nav.agent_max_climb = 0.2
	nav.agent_max_slope = 45.0

	var src := NavigationMeshSourceGeometryData3D.new()
	var count := 0
	for s in scenes:
		count += _collect_nav_geometry(s, src)
	if count == 0:
		push_warning("Navmesh: no source meshes found; skipping bake.")
		return
	NavigationServer3D.bake_from_source_geometry_data(nav, src)
	var polys := nav.get_polygon_count()
	if polys == 0:
		push_warning("Navmesh: bake produced 0 polygons (check geometry/agent size).")
		return

	var region := NavigationRegion3D.new()
	region.navigation_mesh = nav
	add_child(region)
	print("Navmesh baked: %d source mesh(es) -> %d polygons" % [count, polys])
	_draw_navmesh(nav)
	_verify_connectivity(region)

func _collect_nav_geometry(node: Node, src: NavigationMeshSourceGeometryData3D) -> int:
	# Add every mesh except ceilings (their up-facing top would bake a phantom
	# navmesh layer at ceiling height). Collision bodies are StaticBody3D, not
	# MeshInstance3D, so they aren't double-counted.
	var n := 0
	if node is MeshInstance3D and node.mesh != null:
		if not String(node.name).to_lower().contains("ceiling"):
			src.add_mesh(node.mesh, (node as Node3D).global_transform)
			n += 1
	for c in node.get_children():
		n += _collect_nav_geometry(c, src)
	return n

func _draw_navmesh(nav: NavigationMesh) -> void:
	var verts := nav.get_vertices()
	if verts.is_empty():
		return
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	for i in range(nav.get_polygon_count()):
		var poly := nav.get_polygon(i)
		for k in range(1, poly.size() - 1):   # fan-triangulate
			st.add_vertex(verts[poly[0]])
			st.add_vertex(verts[poly[k]])
			st.add_vertex(verts[poly[k + 1]])
	var mi := MeshInstance3D.new()
	mi.mesh = st.commit()
	var mat := StandardMaterial3D.new()
	mat.albedo_color = Color(0.1, 0.6, 1.0, 0.35)
	mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	mat.cull_mode = BaseMaterial3D.CULL_DISABLED
	mi.material_override = mat
	mi.position.y = _room_min.y + 0.03   # lift off the floor to avoid z-fighting
	add_child(mi)

func _verify_connectivity(region: NavigationRegion3D) -> void:
	# The navigation map syncs on the physics frame, so wait before querying.
	await get_tree().physics_frame
	await get_tree().physics_frame
	var map := region.get_navigation_map()
	var fy := _room_min.y + 0.1
	var a := Vector3(_room_min.x + 1.0, fy, _room_min.z + 1.0)
	var b := Vector3(_room_max.x - 1.0, fy, _room_max.z - 1.0)
	var path := NavigationServer3D.map_get_path(map, a, b, true)
	var reached := path.size() >= 2 and path[path.size() - 1].distance_to(b) < 1.5
	if reached:
		var d := 0.0
		for i in range(1, path.size()):
			d += path[i].distance_to(path[i - 1])
		print("Navmesh connectivity OK: %d-point path, %.1fm corner-to-corner " % [path.size(), d]
			+ "(rooms connect through the doorways).")
	else:
		push_warning("Navmesh connectivity FAILED: no corner-to-corner path. "
			+ "Doorways may be narrower than 2*agent_radius (%.2fm)." % (2.0 * NAV_AGENT_RADIUS))
