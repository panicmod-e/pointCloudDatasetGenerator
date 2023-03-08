bl_info = {
    # required
    "name": "Dataset Generator",
    "blender": (3, 0, 0),
    "category": "Object",
    # optional
    "version": (0,1,0),
    "author": "David Schlereth",
    "description": "Addon to automate point cloud dataset creation of a city using the vLiDAR Addon",
}

import bpy
import numpy as np
from mathutils import Vector, Matrix
from math import radians
from collections import deque
import time

# GLOBAL VARIABLES

path_method_items = [
    ('SINGLE', "Single", "Generate single random path"),
    ('MULTIPLE', "Multiple", "Path from multiple randomly generated paths"),
    ('DFS', "DFS traversal", "Path from dfs"),
    ('BFS', "BFS traversal", "Path from bfs"),
    # ('NEIGHBORS_FROM_NODE', "Multiple from node", "Path with fixed number of neighbors traversed"),
    # ('ALL_FROM_NODE', "All from node", "Path from all paths starting in random node"),
]

path_selection_items = [
    ('LONGEST', "Longest Path", "Select longest of all generated Paths"),
    ('RANDOM', "Random Path", "Select path randomly from generated"),
]

PROPS = [
    ("city_collection", bpy.props.StringProperty(name="City Collection", default="city_generated")),
    ("generator_settings_expanded", bpy.props.BoolProperty(name="Subpanel status", default=True)),
    ("city_settings_expanded", bpy.props.BoolProperty(name="Subpanel status", default=True)),
    ("object_modifier_tags", bpy.props.StringProperty(name="Object tags", default="Prop, Modifier, instance, building")),
    ("building_modifier_tags", bpy.props.StringProperty(name="Building tags", default="Modifier, instance, building")),
    ("object_classes", bpy.props.StringProperty(name="Object classes", default="initial, new, removed, moved, rotated, scaled")),
]


class DatasetGeneratorScanSettings(bpy.types.PropertyGroup):
    scans: bpy.props.IntProperty(name="Scans", default=4)
    remove_objects_enable: bpy.props.BoolProperty(name="Remove objects", default=True)
    remove_objects_min: bpy.props.IntProperty(name="min", default=10, min=0, soft_max=100)
    remove_objects_max: bpy.props.IntProperty(name="max", default=20, min=0, soft_max=100)
    add_objects_enable: bpy.props.BoolProperty(name="Add objects", default=True)
    add_objects_min: bpy.props.IntProperty(name="min", default=10, min=0, soft_max=100)
    add_objects_max: bpy.props.IntProperty(name="max", default=20, min=0, soft_max=100)
    rotation_enable: bpy.props.BoolProperty(name="Random rotation", default=True)
    rotation_min: bpy.props.FloatProperty(name="min", default=1.0, min=0.0, soft_max=5.0, step=1)
    rotation_max: bpy.props.FloatProperty(name="max", default=10.0, min=0.0, soft_max=15.0, step=1)
    rotation_objects_min: bpy.props.IntProperty(name="min", default=10, min=0, soft_max=100)
    rotation_objects_max: bpy.props.IntProperty(name="max", default=20, min=0, soft_max=100)
    rotation_positive_x: bpy.props.BoolProperty(name="+X", default=False)
    rotation_negative_x: bpy.props.BoolProperty(name="-X", default=False)
    rotation_positive_y: bpy.props.BoolProperty(name="+Y", default=False)
    rotation_negative_y: bpy.props.BoolProperty(name="-Y", default=False)
    rotation_positive_z: bpy.props.BoolProperty(name="+Z", default=True)
    rotation_negative_z: bpy.props.BoolProperty(name="-Z", default=True)
    translation_enable: bpy.props.BoolProperty(name="Random translation", default=True)
    translation_min: bpy.props.FloatProperty(name="min", default=0.05, min=0.0, soft_max=0.5, step=0.1)
    translation_max: bpy.props.FloatProperty(name="max", default=0.1, min=0.0, soft_max=0.5, step=0.1)
    translation_objects_min: bpy.props.IntProperty(name="min", default=10, min=0, soft_max=100)
    translation_objects_max: bpy.props.IntProperty(name="max", default=20, min=0, soft_max=100)
    translation_negative_x: bpy.props.BoolProperty(name="-X", default=True)
    translation_positive_x: bpy.props.BoolProperty(name="+X", default=True)
    translation_positive_y: bpy.props.BoolProperty(name="+Y", default=True)
    translation_negative_y: bpy.props.BoolProperty(name="-Y", default=True)
    translation_positive_z: bpy.props.BoolProperty(name="+Z", default=False)
    translation_negative_z: bpy.props.BoolProperty(name="-Z", default=True)
    scale_enable: bpy.props.BoolProperty(name="Random scaling", default=True)
    scale_uniform: bpy.props.BoolProperty(name="Uniform scaling", default=True)
    scale_min: bpy.props.FloatProperty(name="min", default=0.9, min=0.0, soft_min=0.8, soft_max=1.2, step=1)
    scale_max: bpy.props.FloatProperty(name="max", default=1.1, min=0.0, soft_min=0.8, soft_max=1.2, step=1)
    scale_objects_min: bpy.props.IntProperty(name="min", default=10, min=0, soft_max=100)
    scale_objects_max: bpy.props.IntProperty(name="max", default=20, min=0, soft_max=100)
    scale_x: bpy.props.BoolProperty(name="X", default=True)
    scale_y: bpy.props.BoolProperty(name="Y", default=True)
    scale_z: bpy.props.BoolProperty(name="Z", default=True)
    seed: bpy.props.IntProperty(name="Seed", default=np.random.default_rng().integers(10000, 100000000), min=10000, max=99999999)
    randomize_seed: bpy.props.BoolProperty(name="Randomize seed", default=True)


class DatasetGeneratorCitySettings(bpy.types.PropertyGroup):
    dimension_x: bpy.props.IntProperty(name="X", default=10, min=1, soft_max=100)
    dimension_y: bpy.props.IntProperty(name="Y", default=10, min=1, soft_max=100)
    block_min: bpy.props.IntProperty(name="min", default=2, min=1, soft_max=15)
    block_max: bpy.props.IntProperty(name="max", default=6, min=1, soft_max=15)
    districts: bpy.props.StringProperty(name="Districts", default="residential, commercial, park")
    seed: bpy.props.IntProperty(name="Seed", default=np.random.default_rng().integers(10000, 100000000), min=10000, max=99999999)
    clear_city: bpy.props.BoolProperty(name="Clear existing city", default=True)
    randomize_seed: bpy.props.BoolProperty(name="Randomize seed", default=True)


class DatasetGeneratorScannerSettings(bpy.types.PropertyGroup):
    path_seed: bpy.props.IntProperty(name="Seed", default=np.random.default_rng().integers(10000, 100000000), min=10000, max=99999999)
    randomize_path_seed: bpy.props.BoolProperty(name="Randomize seed", default=True)
    scanner_path: bpy.props.StringProperty(name="Scanner path", default="")
    path_method: bpy.props.EnumProperty(name="Path generation method", items=path_method_items, default='SINGLE')
    path_selection: bpy.props.EnumProperty(name="Path selection method", items=path_selection_items, default='LONGEST')
    path_multiple_amount: bpy.props.IntProperty(name="Amount of paths for multiple", default=10, min=2, soft_max=30)
    path_neighbor_amount: bpy.props.IntProperty(name="Amount of neighbors", default=2, min=1, max=3)

# OPERATORS


class DatasetGeneratorRunScans(bpy.types.Operator):
    bl_idname = "opr.dataset_generator_run_scans"
    bl_label = "Generate Dataset"

    def execute(self, context):
        run_scans(context)

        return {'FINISHED'}


class DatasetGeneratorBuildCity(bpy.types.Operator):
    bl_idname = "opr.dataset_generator_build_city"
    bl_label = "Generate City"

    def execute(self, context):
        build_city(context)

        return {'FINISHED'}


class DatasetGeneratorBuildPath(bpy.types.Operator):
    bl_idname = "opr.dataset_generator_build_path"
    bl_label = "Generate Scan Path"

    def execute(self, context):
        build_path(context)

        return {'FINISHED'}


class DatasetGeneratorClearPath(bpy.types.Operator):
    bl_idname = "opr.dataset_generator_clear_path"
    bl_label = "Clear Scan Path"

    def execute(self, context):
        clear_path(context)

        return {'FINISHED'}


class DatasetGeneratorClearCity(bpy.types.Operator):
    bl_idname = "opr.dataset_generator_clear_city"
    bl_label = "Clear city"

    def execute(self, context):
        clear_city(context)

        return {'FINISHED'}


class DatasetGeneratorResetCity(bpy.types.Operator):
    bl_idname = "opr.dataset_generator_reset_city"
    bl_label = "Reset city"

    def execute(self, context):
        reset_city(context)

        return {'FINISHED'}


class DatasetGeneratorScanSeed(bpy.types.Operator):
    bl_idname = "opr.dataset_generator_scan_seed"
    bl_label = "Randomize Seed"

    def execute(self, context):
        randomize_generator_seed(context)

        return {'FINISHED'}


class DatasetGeneratorPathSeed(bpy.types.Operator):
    bl_idname = "opr.dataset_generator_path_seed"
    bl_label = "Randomize Seed"

    def execute(self, context):
        randomize_path_seed(context)

        return {'FINISHED'}


class DatasetGeneratorCitySeed(bpy.types.Operator):
    bl_idname = "opr.dataset_generator_city_seed"
    bl_label = "Randomize Seed"

    def execute(self, context):
        randomize_city_seed(context)

        return {'FINISHED'}


# PANELS


class DatasetGeneratorBasePanel():
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "render"


class DatasetGeneratorPanel(DatasetGeneratorBasePanel, bpy.types.Panel):
    bl_idname = 'RENDER_PT_DatasetGeneratorPanel'
    bl_label = "Dataset Generator"

    def draw(self, context):
        col = self.layout.column()
        col.operator("opr.dataset_generator_run_scans", text="Generate Data Set")


class DatasetGeneratorCityPanel(DatasetGeneratorBasePanel, bpy.types.Panel):
    bl_idname = 'RENDER_PT_DatasetGeneratorCityPanel'
    bl_parent_id ='RENDER_PT_DatasetGeneratorPanel'
    bl_label = "City Generation"

    def draw(self, context):
        settings = context.scene.city_settings
        layout = self.layout
        col = layout.column()
        row = col.row()
        row.label(text="City generation seed")
        row.prop(settings, "seed")
        row.operator("opr.dataset_generator_city_seed", text="Randomize seed")

        col.separator()
        row = col.row()
        icon = 'DOWNARROW_HLT' if context.scene.city_settings_expanded else 'RIGHTARROW'
        row.prop(context.scene, "city_settings_expanded", icon=icon, icon_only=True)
        row.label(text="City generation settings")
        if context.scene.city_settings_expanded:
            box = col.box()
            subcol = box.column()
            subrow = subcol.row()
            subrow.label(text="City size")
            subrow.prop(settings, "dimension_x", slider=True)
            subrow.prop(settings, "dimension_y", slider=True)
            subrow = subcol.row()
            subrow.label(text="Block size")
            subrow.prop(settings, "block_min", slider=True)
            subrow.prop(settings, "block_max", slider=True)

        col.separator()
        row = col.row()
        row.prop(settings, "clear_city")
        row.prop(settings, "randomize_seed")
        row.operator("opr.dataset_generator_build_city", text="Generate city")
        split = col.split(factor=0.6725)
        split.column()
        splitrow = split.row()
        splitrow.operator("opr.dataset_generator_clear_city", text="Clear city")
        split = col.split(factor=0.6725)
        split.column()
        splitrow = split.row()
        splitrow.operator("opr.dataset_generator_reset_city", text="Reset city")


class DatasetGeneratorScannerPanel(DatasetGeneratorBasePanel, bpy.types.Panel):
    bl_idname = 'RENDER_PT_DatasetGeneratorScanerPanel'
    bl_parent_id = bl_parent_id = 'RENDER_PT_DatasetGeneratorPanel'
    bl_label = 'Scanner Configuration'

    def draw(self, context):
        settings = context.scene.scanner_settings
        layout = self.layout
        col = layout.column()
        row = col.row()
        row.label(text="Path seed")
        row.prop(settings, "path_seed")
        row.operator("opr.dataset_generator_path_seed", text="Randomize seed")
        box = col.box()
        boxcol = box.column()
        boxrow = boxcol.row()
        boxrow.label(text="Path generation method")
        boxrow.prop(settings, "path_method", text="")
        if settings.path_method == 'MULTIPLE':
            boxrow = boxcol.row()
            boxrow.label(text="Number of paths")
            boxrow.prop(settings, "path_multiple_amount", text="")
        elif settings.path_method == 'NEIGHBORS_FROM_NODE':
            boxrow = boxcol.row()
            boxrow.label(text="Neighbors to traverse")
            boxrow.prop(settings, "path_neighbor_amount", text="")
        boxrow = boxcol.row()
        boxrow.label(text="Path Selection")
        boxrow.prop(settings, "path_selection", text="")
        row = col.row()
        row.label(text="Randomize seed")
        row.prop(settings, "randomize_path_seed")
        row.operator("opr.dataset_generator_build_path", text="Generate scanner path")
        split = col.split(factor=0.6725)
        split.column()
        splitrow = split.row()
        splitrow.operator("opr.dataset_generator_clear_path", text="Clear scanner path")


class DatasetGeneratorDatasetPanel(DatasetGeneratorBasePanel, bpy.types.Panel):
    bl_idname = 'RENDER_PT_DatasetGeneratorDatasetPanel'
    bl_parent_id = 'RENDER_PT_DatasetGeneratorPanel'
    bl_label = "Dataset Generation"

    def draw(self, context):
        settings = context.scene.generator_settings
        layout = self.layout
        col = layout.column()
        row = col.row()
        row.label(text="Dataset seed")
        row.prop(settings, "seed")
        row.operator("opr.dataset_generator_scan_seed", text="Randomize seed")

        col.separator()
        row = col.row()
        icon = 'DOWNARROW_HLT' if context.scene.generator_settings_expanded else 'RIGHTARROW'
        row.prop(context.scene, "generator_settings_expanded", icon=icon, icon_only=True)
        row.label(text="Dataset generation settings")

        if context.scene.generator_settings_expanded:
            box = col.box()
            subcol = box.column()
            subcol.prop(settings, "remove_objects_enable")
            if settings.remove_objects_enable:
                subrow = subcol.row()
                subrow.label(text="Objects per scan")
                subrow.prop(settings, "remove_objects_min", slider=True)
                subrow.prop(settings, "remove_objects_max", slider=True)

            subcol.prop(settings, "add_objects_enable")
            if settings.add_objects_enable:
                subrow = subcol.row()
                subrow.label(text="Objects per scan")
                subrow.prop(settings, "add_objects_min", slider=True)
                subrow.prop(settings, "add_objects_max", slider=True)

            subcol.prop(settings, "rotation_enable")
            if settings.rotation_enable:
                subrow = subcol.row()
                subrow.label(text="Rotation (degrees)")
                subrow.prop(settings, "rotation_min", slider=True)
                subrow.prop(settings, "rotation_max", slider=True)
                subrow = subcol.row()
                subrow.label(text="Objects per scan")
                subrow.prop(settings, "rotation_objects_min", slider=True)
                subrow.prop(settings, "rotation_objects_max", slider=True)
                split = subcol.split(factor=0.25)
                splitcol = split.column()
                splitcol.label(text="Rotate along axes")
                splitcol = split.column()
                splitrow = splitcol.row()
                splitrow.prop(settings, "rotation_positive_x")
                splitrow.prop(settings, "rotation_negative_x")
                splitrow.prop(settings, "rotation_positive_y")
                splitrow.prop(settings, "rotation_negative_y")
                splitrow.prop(settings, "rotation_positive_z")
                splitrow.prop(settings, "rotation_negative_z")

            subcol.prop(settings, "translation_enable")
            if settings.translation_enable:
                subrow = subcol.row()
                subrow.label(text="Movement")
                subrow.prop(settings, "translation_min", slider=True)
                subrow.prop(settings, "translation_max", slider=True)
                subrow = subcol.row()
                subrow.label(text="Objects per scan")
                subrow.prop(settings, "translation_objects_min", slider=True)
                subrow.prop(settings, "translation_objects_max", slider=True)
                split = subcol.split(factor=0.25)
                splitcol = split.column()
                splitcol.label(text="Move along axes")
                splitcol = split.column()
                splitrow = splitcol.row()
                splitrow.prop(settings, "translation_positive_x")
                splitrow.prop(settings, "translation_negative_x")
                splitrow.prop(settings, "translation_positive_y")
                splitrow.prop(settings, "translation_negative_y")
                splitrow.prop(settings, "translation_positive_z")
                splitrow.prop(settings, "translation_negative_z")

            subcol.prop(settings, "scale_enable")
            if settings.scale_enable:
                subrow = subcol.row()
                subrow.label(text="Scale")
                subrow.prop(settings, "scale_min", slider=True)
                subrow.prop(settings, "scale_max", slider=True)
                subrow = subcol.row()
                subrow.label(text="Objects per scan")
                subrow.prop(settings, "scale_objects_min", slider=True)
                subrow.prop(settings, "scale_objects_max", slider=True)
                split = subcol.split(factor=0.33)
                splitcol = split.column()
                splitcol.prop(settings, "scale_uniform")
                if not settings.scale_uniform:
                    splitcol = split.column()
                    splitrow = splitcol.row()
                    splitrow.prop(settings, "scale_x")
                    splitrow.prop(settings, "scale_y")
                    splitrow.prop(settings, "scale_z")

        col.separator()
        split = col.split(factor=0.33)
        split.column()
        splitrow = split.row()
        splitrow.prop(settings, "randomize_seed")
        splitrow.operator("opr.dataset_generator_run_scans", text="Run Scans")


# CLASS COLLECTION

CLASSES = [
    DatasetGeneratorPanel,
    DatasetGeneratorCityPanel,
    DatasetGeneratorScannerPanel,
    DatasetGeneratorDatasetPanel,
    DatasetGeneratorRunScans,
    DatasetGeneratorCitySettings,
    DatasetGeneratorScanSettings,
    DatasetGeneratorScannerSettings,
    DatasetGeneratorBuildCity,
    DatasetGeneratorBuildPath,
    DatasetGeneratorClearPath,
    DatasetGeneratorClearCity,
    DatasetGeneratorResetCity,
    DatasetGeneratorScanSeed,
    DatasetGeneratorCitySeed,
    DatasetGeneratorPathSeed,
]

# FUNCTIONS

####
##
## TODO
##
## report/fix bug in pointCloudScanner scanner.py lines 362, 393
## create/get laser scanner
## move/rotate laser scanner to path -> assign path
## pathfinding options: single path, longest of fixed number, longest of all 2-neigbor paths
## add scan settings (# scans, path, name...)
## optional -> laser scanner options in interface (use default for now)
##
####

def randomize_generator_seed(context):
    rng = np.random.default_rng()
    context.scene.generator_settings.seed = rng.integers(10000, 100000000)

def randomize_city_seed(context):
    rng = np.random.default_rng()
    context.scene.city_settings.seed = rng.integers(10000, 100000000)

def randomize_path_seed(context):
    rng = np.random.default_rng()
    context.scene.scanner_settings.path_seed = rng.integers(10000, 100000000)

def reset_city(context):
    city_collection = context.scene.city_collection
    city = bpy.data.collections[city_collection]
    tags = [tag.strip() for tag in context.scene.object_modifier_tags.split(",")]
    for district in city.children_recursive:
        for obj in district.objects:
            if any(tag in obj.name for tag in tags):
                obj.hide_viewport = False
                if (obj.delta_location[:3] != (0.0, 0.0, 0.0) or
                    obj.delta_rotation_euler[:3] != (0.0, 0.0, 0.0) or
                    obj.delta_scale[:3] != (1.0, 1.0, 1.0)):
                    # if any delta transforms are set they switched
                    reset_transforms(obj)
                    transforms_to_deltas(obj)
                for child in obj.children_recursive:
                    child.hide_viewport = False

def clear_city(context):
    city_collection = context.scene.city_collection
    bpy.ops.object.select_all(action='DESELECT')
    try:
        city = bpy.data.collections[city_collection]
        for district in city.children_recursive:
            for obj in district.objects:
                obj.select_set(True)
        bpy.ops.object.delete()
        for district in city.children_recursive:
            bpy.data.collections.remove(district)
        bpy.context.scene.collection.children.unlink(city)
        bpy.data.collections.remove(city)
    except Exception as exception:
        print(exception)

def bound_city_settings(context):
    settings = context.scene.city_settings
    settings.block_max = max(settings.block_min, settings.block_max)

def configure_scenecity_nodes(context):
    settings = context.scene.city_settings
    bpy.data.node_groups["PCGeneratorCity"].nodes["grid_layout_generator"].random_seed = settings.seed
    bpy.data.node_groups["PCGeneratorCity"].nodes["grid_layout_generator"].boxes_min_max_size[0] = settings.block_min
    bpy.data.node_groups["PCGeneratorCity"].nodes["grid_layout_generator"].boxes_min_max_size[1] = settings.block_max
    bpy.data.node_groups["PCGeneratorCity"].nodes["Grid"].grid_size[0] = settings.dimension_x
    bpy.data.node_groups["PCGeneratorCity"].nodes["Grid"].grid_size[1] = settings.dimension_y

def randomize_buildify_levels(collection, rng):
    buildings = []
    for obj in collection.objects:
        if "Modifier" in obj.name:
            buildings.append(obj)
    buildings.sort(key=lambda obj: (obj.matrix_world.translation.x, obj.matrix_world.translation.y))
    for obj in buildings:
        try:
            nodes = obj.modifiers["GeometryNodes"]
            floors = int(rng.integers(3, 11))
            nodes[nodes.node_group.inputs["Max number of floors"].identifier] = floors
            # obj.update_tag() tags object to be updated in viewport
            # otherwise changes in level will not be displayed until
            # further changes to the object are made
            obj.update_tag()
        except Exception as exception:
            print(exception)

def build_city(context):
    city_collection = context.scene.city_collection
    settings = context.scene.city_settings
    if settings.randomize_seed:
        randomize_city_seed(context)
    rng = np.random.default_rng(settings.seed)
    if settings.clear_city:
        clear_city(context)
    bound_city_settings(context)
    configure_scenecity_nodes(context)

    districts = ["road"]
    districts.extend(settings.districts.replace(" ", "").split(","))
    city = bpy.data.collections.new(city_collection)
    prefix = "city_"
    scene_collection = bpy.context.scene.collection
    scene_collection.children.link(city)
    for district in districts:
        city.children.link(bpy.data.collections.new(prefix + district))
    for district in districts:
        bpy.data.node_groups["PCGeneratorCity"].nodes[district + "_portion_instancer"].random_seed = settings.seed
        layer_collection = bpy.context.view_layer.layer_collection.children[city_collection].children[prefix + district]
        node_path = "bpy.data.node_groups[\"PCGeneratorCity\"].nodes[\"" + district + "_instancer\"]"
        bpy.context.view_layer.active_layer_collection = layer_collection
        bpy.ops.node.objects_instancer_node_create(source_node_path=node_path)
    randomize_buildify_levels(bpy.data.collections["city_residential"], rng)

def clear_path(context):
    scanner_path = context.scene.scanner_settings.scanner_path
    if scanner_path != "":
        try:
            bpy.ops.object.select_all(action='DESELECT')
            bpy.data.objects[scanner_path].select_set(True)
            bpy.ops.object.delete()
            context.scene.scanner_settings.scanner_path = ""
        except Exception:
            print(Exception)

def build_adjacency(graph, road_grid):
    for _, data in graph.items():
        node_x, node_y = data["location"]
        neighbours = road_grid[node_x][node_y]["neighbours"]
        for offset_x, offset_y in neighbours:
            x = offset_x
            y = offset_y
            # changed from while road_grid[node_x + x][node_y + y]["neighbours"] is None:
            while road_grid[node_x + x][node_y + y]["type"] == "straight":
                x += offset_x
                y += offset_y
            if road_grid[node_x + x][node_y + y]["type"] == "node":
                data["adjacent_nodes"].append(road_grid[node_x + x][node_y + y]["node"])
            elif road_grid[node_x + x][node_y + y]["type"] == "leaf":
                data["adjacent_leaves"].append(road_grid[node_x + x][node_y + y]["node"])

def build_graph(dimension_x, dimension_y, road_grid):
    graph = {}
    node = 0
    for obj in bpy.data.objects["road"].children:
        obj_x = int(obj.matrix_world.translation.x + float(dimension_x / 2))
        obj_y = int(obj.matrix_world.translation.y + float(dimension_y / 2))
        road_grid[obj_x][obj_y]["neighbours"] = []
        neighbours = 0
        for x, y in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            # since bool operators short circuit, e.g. (x and y) => if x == False then x else y, there should be no error here
            if (0 <= obj_x + x < dimension_x and 0 <= obj_y + y < dimension_y and road_grid[obj_x + x][obj_y + y]):
                neighbours += 1
                road_grid[obj_x][obj_y]["neighbours"].append((x, y))
        if neighbours != 2:
            type = "node" if neighbours > 2 else "leaf"
            graph[str(node)] = {"location": (obj_x, obj_y), "type": type, "adjacent_nodes": [], "adjacent_leaves": []}
            road_grid[obj_x][obj_y]["node"] = str(node)
            road_grid[obj_x][obj_y]["type"] = type
            node += 1
        else:
            # changed from road_grid[obj_x][obj_y]["neighbours"] = None
            road_grid[obj_x][obj_y]["type"] = "straight"
    build_adjacency(graph, road_grid)
    return graph

def build_road_grid(dimension_x, dimension_y, city_grid):
    road_grid = [[None for y in range(dimension_y)] for x in range(dimension_x)]
    for x in range(dimension_x):
        for y in range(dimension_y):
            try:
                city_grid.data[x][y]["road"]
                road_grid[x][y] = {"location": (x,y)}
            except:
                road_grid[x][y] = None
    return road_grid

def generate_paths(graph, context):
    settings = context.scene.scanner_settings
    rng = np.random.default_rng(settings.path_seed)
    paths = []

    def step(node, path):
        no_neighbours = True
        for neighbour in graph[node]["adjacent_nodes"]:
            if neighbour not in path:
                step(neighbour, path + [node])
                no_neighbours = False
        if no_neighbours and graph[node]["adjacent_leaves"]:
            for leaf in graph[node]["adjacent_leaves"]:
                paths.append(path + [node, leaf])
        else:
            paths.append(path + [node])

    def step_limited(node, path, limit):
        neighbors = [neighbor for neighbor in graph[node]["adjacent_nodes"] if neighbor not in path]
        leaves = [leaf for leaf in graph[node]["adjacent_leaves"] if leaf not in path]
        rng.shuffle(neighbors)
        if neighbors:
            for _ in range(limit):
                if neighbors:
                    neighbor = neighbors.pop()
                    step_limited(neighbor, path + [node], limit)
        elif leaves:
            paths.append(path + [node, rng.choice(leaves)])
        else:
            paths.append(path + [node])

    def dfs(node):
        visited = []
        visited.append(node)
        stack = deque()

        def step_dfs(node, path):
            visited.append(node)
            neighbors = [neighbor for neighbor in graph[node]["adjacent_nodes"] if neighbor not in visited]
            rng.shuffle(neighbors)
            leaves = [leaf for leaf in graph[node]["adjacent_leaves"] if leaf not in visited]
            for neighbor in neighbors:
                stack.append((neighbor, path + [node]))
            for leaf in leaves:
                paths.append(path + [node, leaf])
        
        neighbors = graph[node]["adjacent_nodes"]
        leaves = graph[node]["adjacent_leaves"]
        if neighbors:
            stack.append((neighbors[0], [node]))
        elif leaves:
            stack.append((leaves[0], [node]))
        while stack:
            node, path = stack.pop()
            if node not in visited:
                step_dfs(node, path)

    def bfs(node):
        visited = []
        visited.append(node)
        queue = deque()

        def step_bfs(node, path):
            visited.append(node)
            neighbors = [neighbor for neighbor in graph[node]["adjacent_nodes"] if neighbor not in visited]
            rng.shuffle(neighbors)
            leaves = [leaf for leaf in graph[node]["adjacent_leaves"] if leaf not in visited]
            for neighbor in neighbors:
                queue.append((neighbor, path + [node]))
            for leaf in leaves:
                paths.append(path + [node, leaf])

        neighbors = graph[node]["adjacent_nodes"]
        leaves = graph[node]["adjacent_leaves"]
        if neighbors:
            queue.append((neighbors[0], [node]))
        elif leaves:
            queue.append((leaves[0], [node]))
        while queue:
            node, path = queue.popleft()
            if node not in visited:
                step_bfs(node, path)

    nodes = [node for node, _ in graph.items()]
    node = rng.choice(nodes)
    while graph[node]["type"] == "node":
        node = rng.choice(nodes)
    mode = settings.path_method
    if mode == 'MULTIPLE':
        for _ in range(settings.path_multiple_amount):
            step_limited(node, [], 1)
    elif mode == 'NEIGHBORS_FROM_NODE':
        step_limited(node, [], settings.path_neighbor_amount)
    elif mode == 'ALL_FROM_NODE':
        step(node, [])
    elif mode == 'DFS':
        dfs(node)
    elif mode == 'BFS':
        bfs(node)
    else:
        step_limited(node, [], 1)
    return paths

def generate_curve(context, path, graph):
    context.view_layer.active_layer_collection = context.view_layer.layer_collection
    bpy.ops.curve.primitive_bezier_curve_add()
    city_settings = context.scene.city_settings
    curve = bpy.data.objects["BezierCurve"]
    curve.name = "scanner_path"
    bezier_points = curve.data.splines.active.bezier_points
    if len(path) - 2 > 0:
        bezier_points.add(len(path) - 2)
    offset_x, offset_y = graph[path[0]]["location"]
    for point, node in zip(bezier_points, path):
        x, y = graph[node]["location"]
        x -= offset_x
        y -= offset_y
        point.co = Vector((x, y, 0))
        point.handle_right = point.co.copy()
        point.handle_left = point.co.copy()
    curve.location.x = float(offset_x) - (city_settings.dimension_x / 2)
    curve.location.y = float(offset_y) - (city_settings.dimension_y / 2)
    curve.location.z = 0.1
    context.scene.scanner_settings.scanner_path = curve.name

def build_path(context):
    clear_path(context)
    city_grid = bpy.data.node_groups["PCGeneratorCity"].nodes["grid_layout_generator"].get_grid()
    city_settings = context.scene.city_settings
    scanner_settings = context.scene.scanner_settings
    dimension_x = city_settings.dimension_x
    dimension_y = city_settings.dimension_y
    road_grid = build_road_grid(dimension_x, dimension_y, city_grid)
    graph = build_graph(dimension_x, dimension_y, road_grid)
    if scanner_settings.randomize_path_seed:
        randomize_path_seed(context)
    paths = generate_paths(graph, context)
    rng = np.random.default_rng(scanner_settings.path_seed)
    if scanner_settings.path_selection == 'RANDOM':
        rng.shuffle(paths)
        path = paths[0]
    else:
        path = max(paths, key=len)
    # path = rng.choice(paths,) if scanner_settings.path_selection == 'RANDOM' else max(paths, key=len)
    generate_curve(context, path, graph)

def add_objects(settings, hidden_objects, rng):
    amount = rng.integers(settings.add_objects_min, settings.add_objects_max + 1)
    added_objects = []
    rng.shuffle(hidden_objects)
    for _ in range(amount):
        obj = hidden_objects.pop()
        obj.hide_viewport = False
        obj.class_name = "new"
        added_objects.append(obj)
    return added_objects

def remove_objects(settings, objects, rng):
    amount = rng.integers(settings.remove_objects_min, settings.remove_objects_max + 1)
    removed_objects = []
    rng.shuffle(objects)
    for _ in range(amount):
        obj = objects.pop()
        obj.class_name = "removed"
        removed_objects.append(obj)
    return removed_objects

def translate_objects(settings, objects, modified_objects, rng):
    amount = rng.integers(settings.translation_objects_min, settings.translation_objects_max + 1)
    possible_translation = [
        (settings.translation_negative_x, ("x", -1)),
        (settings.translation_positive_x, ("x", 1)),
        (settings.translation_negative_y, ("y", -1)),
        (settings.translation_positive_y, ("y", 1)),
        (settings.translation_negative_z, ("z", -1)),
        (settings.translation_positive_z, ("z", 1)),
        ]
    enabled_translation = [(axis, direction) for (setting, (axis, direction)) in possible_translation if setting]
    for _ in range(amount):
        obj = objects.pop()
        obj.class_name = "moved"
        modified_objects.append(obj)
        axis, direction = rng.choice(enabled_translation)
        value = rng.uniform(settings.translation_min, settings.translation_max) * int(direction)
        setattr(obj.location, axis, getattr(obj.location, axis) + value)

def rotate_objects(context, settings, objects, modified_objects, rng):
    amount = rng.integers(settings.rotation_objects_min, settings.rotation_objects_max + 1)
    possible_rotations = [
        (settings.rotation_negative_x, ("X", -1)),
        (settings.rotation_positive_x, ("X", 1)),
        (settings.rotation_negative_y, ("Y", -1)),
        (settings.rotation_positive_y, ("Y", 1)),
        (settings.rotation_negative_z, ("Z", -1)),
        (settings.rotation_positive_z, ("Z", 1)),
    ]
    enabled_rotations = [(axis, direction) for (setting, (axis, direction)) in possible_rotations if setting]
    building_tags = context.scene.building_modifier_tags
    for _ in range(amount):
        obj = objects.pop()
        obj.class_name = "rotated"
        modified_objects.append(obj)
        axis, direction = rng.choice(enabled_rotations)
        degrees = rng.uniform(settings.rotation_min, settings.rotation_max) * int(direction)
        degrees = degrees * 0.1 if any(tag in obj.name for tag in building_tags) else degrees
        obj.rotation_euler = (obj.rotation_euler.to_matrix() @ Matrix.Rotation(radians(degrees), 3, axis)).to_euler()

def scale_objects(settings, objects, modified_objects, rng):
    amount = rng.integers(settings.scale_objects_min, settings.scale_objects_max + 1)
    for _ in range(amount):
        obj = objects.pop()
        obj.class_name = "scaled"
        modified_objects.append(obj)
        value = rng.uniform(settings.scale_min, settings.scale_max)
        if settings.scale_uniform:
            scale = Vector((value, value, value))
        else:
            x = value if settings.scale_x else 1.0
            y = value if settings.scale_y else 1.0
            z = value if settings.scale_z else 1.0
            scale = Vector((x, y, z))
        obj.scale *= scale

def post_scan_cleanup(objects, hidden_objects, removed_objects, modified_objects, added_objects):
    for _ in range(len(removed_objects)):
        obj = removed_objects.pop()
        obj.class_name = "initial"
        obj.hide_viewport = True
        for child in obj.children_recursive:
            child.hide_viewport = True
        hidden_objects.append(obj)
    for _ in range(len(modified_objects)):
        obj = modified_objects.pop()
        obj.class_name = "initial"
        objects.append(obj)
    for _ in range(len(added_objects)):
        obj = added_objects.pop()
        obj.class_name = "initial"
        objects.append(obj)

# Transfers object transforms to delta transforms and vice versa
# Using blenders built in operators tends to be much slower in comparison
def transforms_to_deltas(obj):
    translation = obj.location.copy()
    rotation = obj.rotation_euler.copy()
    scale = obj.scale.copy()
    obj.location = obj.delta_location
    obj.rotation_euler = obj.delta_rotation_euler
    obj.scale = obj.delta_scale
    obj.delta_location = translation
    obj.delta_rotation_euler = rotation
    obj.delta_scale = scale

def reset_transforms(obj):
    obj.scale[:3] = (1, 1, 1)
    obj.rotation_euler[:3] = (0, 0, 0)
    obj.location[:3] = (0, 0, 0)

def build_object_collection(context):
    city_collection = context.scene.city_collection
    objects = []
    tags = [tag.strip() for tag in context.scene.object_modifier_tags.split(",")]
    city = bpy.data.collections[city_collection]
    for district in city.children_recursive:
        props = []
        for obj in district.objects:
            obj.class_name = "initial"
            obj.hide_viewport = False
            if any(tag in obj.name for tag in tags):
                props.append(obj)
                transforms_to_deltas(obj)
        props.sort(key=lambda obj: (obj.matrix_world.translation.x, obj.matrix_world.translation.y))
        objects.extend(props)
    return objects

def build_hidden_object_collection(settings, objects, rng):
    hidden_objects = []
    amount = settings.scans * settings.add_objects_max
    rng.shuffle(objects)
    for i in range(amount):
        obj = objects.pop()
        hidden_objects.append(obj)
        obj.hide_viewport = True
        for child in obj.children_recursive:
            child.hide_viewport = True
    return hidden_objects

def bound_scan_settings(settings, objects):
    limit = int(len(objects) / (settings.scans * 3))
    settings.rotation_max = max(settings.rotation_min, settings.rotation_max)
    settings.rotation_objects_max = min(settings.rotation_objects_max, limit)
    settings.rotation_objects_min = min(settings.rotation_objects_min, settings.rotation_objects_max)
    settings.translation_max = max(settings.translation_min, settings.translation_max)
    settings.translation_objects_max = min(settings.translation_objects_max, limit)
    settings.translation_objects_min = min(settings.translation_objects_min, settings.translation_objects_max)
    settings.scale_max = max(settings.scale_min, settings.scale_max)
    settings.scale_objects_max = min(settings.scale_objects_max, limit)
    settings.scale_objects_min = min(settings.scale_objects_min, settings.scale_objects_max)
    settings.add_objects_max = min(settings.add_objects_max, limit)
    settings.add_objects_min = min(settings.add_objects_min, settings.add_objects_max)
    settings.remove_objects_max = min(settings.remove_objects_max, limit)
    settings.remove_objects_min = min(settings.remove_objects_min, settings.remove_objects_max)

def create_missing_classes(context):
    classes = [klass.strip() for klass in context.scene.object_classes.split(",")]
    pc_classes = context.scene.pointCloudRenderProperties.classes
    for klass in classes:
        if klass not in pc_classes:
            bpy.ops.pcscanner.add_class()
            classes[-1].name = klass
            classes[-1].class_id = len(pc_classes) - 1

def run_scans(context):
    settings = context.scene.generator_settings
    scanner = bpy.data.scenes["Scene"].pointCloudRenderProperties.laser_scanners[0]
    if settings.randomize_seed:
        randomize_generator_seed(context)
    rng = np.random.default_rng(settings.seed)
    objects = build_object_collection(context)
    create_missing_classes(context)
    bound_scan_settings(settings, objects)
    hidden_objects = build_hidden_object_collection(settings, objects, rng)

    print("-- starting initial scan --")
    scanner.file_path = "//initial_scan.csv"
    # bpy.ops.render.render_point_cloud()
    for _ in range(settings.scans):
        removed_objects = remove_objects(settings, objects, rng)
        modified_objects = []
        scale_objects(settings, objects, modified_objects, rng)
        translate_objects(settings, objects, modified_objects, rng)
        rotate_objects(context, settings, objects, modified_objects, rng)
        added_objects = add_objects(settings, hidden_objects, rng)
        scanner.file_path = "//scan_" + str(_ + 1) + ".csv"
        print("-- starting scan " + str(_ + 1) + " --")
        # bpy.ops.render.render_point_cloud()
        post_scan_cleanup(objects, hidden_objects, removed_objects, modified_objects, added_objects)
    # reset_city()


def register():
    print("registering dataset generator")
    for (prop, value) in PROPS:
        setattr(bpy.types.Scene, prop, value)
    for klass in CLASSES:
        bpy.utils.register_class(klass)
    bpy.types.Scene.generator_settings = bpy.props.PointerProperty(type=DatasetGeneratorScanSettings)
    bpy.types.Scene.city_settings = bpy.props.PointerProperty(type=DatasetGeneratorCitySettings)
    bpy.types.Scene.scanner_settings = bpy.props.PointerProperty(type=DatasetGeneratorScannerSettings)


def unregister():
    print("unregistering dataset generator")
    for (prop, _) in PROPS:
        delattr(bpy.types.Scene, prop)
    for klass in CLASSES:
        bpy.utils.unregister_class(klass)
    del bpy.types.Scene.generator_settings
    del bpy.types.Scene.city_settings
    del bpy.types.Scene.scanner_settings


if __name__ == "__main__":
    register()
