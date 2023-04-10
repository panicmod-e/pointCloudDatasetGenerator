bl_info = {
    # required
    "name": "Dataset Generator",
    "blender": (3, 0, 0),
    "category": "Object",
    # optional
    "version": (0,5,0),
    "author": "David Schlereth",
    "description": "Addon to automate point cloud dataset creation of a city using the vLiDAR Addon",
}

import bpy
import numpy as np
from mathutils import Vector, Euler, Matrix
from math import radians
from collections import deque
import time

# ---------------------------------------------------------------- #
#                      GLOBAL PROPERTIES
# ---------------------------------------------------------------- #
#
# Section for property and property group definitions.
#
# ---------------------------------------------------------------- #

# Contains enum-items for path creation
# note: neighbors/all from node are very performance intensive and seem to have issues with multi-threading
path_method_items = [
    ('SINGLE', "Single", "Generate single random path"),
    ('MULTIPLE', "Multiple", "Path from multiple randomly generated paths"),
    ('DFS', "DFS traversal", "Path from dfs"),
    ('BFS', "BFS traversal", "Path from bfs"),
    # ('NEIGHBORS_FROM_NODE', "Multiple from node", "Path with fixed number of neighbors traversed"),
    # ('ALL_FROM_NODE', "All from node", "Path from all paths starting in random node"),
]

# enum-items for path seletion
path_selection_items = [
    ('LONGEST', "Longest Path", "Select longest of all generated Paths"),
    ('RANDOM', "Random Path", "Select randomly from all generated Paths"),
]

# general properties that should be directly accessible without being tied to a specific settings group
# for easier registration the properties are defined using a list
PROPS = [
    # city collection is the collection in which the generated city is palced
    ("city_collection", bpy.props.StringProperty(name="City Collection", default="city_generated")),
    ("scan_settings_expanded", bpy.props.BoolProperty(name="Subpanel status", default=True)),
    ("city_settings_expanded", bpy.props.BoolProperty(name="Subpanel status", default=True)),
    # object modifier tags mark any objects that can be modified between scans
    ("object_modifier_tags", bpy.props.StringProperty(name="Object tags", default="prop, building, Prop, Building")),
    ("building_modifier_tags", bpy.props.StringProperty(name="Building tags", default="building")),
    ("buildify_building_modifier_tags", bpy.props.StringProperty(name="Buildify tags", default="buildify_building")),
    ("object_classes", bpy.props.StringProperty(name="Object classes", default="initial, new, removed, moved, rotated, scaled")),
]


# property group for all settings concerning scan execution
class DatasetGeneratorDatasetSettings(bpy.types.PropertyGroup):
    scans: bpy.props.IntProperty(name="Scans", default=4, min=1)
    scans_directory: bpy.props.StringProperty(name="Output directory", default="//", subtype='DIR_PATH')
    scans_prefix: bpy.props.StringProperty(name="Prefix for set", default="pcset")
    scans_new_path: bpy.props.BoolProperty(name="Follow new path each scan", default=False)
    generate_city: bpy.props.BoolProperty(name="Generate new city (uses existing city if disabled)", default=False)
    randomize_city_seed: bpy.props.BoolProperty(name="Randomize city seed", default=True)
    randomize_path_seed: bpy.props.BoolProperty(name="Randomize path seed", default=True)
    randomize_scan_seed: bpy.props.BoolProperty(name="Randomize scan seed", default=True)


# property group for all settings concerning object modification during scans
class DatasetGeneratorScanSettings(bpy.types.PropertyGroup):
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


# property group for all settings for city generation
class DatasetGeneratorCitySettings(bpy.types.PropertyGroup):
    # dimensions x and y denote the size of the city in number of cells
    dimension_x: bpy.props.IntProperty(name="X", default=10, min=1, soft_max=100)
    dimension_y: bpy.props.IntProperty(name="Y", default=10, min=1, soft_max=100)
    block_min: bpy.props.IntProperty(name="min", default=2, min=1, soft_max=15)
    block_max: bpy.props.IntProperty(name="max", default=6, min=1, soft_max=15)
    # list of districts to be generated
    districts: bpy.props.StringProperty(name="Districts", default="residential, commercial, park")
    seed: bpy.props.IntProperty(name="Seed", default=np.random.default_rng().integers(10000, 100000000), min=10000, max=99999999)
    clear_city: bpy.props.BoolProperty(name="Clear existing city", default=True)
    randomize_seed: bpy.props.BoolProperty(name="Randomize seed", default=True)


# property group for all settings for scanner path generation
class DatasetGeneratorScannerSettings(bpy.types.PropertyGroup):
    path_seed: bpy.props.IntProperty(name="Seed", default=np.random.default_rng().integers(10000, 100000000), min=10000, max=99999999)
    randomize_path_seed: bpy.props.BoolProperty(name="Randomize seed", default=True)
    scanner_path: bpy.props.StringProperty(name="Scanner path", default="")
    placeholder_path: bpy.props.StringProperty(name="Placeholder scanner path", default="")
    path_method: bpy.props.EnumProperty(name="Path generation method", items=path_method_items, default='BFS')
    path_selection: bpy.props.EnumProperty(name="Path selection method", items=path_selection_items, default='LONGEST')
    path_multiple_amount: bpy.props.IntProperty(name="Amount of paths for multiple", default=10, min=2, soft_max=30)
    path_neighbor_amount: bpy.props.IntProperty(name="Amount of neighbors", default=2, min=1, max=3)


# ---------------------------------------------------------------- #
#                            OPERATORS
# ---------------------------------------------------------------- #
#
# Section for operator class definitions
#
# ---------------------------------------------------------------- #



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


# ---------------------------------------------------------------- #
#                             PANELS
# ---------------------------------------------------------------- #
#
# Section for UI panel class definitions
#
# ---------------------------------------------------------------- #

# base panel contains some base definitions shared by all panels
class DatasetGeneratorBasePanel():
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "render"


# parent panel containing all other panels
# panel is placed in the "Render" context menu
class DatasetGeneratorPanel(DatasetGeneratorBasePanel, bpy.types.Panel):
    bl_idname = 'RENDER_PT_DatasetGeneratorPanel'
    bl_label = "Dataset Generator"

    def draw(self, context):
        col = self.layout.column()
        col.label(text="Pointcloud Dataset Generator")


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


class DatasetGeneratorSettingsPanel(DatasetGeneratorBasePanel, bpy.types.Panel):
    bl_idname = 'RENDER_PT_DatasetGeneratorSettingsPanel'
    bl_parent_id = 'RENDER_PT_DatasetGeneratorPanel'
    bl_label = "Dataset Scan Settings"

    def draw(self, context):
        settings = context.scene.scan_settings
        layout = self.layout
        col = layout.column()
        row = col.row()
        row.label(text="Dataset seed")
        row.prop(settings, "seed")
        row.operator("opr.dataset_generator_scan_seed", text="Randomize seed")
        col.separator()
        row = col.row()
        icon = 'DOWNARROW_HLT' if context.scene.scan_settings_expanded else 'RIGHTARROW'
        row.prop(context.scene, "scan_settings_expanded", icon=icon, icon_only=True)
        row.label(text="Settings")
        if context.scene.scan_settings_expanded:
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


class DatasetGeneratorDatasetPanel(DatasetGeneratorBasePanel, bpy.types.Panel):
    bl_idname = 'RENDER_PT_DatasetGeneratorDatasetPanel'
    bl_parent_id = 'RENDER_PT_DatasetGeneratorPanel'
    bl_label = "Create Dataset"

    def draw(self, context):
        dataset_settings = context.scene.dataset_settings
        scan_settings = context.scene.scan_settings
        layout = self.layout
        col = layout.column()
        box = col.box()
        boxcol = box.column()
        boxrow = boxcol.row()
        boxrow.label(text="Number of scans")
        boxrow.prop(dataset_settings, "scans", text="")
        boxrow = boxcol.row()
        boxrow.label(text="Scan set prefix")
        boxrow.prop(dataset_settings, "scans_prefix", text="")
        boxrow = boxcol.row()
        boxrow.label(text="Output directory")
        boxrow.prop(dataset_settings, "scans_directory", text="")
        boxrow = boxcol.row()
        boxrow.prop(dataset_settings, "randomize_scan_seed")
        boxrow.prop(dataset_settings, "scans_new_path")
        col.separator()
        box = col.box()
        boxcol = box.column()
        boxcol.prop(dataset_settings, "generate_city")
        if dataset_settings.generate_city:
            boxcol.separator()
            boxrow = boxcol.row()
            boxrow.prop(dataset_settings, "randomize_city_seed")
            boxrow.prop(dataset_settings, "randomize_path_seed")
        row = col.row()
        row.label(text="")
        row.label(text="")
        row.operator("opr.dataset_generator_run_scans", text="Run Scans")
        layout.separator()


# ---------------------------------------------------------------- #
#                       CLASS COLLECTION
# ---------------------------------------------------------------- #
#
# All defined classes should be listed here for easy registration
# later on. Excluded are property collections and properties
# defined in a separate collection.
#
# ---------------------------------------------------------------- #


CLASSES = [
    DatasetGeneratorPanel,
    DatasetGeneratorCityPanel,
    DatasetGeneratorScannerPanel,
    DatasetGeneratorSettingsPanel,
    DatasetGeneratorDatasetPanel,
    DatasetGeneratorRunScans,
    DatasetGeneratorCitySettings,
    DatasetGeneratorDatasetSettings,
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

# ---------------------------------------------------------------- #
#                           FUNCTIONS
# ---------------------------------------------------------------- #
#
# Section for function/method definitions. Functions are grouped
# according to task, i.e. city-/path-/scan-generation
#
# ---------------------------------------------------------------- #

def randomize_generator_seed(context):
    rng = np.random.default_rng()
    context.scene.scan_settings.seed = rng.integers(10000, 100000000)

def randomize_city_seed(context):
    rng = np.random.default_rng()
    context.scene.city_settings.seed = rng.integers(10000, 100000000)

def randomize_path_seed(context):
    rng = np.random.default_rng()
    context.scene.scanner_settings.path_seed = rng.integers(10000, 100000000)

# ------------------------------------- #
#            City Generation
# ------------------------------------- #

def reset_city(context):
    # resets transformations made to city objects
    # restores original transforms by pulling them from delta transforms
    city_collection = context.scene.city_collection
    try:
        city = bpy.data.collections[city_collection]
    except Exception:
        return
    tags = [tag.strip() for tag in context.scene.object_modifier_tags.split(",")]
    for district in city.children_recursive:
        for obj in district.objects:
            if any(tag in obj.name for tag in tags):
                obj.hide_viewport = False
                if (obj.delta_location[:3] != (0.0, 0.0, 0.0) or
                    obj.delta_rotation_euler[:3] != (0.0, 0.0, 0.0) or
                    obj.delta_scale[:3] != (1.0, 1.0, 1.0)):
                    # if any delta transforms are set they are switched
                    reset_transforms(obj)
                    transforms_to_deltas(obj)
                for child in obj.children_recursive:
                    child.hide_viewport = False

def clear_city(context):
    # removes all objects and collections created during city generation
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
    except Exception:
        return

def bound_city_settings(context):
    settings = context.scene.city_settings
    settings.block_max = max(settings.block_min, settings.block_max)

def configure_scenecity_nodes(context):
    # applies any relevant settings set in ui to relevant scenecity nodes
    settings = context.scene.city_settings
    bpy.data.node_groups["PCGeneratorCity"].nodes["grid_layout_generator"].boxes_values = settings.districts
    bpy.data.node_groups["PCGeneratorCity"].nodes["grid_layout_generator"].random_seed = settings.seed
    bpy.data.node_groups["PCGeneratorCity"].nodes["grid_layout_generator"].boxes_min_max_size[0] = settings.block_min
    bpy.data.node_groups["PCGeneratorCity"].nodes["grid_layout_generator"].boxes_min_max_size[1] = settings.block_max
    bpy.data.node_groups["PCGeneratorCity"].nodes["Grid"].grid_size[0] = settings.dimension_x
    bpy.data.node_groups["PCGeneratorCity"].nodes["Grid"].grid_size[1] = settings.dimension_y

def randomize_buildify_levels(context, city, rng):
    # randomizes the floors of buildify buildings, otherwise all buildify buildings would be the same height
    buildings = []
    tags = context.scene.buildify_building_modifier_tags
    # this section collects all tagged buildings in a list which is then sorted by object location
    # this step is necessary to make the city generation (specifically assigning the building floors) deterministic
    # since SceneCity itself does not seem to name or place the buildings in a deterministic order based on the seed used
    for district in city.children:
        for obj in district.objects:
            if any(tag in obj.name for tag in tags):
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
            None

def build_city(context):
    start = time.time()
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
        # create new collection for each district and link it to city collection
        city.children.link(bpy.data.collections.new(prefix + district))
    for district in districts:
        # for each district the corresponding scenecity instancer node is called separately
        bpy.data.node_groups["PCGeneratorCity"].nodes[district + "_portion_instancer"].random_seed = settings.seed
        # active layer collection determines the collection in which the instancer places the new objects
        layer_collection = bpy.context.view_layer.layer_collection.children[city_collection].children[prefix + district]
        node_path = "bpy.data.node_groups[\"PCGeneratorCity\"].nodes[\"" + district + "_instancer\"]"
        bpy.context.view_layer.active_layer_collection = layer_collection
        bpy.ops.node.objects_instancer_node_create(source_node_path=node_path)
    randomize_buildify_levels(context, bpy.data.collections[city_collection], rng)
    end = time.time()
    print("City generated in " + str(end - start))

# ------------------------------------- #
#         Scan Path Generation
# ------------------------------------- #

def generate_placeholder_path(context):
    # Due to how the vLiDAR scanner is implemented simply unassigning the scanner path
    # is not supported. To cleanly delete the created scanner path this function
    # creates a placeholder path, unless one already exists
    settings = context.scene.scanner_settings
    try:
        bpy.data.objects[settings.placeholder_path]
    except Exception:
        settings.placeholder_path = ""
    if not settings.placeholder_path:
        path_name = "pcdg_placeholder_path"
        path_data = bpy.data.curves.new(path_name, type='CURVE')
        path_data.dimensions = '3D'
        path_spline = path_data.splines.new(type='BEZIER')
        path_spline.bezier_points[0].handle_left_type = 'VECTOR'
        path_spline.bezier_points[0].handle_right_type = 'VECTOR'
        placeholder_path = bpy.data.objects.new(path_name, path_data)
        context.scene.collection.objects.link(placeholder_path)
        settings.placeholder_path = path_name

def clear_path(context):
    scanner_path = context.scene.scanner_settings.scanner_path
    if scanner_path != "":
        try:
            # assigns the placeholder path to the laser scanner and removes the old path
            current_path_object = bpy.data.objects[scanner_path]
            generate_placeholder_path(context)
            placeholder_path = context.scene.scanner_settings.placeholder_path
            placeholder_path_object = bpy.data.objects[placeholder_path]
            laser_scanners = bpy.context.scene.pointCloudRenderProperties.laser_scanners
            for scanner in laser_scanners:
                if scanner.path.path_object and scanner.path.path_object == current_path_object:
                    scanner.path.path_object = placeholder_path_object
            bpy.data.objects.remove(current_path_object)
            context.scene.scanner_settings.scanner_path = ""
        except Exception:
            return

def build_adjacency(graph, road_grid):
    # Builds adjacency lists for each graph node.
    # road_grid saves directions where neighboring nodes are found
    # these directions are traversed until a node is encountered
    for _, data in graph.items():
        node_x, node_y = data["location"]
        neighbours = road_grid[node_x][node_y]["neighbours"]
        for offset_x, offset_y in neighbours:
            x = offset_x
            y = offset_y
            while road_grid[node_x + x][node_y + y]["type"] == "straight":
                x += offset_x
                y += offset_y
            if road_grid[node_x + x][node_y + y]["type"] == "node":
                data["adjacent_nodes"].append(road_grid[node_x + x][node_y + y]["node"])
            elif road_grid[node_x + x][node_y + y]["type"] == "leaf":
                data["adjacent_leaves"].append(road_grid[node_x + x][node_y + y]["node"])

def build_graph(dimension_x, dimension_y, road_grid):
    # builds the initial graph from the road_grid
    graph = {}
    node = 0
    for obj in bpy.data.objects["road"].children:
        # checks each street object itself instead of traversing the entire grid
        # index in grid correspond to the objects world location offset by the cities dimensions
        obj_x = int(obj.matrix_world.translation.x + float(dimension_x / 2))
        obj_y = int(obj.matrix_world.translation.y + float(dimension_y / 2))
        road_grid[obj_x][obj_y]["neighbours"] = []
        neighbours = 0
        for x, y in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            # checks adjacent cells if they contain a road portion, if so the direction is marked as containing a neighbor
            # since bool operators short circuit, e.g. (x and y) => if x == False then x else y, there should be no error here
            if (0 <= obj_x + x < dimension_x and 0 <= obj_y + y < dimension_y and road_grid[obj_x + x][obj_y + y]):
                neighbours += 1
                road_grid[obj_x][obj_y]["neighbours"].append((x, y))
        if neighbours != 2:
            # if a road portion only has exactly two neighbors it is a straight road portion that is of no interest
            # otherwise it is either a crossroad or a road portion at the edge of the city
            # the road portion is then either classified as a node or a leaf respectively
            type = "node" if neighbours > 2 else "leaf"
            graph[str(node)] = {"location": (obj_x, obj_y), "type": type, "adjacent_nodes": [], "adjacent_leaves": []}
            road_grid[obj_x][obj_y]["node"] = str(node)
            road_grid[obj_x][obj_y]["type"] = type
            node += 1
        else:
            road_grid[obj_x][obj_y]["type"] = "straight"
    build_adjacency(graph, road_grid)
    return graph

def build_road_grid(dimension_x, dimension_y, city_grid):
    # traverses scenecity grid and creates grid containing all road portions without districts
    road_grid = [[None for y in range(dimension_y)] for x in range(dimension_x)]
    for x in range(dimension_x):
        for y in range(dimension_y):
            try:
                # since scenecity denotes roads and districts differently in its grid
                # the try-except block makes use of an exception to detect road portions
                city_grid.data[x][y]["road"]
                road_grid[x][y] = {"location": (x,y)}
            except:
                road_grid[x][y] = None
    return road_grid

def generate_paths(graph, context):
    # generates path(s) from road graph using selected method
    settings = context.scene.scanner_settings
    rng = np.random.default_rng(settings.path_seed)
    paths = []

    def step(node, path):
        # step function used to find all paths starting in specified node
        # currently not advised to be used as it is exponential in terms of time complexity
        # and seems to suffer some issues with multi-threading
        # currently still included for reference
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
        # limited step function which only traverses a set number of neighboring nodes
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
        # depth first search through graph
        # can generate relatively long winding paths through the city
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
        # breadth first search through the graph
        # can generate relatively straight paths from one edge of the city to another
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
        # makes sure the starting node is a leaf, i.e. a road portion at the edge of the city
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
    # generates a new bezier curve for the newly generated path through the city
    context.view_layer.active_layer_collection = context.view_layer.layer_collection
    bpy.ops.curve.primitive_bezier_curve_add()
    city_settings = context.scene.city_settings
    curve = bpy.data.objects["BezierCurve"]
    curve.name = "scanner_path"
    bezier_points = curve.data.splines.active.bezier_points
    if len(path) - 2 > 0:
        # adds missing points to the curve so they match the number of points the generated path has
        bezier_points.add(len(path) - 2)
    offset_x, offset_y = graph[path[0]]["location"]
    for point, node in zip(bezier_points, path):
        x, y = graph[node]["location"]
        x -= offset_x
        y -= offset_y
        point.co = Vector((x, y, 0))
        # left and right handle type is set to 'VECTOR'
        # handle type is very important for the vLiDAR scanner to work correclty
        point.handle_right_type = 'VECTOR'
        point.handle_left_type = 'VECTOR'
    curve.location.x = float(offset_x) - (city_settings.dimension_x / 2)
    curve.location.y = float(offset_y) - (city_settings.dimension_y / 2)
    curve.location.z = 0.1
    context.scene.scanner_settings.scanner_path = curve.name

def assign_path_to_scanner(context, scanner):
    # assigns newly generated bezier curve to vLiDAR scanner
    scanner_path = context.scene.scanner_settings.scanner_path
    new_path_object = bpy.data.objects[scanner_path]
    if scanner.path.path_object:
        # if scanner has an assigned path it is saved as the placeholder path
        context.scene.scanner_settings.placeholder_path = scanner.path.path_object.name
    else:
        generate_placeholder_path(context)
    scanner.path.path_object = new_path_object
    # vLiDAR scanner path length is updated and the scan duration is set accordingly
    bpy.ops.pcscanner.update_path_length()
    scanner.scan_duration = int(scanner.path.length * 2)
    if scanner.scanner_type == "mobile_mapping_scanner":
        scanner.mobile_mapping_velocity = 0.5
    elif scanner.scanner_type == "artificial_scanner":
        scanner.artificial_velocity = 0.5
    scanner_object = scanner.camera
    # for the scans to work correctly the object representing the scanner in the scene has to be rotated correctly
    # the first two points of the scanner path determine the direction the object has to point
    # which determines the axis along which the object is then rotated accordingly
    curve_point_0 = new_path_object.data.splines.active.bezier_points[0]
    curve_point_1 = new_path_object.data.splines.active.bezier_points[1]
    if curve_point_1.co.x - curve_point_0.co.x != 0:
        difference = curve_point_1.co.x - curve_point_0.co.x
        axis = 'Y'
    elif curve_point_0.co.y - curve_point_1.co.y != 0:
        difference = curve_point_0.co.y - curve_point_1.co.y
        axis = 'X'
    degrees = 75 * (difference / abs(difference))
    # rotation is achieved using matrix rotation and multiplication provided by Blenders mathutils library
    # the result is then converted to Euler and assigned to the scanner object as the new Euler rotation
    scanner_object.rotation_euler = (Euler((0.0, 0.0, 0.0), 'XYZ').to_matrix() @ Matrix.Rotation(radians(degrees), 3, axis)).to_euler()

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
    generate_curve(context, path, graph)
    selected_scanner = bpy.context.scene.pointCloudRenderProperties.selected_scanner
    scanner = bpy.context.scene.pointCloudRenderProperties.laser_scanners[selected_scanner]
    assign_path_to_scanner(context, scanner)

# ------------------------------------- #
#       Scan Generation/Automation
# ------------------------------------- #

def add_objects(settings, hidden_objects, rng):
    # adds new objects to scene by revealing a number of hidden objects in the viewport
    # added objects are classified as "new", removed from hidden_objects list and
    # appended to added_objects list which is the returned
    amount = rng.integers(settings.add_objects_min, settings.add_objects_max + 1)
    added_objects = []
    # random access is achieved by shuffling the list and popping the last element(s)
    rng.shuffle(hidden_objects)
    for _ in range(amount):
        obj = hidden_objects.pop()
        obj.hide_viewport = False
        for child in obj.children_recursive:
            child.hide_viewport = False
        obj.class_name = "new"
        added_objects.append(obj)
    return added_objects

def remove_objects(settings, objects, rng):
    # objects are classified as removed and moved to separate list, which is then returned
    amount = rng.integers(settings.remove_objects_min, settings.remove_objects_max + 1)
    removed_objects = []
    for _ in range(amount):
        obj = objects.pop()
        obj.class_name = "removed"
        removed_objects.append(obj)
    return removed_objects

def translate_objects(settings, objects, modified_objects, rng):
    # moves random number (amount) of objects in a single random direction
    amount = rng.integers(settings.translation_objects_min, settings.translation_objects_max + 1)
    # list includes all axis as well as direction (positive and negative) along which an object can be moved
    possible_translation = [
        (settings.translation_negative_x, ("x", -1)),
        (settings.translation_positive_x, ("x", 1)),
        (settings.translation_negative_y, ("y", -1)),
        (settings.translation_positive_y, ("y", 1)),
        (settings.translation_negative_z, ("z", -1)),
        (settings.translation_positive_z, ("z", 1)),
        ]
    # builds a list of all directions that are enabled in the settings along which an object can be moved
    enabled_translation = [(axis, direction) for (setting, (axis, direction)) in possible_translation if setting]
    for _ in range(amount):
        obj = objects.pop()
        obj.class_name = "moved"
        modified_objects.append(obj)
        axis, direction = rng.choice(enabled_translation)
        value = rng.uniform(settings.translation_min, settings.translation_max) * int(direction)
        # moving the object by making use of getattr and setattr for easier access to a specific axis
        setattr(obj.location, axis, getattr(obj.location, axis) + value)

def rotate_objects(context, settings, objects, modified_objects, rng):
    # rotates a number (amount) of objects along a single random axis/direction
    amount = rng.integers(settings.rotation_objects_min, settings.rotation_objects_max + 1)
    possible_rotations = [
        (settings.rotation_negative_x, ("X", -1)),
        (settings.rotation_positive_x, ("X", 1)),
        (settings.rotation_negative_y, ("Y", -1)),
        (settings.rotation_positive_y, ("Y", 1)),
        (settings.rotation_negative_z, ("Z", -1)),
        (settings.rotation_positive_z, ("Z", 1)),
    ]
    # list of all directions that are enabled in the settings
    enabled_rotations = [(axis, direction) for (setting, (axis, direction)) in possible_rotations if setting]
    building_tags = context.scene.building_modifier_tags
    for _ in range(amount):
        obj = objects.pop()
        obj.class_name = "rotated"
        modified_objects.append(obj)
        axis, direction = rng.choice(enabled_rotations)
        degrees = rng.uniform(settings.rotation_min, settings.rotation_max) * int(direction)
        # if object is a building the rotation is less pronounced but not entirely ignored
        degrees = degrees * 0.1 if any(tag in obj.name for tag in building_tags) else degrees
        # object rotation is achieved by making use of matrix rotation and multiplication provided
        # by Blenders mathutils library, the resulting Euler is then assigned to the object
        obj.rotation_euler = (obj.rotation_euler.to_matrix() @ Matrix.Rotation(radians(degrees), 3, axis)).to_euler()

def scale_objects(settings, objects, modified_objects, rng):
    # scales a number (amount) of objects either uniformly or along the axes enabled in the settings
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
    # resets object classifications, hides removed objects in viewport
    # objects from modified and added objects lists are moved to objects list
    # objects from removed objects list are moved to hidden objects list
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

def transforms_to_deltas(obj):
    # Transfers object transforms to delta transforms and vice versa
    # Using blenders built in operators tends to be much slower in comparison
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
    # builds object list of all buildings and props that can receive modifications between scans
    # modifiable objects have a corresponding tag in their object name
    city_collection = context.scene.city_collection
    objects = []
    tags = [tag.strip() for tag in context.scene.object_modifier_tags.split(",")]
    city = bpy.data.collections[city_collection]
    for district in city.children_recursive:
        # props list is generated separately for each district
        # this is mainly done because sorting multiple shorter lists is faster
        # than sorting the longer combined list of all districts
        props = []
        for obj in district.objects:
            obj.class_name = "initial"
            obj.hide_viewport = False
            if any(tag in obj.name for tag in tags):
                props.append(obj)
                transforms_to_deltas(obj)
        # list of props is sorted by their location in the scene
        # this is done in order to achieve the same order each time and is required to make scans
        # of the same city repeatable/deterministic as the order can vary otherwise
        props.sort(key=lambda obj: (obj.matrix_world.translation.x, obj.matrix_world.translation.y))
        objects.extend(props)
    return objects

def build_hidden_object_collection(scan_settings, dataset_settings, objects, rng):
    # initial list of objects hidden from the city, these can later be added/revealed
    hidden_objects = []
    amount = dataset_settings.scans * scan_settings.add_objects_max
    rng.shuffle(objects)
    for i in range(amount):
        obj = objects.pop()
        hidden_objects.append(obj)
        obj.hide_viewport = True
        for child in obj.children_recursive:
            child.hide_viewport = True
    return hidden_objects

def bound_scan_settings(scan_settings, dataset_settings, objects):
    limit = int(len(objects) / (dataset_settings.scans * 3))
    scan_settings.rotation_max = max(scan_settings.rotation_min, scan_settings.rotation_max)
    scan_settings.rotation_objects_max = min(scan_settings.rotation_objects_max, limit)
    scan_settings.rotation_objects_min = min(scan_settings.rotation_objects_min, scan_settings.rotation_objects_max)
    scan_settings.translation_max = max(scan_settings.translation_min, scan_settings.translation_max)
    scan_settings.translation_objects_max = min(scan_settings.translation_objects_max, limit)
    scan_settings.translation_objects_min = min(scan_settings.translation_objects_min, scan_settings.translation_objects_max)
    scan_settings.scale_max = max(scan_settings.scale_min, scan_settings.scale_max)
    scan_settings.scale_objects_max = min(scan_settings.scale_objects_max, limit)
    scan_settings.scale_objects_min = min(scan_settings.scale_objects_min, scan_settings.scale_objects_max)
    scan_settings.add_objects_max = min(scan_settings.add_objects_max, limit)
    scan_settings.add_objects_min = min(scan_settings.add_objects_min, scan_settings.add_objects_max)
    scan_settings.remove_objects_max = min(scan_settings.remove_objects_max, limit)
    scan_settings.remove_objects_min = min(scan_settings.remove_objects_min, scan_settings.remove_objects_max)

def create_missing_classes(context):
    # creates vLiDAR classes required for classification of objects
    required_classes = [klass.strip() for klass in context.scene.object_classes.split(",")]
    pc_class_names = [klass.name for klass in context.scene.pointCloudRenderProperties.classes]
    pc_classes = context.scene.pointCloudRenderProperties.classes
    for klass in required_classes:
        if klass not in pc_class_names:
            bpy.ops.pcscanner.add_class()
            pc_classes[-1].name = klass
            pc_classes[-1].class_id = len(pc_classes) - 1

def run_scans(context):
    reset_city(context)
    scan_settings = context.scene.scan_settings
    dataset_settings = context.scene.dataset_settings
    city_settings = context.scene.city_settings
    scanner_settings = context.scene.scanner_settings
    try:
        selected_scanner = context.scene.pointCloudRenderProperties.selected_scanner
        scanner = context.scene.pointCloudRenderProperties.laser_scanners[selected_scanner]
    except Exception:
        print("Could not access selected laser scanner")
        print(Exception)
        return

    if dataset_settings.randomize_scan_seed:
        randomize_generator_seed(context)

    if dataset_settings.generate_city:
        city_settings.randomize_seed = True if dataset_settings.randomize_city_seed else False
        scanner_settings.randomize_path_seed = True if dataset_settings.randomize_path_seed else False
        build_city(context)
        build_path(context)

    rng = np.random.default_rng(scan_settings.seed)
    objects = build_object_collection(context)
    create_missing_classes(context)
    bound_scan_settings(scan_settings, dataset_settings, objects)
    hidden_objects = build_hidden_object_collection(scan_settings, dataset_settings, objects, rng)

    file_name = dataset_settings.scans_directory + dataset_settings.scans_prefix + "_scan_"
    print("-- starting initial scan --")
    scanner.file_path = file_name + "01.csv"
    bpy.ops.render.render_point_cloud()
    for scans in range(dataset_settings.scans - 1):
        if dataset_settings.scans_new_path:
            scanner_settings.path_seed = rng.integers(10000, 100000000)
            build_path(context)
        # random access is achieved by shuffling the list and popping the last element(s)
        rng.shuffle(objects)
        removed_objects = remove_objects(scan_settings, objects, rng)
        modified_objects = []
        scale_objects(scan_settings, objects, modified_objects, rng)
        translate_objects(scan_settings, objects, modified_objects, rng)
        rotate_objects(context, scan_settings, objects, modified_objects, rng)
        added_objects = add_objects(scan_settings, hidden_objects, rng)
        file_suffix = "0" + str(scans + 2) + ".csv" if scans < 8 else str(scans + 2) + ".csv"
        scanner.file_path = file_name + file_suffix
        print("-- starting scan " + str(scans + 2) + " --")
        bpy.ops.render.render_point_cloud()
        post_scan_cleanup(objects, hidden_objects, removed_objects, modified_objects, added_objects)

# ------------------------------------- #
#          Plugin Registration
# ------------------------------------- #


def register():
    # registers all custom classes and properties when adding the plugin to Blender
    print("registering dataset generator")
    for (prop, value) in PROPS:
        setattr(bpy.types.Scene, prop, value)
    for klass in CLASSES:
        bpy.utils.register_class(klass)
    # registration of property groups
    bpy.types.Scene.scan_settings = bpy.props.PointerProperty(type=DatasetGeneratorScanSettings)
    bpy.types.Scene.city_settings = bpy.props.PointerProperty(type=DatasetGeneratorCitySettings)
    bpy.types.Scene.scanner_settings = bpy.props.PointerProperty(type=DatasetGeneratorScannerSettings)
    bpy.types.Scene.dataset_settings = bpy.props.PointerProperty(type=DatasetGeneratorDatasetSettings)


def unregister():
    # unregisters all custom classes and properties
    print("unregistering dataset generator")
    for (prop, _) in PROPS:
        delattr(bpy.types.Scene, prop)
    for klass in CLASSES:
        bpy.utils.unregister_class(klass)
    del bpy.types.Scene.scan_settings
    del bpy.types.Scene.city_settings
    del bpy.types.Scene.scanner_settings
    del bpy.types.Scene.dataset_settings


if __name__ == "__main__":
    register()
