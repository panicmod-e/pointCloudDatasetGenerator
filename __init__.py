bl_info = {
    # required
    "name": "Dataset Generator",
    "blender": (3, 0, 0),
    "category": "Object",
    # optional
    "version": (0,0,1),
    "author": "David Schlereth",
    "description": "Addon to automate point cloud dataset creation of a city using the vLiDAR Addon",
}

import bpy
import random
import numpy as np
import mathutils

# GLOBAL VARIABLES

PROPS = [
    ("city_collection", bpy.props.StringProperty(name="City Collection", default=""), 'OUTLINER_COLLECTION'),
    ("settings_subpanel", bpy.props.BoolProperty(name="Subpanel status", default=False), 'RIGHTARROW')
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
    rotation_min: bpy.props.FloatProperty(name="min", default=1.0, min=0.0, soft_max=180.0, step=10)
    rotation_max: bpy.props.FloatProperty(name="max", default=10.0, min=0.0, soft_max=180.0, step=10)
    rotation_positive_x: bpy.props.BoolProperty(name="+X", default=False)
    rotation_negative_x: bpy.props.BoolProperty(name="-X", default=False)
    rotation_positive_y: bpy.props.BoolProperty(name="+Y", default=False)
    rotation_negative_y: bpy.props.BoolProperty(name="-Y", default=False)
    rotation_positive_z: bpy.props.BoolProperty(name="+Z", default=True)
    rotation_negative_z: bpy.props.BoolProperty(name="-Z", default=True)
    translation_enable: bpy.props.BoolProperty(name="Random translation", default=True)
    translation_min: bpy.props.FloatProperty(name="min", default=0.5, min=0.0, soft_max=10.0, step=10)
    translation_max: bpy.props.FloatProperty(name="max", default=1.5, min=0.0, soft_max=10.0, step=10)
    translation_negative_x: bpy.props.BoolProperty(name="-X", default=True)
    translation_positive_x: bpy.props.BoolProperty(name="+X", default=True)
    translation_positive_y: bpy.props.BoolProperty(name="+Y", default=True)
    translation_negative_y: bpy.props.BoolProperty(name="-Y", default=True)
    translation_positive_z: bpy.props.BoolProperty(name="+Z", default=False)
    translation_negative_z: bpy.props.BoolProperty(name="-Z", default=True)
    scale_enable: bpy.props.BoolProperty(name="Random scaling", default=True)
    scale_uniform: bpy.props.BoolProperty(name="Uniform scaling", default=True)
    scale_min: bpy.props.FloatProperty(name="min", default=0.9, min=0.0, soft_min=0.5, soft_max=1.5, step=10)
    scale_max: bpy.props.FloatProperty(name="max", default=1.1, min=0.0, soft_min=0.5, soft_max=1.5, step=10)
    scale_x: bpy.props.BoolProperty(name="X", default=True)
    scale_y: bpy.props.BoolProperty(name="Y", default=True)
    scale_z: bpy.props.BoolProperty(name="Z", default=True)
    seed: bpy.props.IntProperty(name="Seed", default=np.random.default_rng().integers(10000, 100000000), min=10000, max=99999999)

class DatasetGeneratorCitySettings(bpy.types.PropertyGroup):
    dimension_x: bpy.props.IntProperty(name="X", default=10, min=1, soft_max=100)
    dimension_y: bpy.props.IntProperty(name="Y", default=10, min=1, soft_max=100)
    block_min: bpy.props.IntProperty(name="min", default=2, min=1, soft_max=15)
    block_max: bpy.props.IntProperty(name="max", default=6, min=1, soft_max=15)
    districts: bpy.props.StringProperty(name="Districts", default="residential, commercial, park")
    seed: bpy.props.IntProperty(name="Seed", default=np.random.default_rng().integers(10000, 100000000), min=10000, max=99999999)
    delete_city: bpy.props.BoolProperty(name="Clear existing city", default=True)
    randomize_seed: bpy.props.BoolProperty(name="Randomize seed", default=True)


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

class DatasetGeneratorScanSeed(bpy.types.Operator):
    bl_idname = "opr.dataset_generator_scan_seed"
    bl_label = "Randomize Seed"

    def execute(self, context):
        randomize_generator_seed(context)

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

        layout.separator()
        col.label(text="City generation settings")

        box = col.box()
        subcol = box.column()
        subrow = subcol.row()
        subrow.label(text="City size")
        subrow.prop(settings, "dimension_x")
        subrow.prop(settings, "dimension_y")
        subrow = subcol.row()
        subrow.label(text="Block size")
        subrow.prop(settings, "block_min")
        subrow.prop(settings, "block_max")

        row = col.row()
        row.prop(settings, "delete_city")
        row.prop(settings, "randomize_seed")
        row.operator("opr.dataset_generator_build_city", text="Generate City")

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
        col.label(text="Dataset generation settings")

        box = col.box()
        subcol = box.column()
        subcol.prop(settings, "remove_objects_enable")
        if settings.remove_objects_enable:
            subrow = subcol.row()
            subrow.label(text="Objects per scan")
            subrow.prop(settings, "remove_objects_min")
            subrow.prop(settings, "remove_objects_max")

        subcol.prop(settings, "add_objects_enable")
        if settings.add_objects_enable:
            subrow = subcol.row()
            subrow.label(text="Objects per scan")
            subrow.prop(settings, "add_objects_min")
            subrow.prop(settings, "add_objects_max")

        subcol.prop(settings, "rotation_enable")
        if settings.rotation_enable:
            subrow = subcol.row()
            subrow.label(text="Rotation (degrees)")
            subrow.prop(settings, "rotation_min")
            subrow.prop(settings, "rotation_max")
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
            subrow.prop(settings, "translation_min")
            subrow.prop(settings, "translation_max")
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
            subrow.prop(settings, "scale_min")
            subrow.prop(settings, "scale_max")
            split = subcol.split(factor=0.33)
            splitcol = split.column()
            splitcol.prop(settings, "scale_uniform")
            if not settings.scale_uniform:
                splitcol = split.column()
                splitrow = splitcol.row()
                splitrow.prop(settings, "scale_x")
                splitrow.prop(settings, "scale_y")
                splitrow.prop(settings, "scale_z")


class DatasetGeneratorScanSettingsPanel(DatasetGeneratorBasePanel, bpy.types.Panel):
    bl_idname = 'RENDER_PT_DatasetGeneratorScanSettingsPanel'
    bl_parent_id = 'RENDER_PT_DatasetGeneratorDatasetPanel'
    bl_label = "Scan Settings"

    def draw(self, context):
        settings = context.scene.generator_settings

        layout = self.layout
        col = layout.column()
        row = col.row()

        col.prop(settings, "remove_objects_enable")
        if settings.remove_objects_enable:
            box = col.box()
            subcol = box.column()
            subrow = subcol.row()
            subrow.label(text="Objects per scan")
            subrow.prop(settings, "remove_objects_min")
            subrow.prop(settings, "remove_objects_max")

        layout.separator()
        col.prop(settings, "add_objects_enable")
        if settings.add_objects_enable:
            box = col.box()
            subcol = box.column()
            subrow = subcol.row()
            subrow.label(text="Objects per scan")
            subrow.prop(settings, "add_objects_min")
            subrow.prop(settings, "add_objects_max")

        layout.separator()
        col.prop(settings, "rotation_enable")
        if settings.rotation_enable:
            box = col.box()
            subcol = box.column()
            subrow = subcol.row()
            subrow.label(text="Rotation")
            subrow.prop(settings, "rotation_min")
            subrow.prop(settings, "rotation_max")
            split = subcol.split(factor=0.25)
            subcol = split.column()
            subcol.label(text="Rotate along axes")
            subcol = split.column()
            subrow = subcol.row()
            subrow.prop(settings, "rotation_positive_x")
            subrow.prop(settings, "rotation_negative_x")
            subrow.prop(settings, "rotation_positive_y")
            subrow.prop(settings, "rotation_negative_y")
            subrow.prop(settings, "rotation_positive_z")
            subrow.prop(settings, "rotation_negative_z")

        layout.separator()
        col.prop(settings, "translation_enable")
        if settings.translation_enable:
            box = col.box()
            subcol = box.column()
            subrow = subcol.row()
            subrow.label(text="Movement")
            subrow.prop(settings, "translation_min")
            subrow.prop(settings, "translation_max")
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

        layout.separator()
        col.prop(settings, "scale_enable")
        if settings.scale_enable:
            box = col.box()
            subcol = box.column()
            subrow = subcol.row()
            subrow.label(text="Scale")
            subrow.prop(settings, "scale_min")
            subrow.prop(settings, "scale_max")
            subcol.prop(settings, "scale_uniform")
            if not settings.scale_uniform:
                split = subcol.split()
                splitcol = split.column()
                splitcol.label(text="Scale along axes")
                splitcol = split.column()
                splitrow = splitcol.row()
                splitrow.prop(settings, "scale_x")
                splitrow.prop(settings, "scale_y")
                splitrow.prop(settings, "scale_z")


# CLASSES

CLASSES = [
    DatasetGeneratorPanel,
    DatasetGeneratorCityPanel,
    DatasetGeneratorDatasetPanel,
    DatasetGeneratorScanSettingsPanel,
    DatasetGeneratorRunScans,
    DatasetGeneratorCitySettings,
    DatasetGeneratorScanSettings,
    DatasetGeneratorBuildCity,
    DatasetGeneratorScanSeed,
    DatasetGeneratorCitySeed,
]


# FUNCTIONS

####
##
## TODO
##
## make object lists global props for easier access instead of passing lots of refs each time?
## maybe still pass settings, definition could get too cunbersome otherwise
##
## move pre-scan cleanup to post-scan cleanup -> less need for ref passing/declaring lists outside loop
##
## expose block size in ui 
##
####

def run_scans(context):
    objects = build_object_collection(context)
    hidden_objects = build_hidden_object_collection(context, objects)
    scanner = bpy.data.scenes["Scene"].pointCloudRenderProperties.laser_scanners[0]
    settings = context.scene.generator_settings

    print("-- starting initial scan --")
    scanner.file_path = "//initial_scan.csv"
    bpy.ops.render.render_point_cloud()

    removed_objects = []
    for _ in range(settings.scans):
        pre_scan_cleanup(objects, removed_objects, hidden_objects)

        ## modify visible objects here, i.e. rotate/translate/scale etc
        remove_objects(settings, objects, removed_objects)
        add_objects(settings, objects, hidden_objects)
        scanner.file_path = "//scan_" + str(_ + 1) + ".csv"

        print("-- starting scan " + str(_ + 1) + " --")
        # bpy.ops.render.render_point_cloud()

    post_scan_cleanup(objects, removed_objects, hidden_objects)


def build_city(context, city_collection="city_generated"):
    settings = context.scene.city_settings
    if settings.randomize_seed:
        randomize_city_seed(context)
    rng = np.random.default_rng(settings.seed)
    if settings.delete_city:
        delete_city()

    districts = ["road"]
    districts.extend(settings.districts.replace(" ", "").split(","))

    bpy.data.node_groups["PCGeneratorCity"].nodes["Non-overlapping boxes layout"].random_seed = settings.seed
    bpy.data.node_groups["PCGeneratorCity"].nodes["Grid"].grid_size[0] = settings.dimension_x
    bpy.data.node_groups["PCGeneratorCity"].nodes["Grid"].grid_size[1] = settings.dimension_y

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


def delete_city(city_collection="city_generated"):
    try:
        city = bpy.data.collections[city_collection]
        for district in city.children_recursive:
            for obj in district.objects:
                bpy.data.objects.remove(obj)
            bpy.data.collections.remove(district)
        bpy.data.collections.remove(city)
    except Exception:
        print(Exception)


def randomize_buildify_levels(collection, rng):
    for obj in collection.objects:
        if "Modifier" in obj.name:
            try:
                nodes = obj.modifiers["GeometryNodes"]
                nodes[nodes.node_group.inputs["Max number of floors"].identifier] = int(rng.integers(3, 11))
                obj.update_tag()
            except Exception:
                print(Exception)


def add_objects(settings, objects, hidden_objects):
    amount = random.randrange(settings.add_objects_min, settings.add_objects_max+1)
    added_objects = []
    for _ in range(amount):
        obj = random.choice(hidden_objects)
        obj.hide_viewport = False
        obj.class_name = "new"
        hidden_objects.remove(obj)
        objects.append(obj)
        added_objects.append(obj)


def remove_objects(settings, objects, removed_objects):
    amount = random.randrange(settings.remove_objects_min, settings.remove_objects_max+1)
    for _ in range(amount):
        obj = random.choice(objects)
        obj.class_name = "removed"
        objects.remove(obj)
        removed_objects.append(obj)


def pre_scan_cleanup(objects, removed_objects, hidden_objects):
    for obj in objects:
        obj.class_name = "initial"
    for obj in removed_objects:
        obj.class_name = "initial"
        obj.hide_viewport = True
        removed_objects.remove(obj)
        hidden_objects.append(obj)
    print(removed_objects)


def post_scan_cleanup(objects, removed_objects, hidden_objects):
    for obj in objects + removed_objects + hidden_objects:
        obj.class_name = "initial"
        obj.hide_viewport = False


def build_object_collection(context):
    objects = []
    tag = "Modifier"
    city = bpy.data.collections["city_residential"]
    for obj in city.objects:
        obj.class_name = "initial"
        obj.hide_viewport = False
        if tag in obj.name:
            objects.append(obj)
    return objects


def build_hidden_object_collection(context, objects):
    hidden_objects = []
    settings = context.scene.generator_settings
    amount = settings.scans * settings.add_objects_max
    for i in range(amount):
        obj = random.choice(objects)
        objects.remove(obj)
        hidden_objects.append(obj)
        obj.hide_viewport = True
    return hidden_objects


def randomize_generator_seed(context):
    rng = np.random.default_rng()
    context.scene.generator_settings.seed = rng.integers(10000, 100000000)


def randomize_city_seed(context):
    rng = np.random.default_rng()
    context.scene.city_settings.seed = rng.integers(10000, 100000000)


def clear_classification():
    return 0


def register():
    print("registering dataset generator")
    for (prop, value, _) in PROPS:
        setattr(bpy.types.Scene, prop, value)
    for klass in CLASSES:
        bpy.utils.register_class(klass)
    bpy.types.Scene.generator_settings = bpy.props.PointerProperty(type=DatasetGeneratorScanSettings)
    bpy.types.Scene.city_settings = bpy.props.PointerProperty(type=DatasetGeneratorCitySettings)


def unregister():
    print("unregistering dataset generator")
    for (prop, _, _) in PROPS:
        delattr(bpy.types.Scene, prop)
    for klass in CLASSES:
        bpy.utils.unregister_class(klass)
    del bpy.types.Scene.generator_settings


if __name__ == "__main__":
    register()
