# Point Cloud Dataset Generator Blender Plugin

This Blender plugin was developed during the seminary project "Introduction to Image and Video Processing Techniques" at the Hasso Plattner Institute. The intended usage of the plugin is to make use of the [vLiDAR scanner plugin](https://github.com/justus-hildebrand/pointCloudRender) by J. Hildebrand to create synthetic point cloud datasets of a city at different timestamps.

The plugin makes use of the [SceneCity Blender plugin](https://www.cgchan.com/store/scenecity) for procedural city generation, and provides automation for city generation, scan path generation and point cloud scan generation, along with modifications in between scans of the same set to simulate changes in the scene at different timestamps.

For added diversity within the city procedural buildings are placed. These are created using the [Buildify geometry nodes library](https://paveloliva.gumroad.com/l/buildify).

## Setup

The basic plugin can be installed like any plugin for Blender. First download or compress the repository as a `.zip` file. Open Blender and navigate to `Edit > Preferences > Add-ons > Install`, select the `.zip` archive and hit install. The plugin should be listed as `Dataset Generator`, make sure it is enabled.
### Requirements

The plugin requires SceneCity 1.9 as well as the required node trees and assets contained in the provided .blend file(s) to work correctly. The provided file or a copy can be used directly or the requirements can be linked or appended to a new file.

The plugin requires the vLiDAR scanner plugin and is currently built to work with the following branch/commit https://github.com/justus-hildebrand/pointCloudRender/tree/d4fedb3cd1f84c0f066d96a96c33559c574b48b7, and requires the corresponding performance patch for Blender. Please refer to the instructions contained in the corresponding readme for further detail.

### Appending/Linking Assets

Assets can be linked/appended to a file opened in Blender by navigating to `File > Link` or `File > Append`, selecting the .blend file that contains the assets, navigating to the assets within the blend file and hitting "Append" or "Link".

The following assets within the provided .blend file(s) should be appended:
```
Collections\Buildify Buildings
Collections\SceneCity Buildings
Collections\SceneCity Streets
Collections\Parks
NodeTree\PCGeneratorCity
```
Any prerequisits should be appended automatically when these assets are added.

## Usage

The plugin can be found in the `Render Properties` panel within Blender.

The plugin provides separate controls for the creation of a city, scanner path, as well as datasets. To avoid any errors during usage make sure to first create a new vLiDAR scanner using the vLiDAR scanner plugin and set the scanner type to `Mobile Mapping Scanner` for optimal results. Any settings such as `Samples per Second` and `Angular Velocity` should be configured here as well.

### City Generation

A new city can be generated by clicking the `Generate city` button in the plugin panel. Be aware that this can take some time as a large amount of objects need to be placed in the city. The size of the city and city blocks can be configured in the UI. The newly generated city will be placed in a new collection called `city_generated` with child collections containing each district within the city.

The checkboxes `Clear existing city` and `Randomize seed` control the behaviour when creating a new city. It is advised to always clear the existing city while creating a new one. The plugin is currently not flexible enough to handle several separate cities and might deliver unexpected results otherwise. To re-create the same city or use a custom seed for city creation the `Randomize seed` checkbox should be unchecked.

### Scanner Path Generation

To create a path for the vLiDAR scanner through the city, make sure that a generated city as well as a vLiDAR scanner are already created and present in the scene.

A new path can then be placed by clicking the `Generate scanner path` button. The available algorithms for path generation include single random, multiple random, breadth first search, and depth first search. one of the generated paths can then be selected either randomly, or by choosing the longest of the generated paths.

### Dataset Generation

To create a dataset make sure that a generated city, a vLiDAR scanner and a scanner path are already created and present in the scene. A Dataset can then be generated by choosing the number of scans to be performed and clicking the `Run Scans` button.

The number of scans includes the initial scan of the city, i.e. entering `1` as the number of scans would only result in the initial point cloud scan of the generated city.

The `Dataset Scan Settings` panel provides several customization options regarding the modifications to objects between any two scans of the same set.

To generate a complete dataset including city and path generation the checkbox `Generate new city` can be checked, in which case a new city and scanner path will be created using the settings provided above and the selected number of scans will be executed.

## Limitations

Please be aware there are currently several limitations to the function of this plugin. Due to the nature of the plugins dependencies any interactions are heavily dependant on the specific implementations and naming and as such are not guaranteed to work with other versions and without the required NodeTree and assets present.

City generation with SceneCity is achieved by calling the corresponding operators of the plugin with a path to the specific nodes used in the NodeTree, referenced specifically by name. The plugin technically supports additional districts, however any additional districts require their own NodeTree pipeline ending with a `Buildings Instancer` and `Objects Instancer` node with the specific name of `<district>_portion_instancer` and `<district>_instancer` respectively, with `<district>` being the name of the district. This new district can be added to the list of districts found in the city settings property group and must be identical (case sensitive) to the district name in the instancer nodes.

New meshes can easily be added to existing settings by creating additional `Objects getter` nodes and linking them up to the districts `Building collection` as seen in the existing `PCGeneratorCity` NodeTree.