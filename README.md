# ResearchModule
Git Repository for all work related to the Research Module.
A synthetic data study for a volcanic diatreme structure is planned.
First a model is generated with GemPy which then will be used for forward calculations in pygimli.

Gempy-input: contains surface points and orientations of geologic structures for the Gempy model creation. Folder also contains a python script rigenerate these. Note that the script creates more information than needed and is therefore not suitable for such a simple geologic model.

.poly-files: pygimli meshes of layered subsurface model, diatreme model and combined model 

RM_gempy_combined_model.ipynb: Jupyter notebook for creation of gempy model of layered subsurface with a simple volcanic intrusion simulating a diatreme structure. The Gempy model is converted to a pygimli mesh and saved as "mesh_combinedmodel.poly". This mesh can currently not be used due to a shared polygon boundary at the lower model boundary.

RM_pygimli_computations.ipynb: Jupyter Notebook for synthetic data generation using pygimli. It also contains the code used to generate the combined model mesh.
