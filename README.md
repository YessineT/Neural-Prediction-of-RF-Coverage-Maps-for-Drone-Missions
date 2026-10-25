# Student datasets

Four synthetic scenes on a 100 x 100 m area. All of them share the same set of ground sources.

| Scene | Ground | Building |
|---|---|---|
| `flat` | flat | no |
| `flat_building` | flat | 30 x 20 x 15 m block in the centre |
| `irregular` | smooth hills, about +/- 3 m | no |
| `irregular_building` | smooth hills, about +/- 3 m | 30 x 20 x 15 m block in the centre |

Sources come from a Poisson process with 0.15 sources per m² (at least 1000 are kept). They sit 1 to 10 m above the ground, at least 5 m from the edge and 2 m from the building. Source ids are the same in every scene, so results can be compared scene by scene.

## Setup (Windows)

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Sionna RT runs on the CPU if there is no GPU, but on Windows it needs LLVM installed for that (see the Sionna installation guide).

## Usage

```
python generate_scenes.py
python check_scenes.py
jupyter notebook explore_scenes.ipynb
```

`generate_scenes.py` writes the scenes into `scenes/`. Each scene folder has the Sionna scene (`<name>.xml` and `meshes/`), the sources (`sources.npz`), a height map (`height_map.npz`) and a top view (`topview.png`). The parameters used are saved in `scenes/config.json`; to change them, edit the top of `generate_scenes.py`.

`check_scenes.py` loads each scene in Sionna and traces a few sources, just to see that everything works.

`explore_scenes.ipynb` shows the scenes, the source statistics and a 3D view.

## Notes

The ground is ITU `medium_dry_ground`, which is only defined between 1 and 10 GHz. The building is ITU `concrete`.

On the irregular terrain neighbouring triangles have slightly different planes, and Sionna can return the same ground reflection more than once. Keep this in mind when counting paths or summing power.
