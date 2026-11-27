"""
YCB type table — single source of truth for the scene-generation modules in
inference/sim/ and for any post-hoc analysis tool.

11 types, semantic IDs 1..11. Every instance of a type gets the same id, even
when the same type is spawned more than once in a scene.

Standing rotation:
  _Q_Y_UP    : local +Y → world +Z   (long axis Y, body at +Y)
  _Q_NEG_Y_UP: local -Y → world +Z   (long axis Y, cap at -Y; mustard / bleach)
  _Q_X_UP    : local +X → world +Z   (long axis X, the box-shaped types)

half_local : USD LOCAL frame (mesh AABB half-extents).
half_world : WORLD frame half-extents once the object is standing, at yaw=0.
             Used for placement bounds, initial z, and the expected voxel
             height extent.
"""
from typing import Dict, Tuple

# ── Type table ──────────────────────────────────────────────────────────────
# (semantic_id 1-based, usd_name, mass[kg], standing_rot, half_local, half_world,
#  shape_label, is_tall)
#
# half_world = (x, y, z) world-frame half-extents once standing, at yaw=0.
#              z component = height/2 → init z = SHELF_Y + half_world[2].
#              voxel y range (standing) = 2 * half_world[2] / VOXEL_SIZE.

_Q_Y_UP     = (0.7071067811865476,  0.7071067811865476, 0.0, 0.0)
_Q_NEG_Y_UP = (0.7071067811865476, -0.7071067811865476, 0.0, 0.0)
_Q_X_UP     = (0.7071067811865476,  0.0, -0.7071067811865476, 0.0)


YCB_TYPES = [
    # (id, usd_name,                mass, rot,         half_local,                half_world,                shape,    is_tall)
    ( 1, "002_master_chef_can",     0.41, _Q_Y_UP,     (0.0512, 0.0701, 0.0512), (0.0512, 0.0512, 0.0701), "cyl",    True ),
    ( 2, "003_cracker_box",         0.41, _Q_Y_UP,     (0.0820, 0.1067, 0.0359), (0.0820, 0.0359, 0.1067), "box",    True ),
    ( 3, "004_sugar_box",           0.51, _Q_Y_UP,     (0.0464, 0.0882, 0.0226), (0.0464, 0.0226, 0.0882), "box",    True ),
    ( 4, "005_tomato_soup_can",     0.35, _Q_Y_UP,     (0.0339, 0.0510, 0.0339), (0.0339, 0.0339, 0.0510), "cyl",    False),
    ( 5, "006_mustard_bottle",      0.60, _Q_NEG_Y_UP, (0.0480, 0.0957, 0.0291), (0.0480, 0.0291, 0.0957), "ellipse", True ),
    ( 6, "007_tuna_fish_can",       0.30, _Q_Y_UP,     (0.0428, 0.0168, 0.0428), (0.0428, 0.0428, 0.0168), "cyl",    False),
    ( 7, "008_pudding_box",         0.30, _Q_X_UP,     (0.0569, 0.0449, 0.0193), (0.0193, 0.0449, 0.0569), "box",    False),
    ( 8, "009_gelatin_box",         0.30, _Q_X_UP,     (0.0446, 0.0366, 0.0150), (0.0150, 0.0366, 0.0446), "box",    False),
    ( 9, "010_potted_meat_can",     0.37, _Q_Y_UP,     (0.0508, 0.0418, 0.0288), (0.0508, 0.0288, 0.0418), "box",    False),
    (10, "021_bleach_cleanser",     1.13, _Q_NEG_Y_UP, (0.0512, 0.1253, 0.0339), (0.0512, 0.0339, 0.1253), "ellipse", True ),
    (11, "061_foam_brick",          0.30, _Q_X_UP,     (0.0390, 0.0256, 0.0263), (0.0263, 0.0256, 0.0390), "box",    False),
]


N_TYPES = len(YCB_TYPES)                          # 11
TYPE_BY_ID:    Dict[int, dict]   = {}
TYPE_BY_USD:   Dict[str, dict]   = {}
USD_BY_ID:     Dict[int, str]    = {}
ID_BY_USD:     Dict[str, int]    = {}
TALL_IDS:      list              = []
SMALL_IDS:     list              = []
HALF_LOCAL:    Dict[int, Tuple]  = {}
HALF_WORLD:    Dict[int, Tuple]  = {}
ROT_BY_ID:     Dict[int, Tuple]  = {}
SHAPE_BY_ID:   Dict[int, str]    = {}
MASS_BY_ID:    Dict[int, float]  = {}

for tid, usd, mass, rot, hl, hw, shape, is_tall in YCB_TYPES:
    rec = {"id": tid, "usd": usd, "mass": mass, "rot": rot,
           "half_local": hl, "half_world": hw, "shape": shape, "is_tall": is_tall}
    TYPE_BY_ID[tid]  = rec
    TYPE_BY_USD[usd] = rec
    USD_BY_ID[tid]   = usd
    ID_BY_USD[usd]   = tid
    HALF_LOCAL[tid]  = hl
    HALF_WORLD[tid]  = hw
    ROT_BY_ID[tid]   = rot
    SHAPE_BY_ID[tid] = shape
    MASS_BY_ID[tid]  = mass
    (TALL_IDS if is_tall else SMALL_IDS).append(tid)


# ── Slot table ──────────────────────────────────────────────────────────────
# 22 slots = 11 types × 2 instances. Matches the rigid_objects spawned in the scene cfg.
# slot_idx → type_id : slot 0,1 = type 1, slot 2,3 = type 2, …
SLOTS_PER_TYPE = 2
N_SLOTS = N_TYPES * SLOTS_PER_TYPE         # 22
# Slots are named obj_0..obj_21 (no zero padding) so that a child scene cfg can
# override any same-named object defined by its parent cfg and simply add the
# rest. BeliefMapSceneCfgNoObj defines no objects, so all 22 slots are new.
SLOT_NAMES = [f"obj_{i}" for i in range(N_SLOTS)]
SLOT_TO_TYPE_ID = {i: (i // SLOTS_PER_TYPE) + 1 for i in range(N_SLOTS)}


# ── Standing voxel-y range (height extent in voxel units) ────────────────────
# Expected per-type voxel height extent, for post-hoc checks such as
# detecting objects that have tipped over.
VOXEL_SIZE = 0.005
TYPE_STANDING_VY = {tid: int(round(2.0 * HALF_WORLD[tid][2] / VOXEL_SIZE))
                    for tid in TYPE_BY_ID}


def slot_init_pos(slot_idx: int):
    """Default spawn location for each of the 22 slots: far outside the shelf,
    i.e. effectively switched off. Every episode, randomize_object_positions
    moves the N_ACTIVE slots it selected onto the shelf and parks the rest
    back here, where they are far enough away to be invisible to every camera.
    """
    return (-50.0 - slot_idx * 0.5, 0.0, -50.0)
