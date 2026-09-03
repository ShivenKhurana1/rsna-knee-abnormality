"""Tensor/metadata contract shared by every Family A component.

Mirrors the already-deployed, already-tested V13 six-slot layout (SLOTS, GROUP,
IMG, CROP_MM, SLOT_PRIOR_TABLE) so a raw-slot-tensor cache built by reusing V13's
own pick_slots/order_slices/read_slot cells (see build_pool_notebook.py) is
directly consumable here without a second, independently-written and unvalidated
DICOM decoder. Family A's model and supervision are new; the pixel pipeline is
reused, not reinvented.
"""

UID = 'StudyInstanceUID'
TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
           'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']

# name, plane, fluid-sensitive (None = don't care), fatsat
SLOTS = [
    ('SAG_FLUID_FS', 'Sagittal', True, True),
    ('COR_FLUID_FS', 'Coronal', True, True),
    ('AX_FLUID_FS', 'Axial', True, True),
    ('SAG_FLUID_NOFS', 'Sagittal', True, False),
    ('COR_T1', 'Coronal', False, False),
    ('SAG_T1', 'Sagittal', False, False),
]
N_SLOT = len(SLOTS)
GROUP = 3        # adjacent slices per window, fed as pseudo-RGB channels
IMG = 336        # per-slot tile side, pixels
CROP_MM = 130.0

SLOT_PRIOR_TABLE = {
    'ACL': (0, 3, 5), 'MCL': (1, 4), 'Medial Meniscus': (0, 1, 3, 4),
    'Lateral Meniscus': (0, 1, 3, 4), 'Medial OA': (1, 4, 5), 'Lateral OA': (1, 4, 5),
    'PF OA': (0, 2, 5), 'Effusion': (0, 2), 'Synovitis': (0, 2), "Baker's": (0,),
    'Contusion': (0, 1, 2), 'Fracture': (0, 1, 2, 4, 5),
}
SLOT_PRIOR_STRENGTH = 0.55
