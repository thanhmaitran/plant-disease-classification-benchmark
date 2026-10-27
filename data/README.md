# Data

Included: `data/task1/`, the PlantVillage potato subset (public, ~42 MB, 2,152 images).

```
data/task1/Plant disease classification/
  Potato___Early_blight/   1000 images
  Potato___Late_blight/    1000 images
  Potato___healthy/         152 images
```

Verified with `python scripts/inspect_dataset.py`: 2,152 readable images, 0 exact duplicates.

Source: PlantVillage (Hughes & Salathe, 2015, arXiv:1511.08060), distributed publicly
(e.g. Kaggle / GitHub mirrors) under an open licence. Check the licence of the mirror you
use before redistributing, and keep this attribution. If you prefer a lighter repository,
remove `data/task1/` and re-download it, the code finds the class folders automatically.
