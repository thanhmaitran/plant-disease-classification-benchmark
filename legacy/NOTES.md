# Legacy coursework results: do not quote as clean

`legacy_*.csv` were derived from `task1_full_results.json` of the original run.

Inconsistencies found:

| Item | Original code | JSON results | Report |
|---|---|---|---|
| Images | 2,152 in the dataset archive | 4,304 (3012/646/646) | 4,304 |
| Healthy support in test | would be 23 | 46 (= 15% of 304) | 46 |
| Image size | 96 | 128 | 96 |
| Epochs | max 30 | 50 ran, best at 44 | max 30 and "epoch 44" |

The healthy class has 152 images in the public dataset but 304 in the results, so
every image was duplicated, which leaks test images into training. The CNN
99.38% figure is therefore not trustworthy until the CNN is re-run on the
de-duplicated split (`python scripts/run_experiments.py --models svm cnn`).

## Errors in the submitted report (verified against the JSON history)

- At epoch 44 the CNN train accuracy is 98.44% (not 99.44%, which is epoch 50).
- Healthy training images: about 213 (70% of 304), not 107 (70% of 152).
- Training time claims ("< 5 min", "10-30 min") are not backed by any measurement.

Full list: `docs/report_errata_task1.md`.

## Open question (needs the original `data_task1.zip`)

Count the images in the original archive and check for repeated class folders.
4,304 unique images -> legacy numbers stand (fix config text only).
Duplicated 2,152 images -> legacy numbers are inflated, use the clean results.
