# Data

This directory intentionally holds only pointers to the public datasets that
back the experiments. No image data is redistributed here.

## PlantVillage

- 38 folder classes (14 crop species, one held-out non-leaf background class
  excluded from training)
- Approximately 54 000 studio-collected leaf images
- Original publication: Mohanty S.P., Hughes D.P., Salathé M. (2016)
  *Using Deep Learning for Image-Based Plant Disease Detection*,
  Frontiers in Plant Science 7:1419. DOI 10.3389/fpls.2016.01419
- Expected local layout matches `code/dataset.py`:
  `PV_root/Plant_leave_diseases_dataset_without_augmentation/<class>/*.jpg`

## PlantDoc

- 27 folder classes; ~2 600 field-captured images
- Original publication: Singh D., Jain N., Jain P., Kayal P., Kumawat S.,
  Batra N. (2020) *PlantDoc: A Dataset for Visual Plant Disease Detection*,
  CoDS-COMAD 2020. DOI 10.1145/3371158.3371196
- Available at: https://github.com/pratikkayal/PlantDoc-Dataset
- Expected local layout: `PlantDoc-Dataset/{train,test}/<class>/*.jpg`

## Reproducibility mapping

The name-based mapping between PlantDoc and PlantVillage labels used for the
cross-dataset evaluation is defined in `code/dataset.py`
(`PLANTDOC_TO_PV` dictionary). Classes without a semantic equivalent in
PlantVillage are dropped from the evaluation.
