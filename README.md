# RiverSemanticSegmentation

## Uruchomienie:

1. Dataset do pobrania: https://drive.google.com/drive/folders/1eZxNSwuRe8KnTQbtwCtDWmsWnufUrwws?usp=sharing
2. Trenowanie odbywa się w pliku train.ipynb. Należy w nim zmodyfikować:
	- ścieżkę do workdir:
	```python
	#set workdir
	os.chdir("/content/drive/MyDrive/RiverSemanticSegmentation/")
	```
	- ścieżkę do datasetu
	```python
	#dataset configuration
	dataset_dir = os.path.normpath("/content/drive/MyDrive/SemanticSegmentationV2/dataset/")
	```