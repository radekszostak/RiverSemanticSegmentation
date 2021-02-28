# RiverSemanticSegmentation

## Informacje ogólne

Repozytorium zawiera program pozwalający na trening modelów opartych na konwolucyjnych sieciach neuronowych służacych do segmentacji obszarów rzecznych na zdjęciach satelitarnych skomponowanych z pasm widzialnych RGB.

## Rezultaty
Autorska implementacja modelu vgg_unet uzyskała wynik IoU=0.90174. Poniżej przedstwaiono przykłądowe segmentacje (Kolumny odpowiednio: wejście, wzór, wyjście modelu).

![results.png](https://i.postimg.cc/Hk06sPNr/results.png)

## Użyte narzędzia
- PyTorch - framework ML
- OpenCV - biblioteka do przetwarzania obrazów
- NumPy - biblioteka do operacji na macierzach
- neptune - narzędzie logujące

## Uruchomienie

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
