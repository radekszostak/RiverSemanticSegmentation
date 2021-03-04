# RiverSemanticSegmentation

## Informacje ogólne

Repozytorium zawiera program pozwalający na trening modeli opartych na konwolucyjnych sieciach neuronowych służacych do segmentacji obszarów rzecznych na zdjęciach satelitarnych skomponowanych z pasm widzialnych RGB.

## Rezultaty
Autorska implementacja modelu vgg_unet uzyskała wynik IoU=0.90174. Poniżej przedstwaiono przykładowe dane (kolumny odpowiednio: wejście, wzorowe wyjście, wyjście modelu).

![results.png](https://i.postimg.cc/Hk06sPNr/results.png)

## Użyte narzędzia
- PyTorch - framework ML
- OpenCV - biblioteka do przetwarzania obrazów
- NumPy - biblioteka do operacji na macierzach
- neptune - narzędzie logujące

## Dataset

Dataset do pobrania z oddzielnego repozytorium: https://github.com/shocik/sentinel-river-segmentation-dataset

## Uruchomienie
Uruchomienie kodu na własnym komputerze wymaga wykonania następujących kroków przygotowujących:

1. Wprowadzenie danych neptune w pliku [config.cfg](config.cfg).
2. Modyfikacja ścieżki do folderu roboczego w pliku [train.ipynb](train.ipynb):
    ```Python
    #set workdir
    os.chdir("/content/drive/MyDrive/RiverSemanticSegmentation/")
    ```
3. Modyfikacja ścieżki do zbioru danych w pliku [train.ipynb](train.ipynb):
    ```Python
    #dataset configuration
    dataset_dir = os.path.normpath("/content/drive/MyDrive/DEM-waterlevel/dataset")
    ```
