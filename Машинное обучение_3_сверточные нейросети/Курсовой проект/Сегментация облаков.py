"""
Описание задания

Постройте сегментацию изображений облаков типа Fish, используя сети Unet, PSPNet
или FPN. В качестве базовых сетей можно использовать ResNet, MobileNet, DenseNet
или любые другие подходящие. Можно использовать обученные модели сетей
(входные размеры 384х256).

Постройте ансамбль предсказаний, выбирая среднее значение из нескольких.
Выгрузите результаты предсказания в требуемом формате (sample_submission.csv).

Данные:
* video.ittensive.com/machine-learning/clouds/train.csv.gz (54 Мб)
* video.ittensive.com/machine-learning/clouds/train_images_small.tar.gz (212 Мб)
* video.ittensive.com/machine-learning/clouds/test_images_small.tar.gz (142 Мб)
* video.ittensive.com/machine-learning/clouds/sample_submission.csv.gz

Модели:
* video.ittensive.com/machine-learning/clouds/unet.fish.h5
* video.ittensive.com/machine-learning/clouds/fpn.fish.h5
* video.ittensive.com/machine-learning/clouds/pspnet.fish.h5

Итоговый файл с кодом (.py или .ipynb) выложите в github с портфолио.
"""


