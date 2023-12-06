Le code se base sur https://github.com/openai/guided-diffusion

Le fichier unet.py doit remplacer celui de la bibliothèque guided-diffusion, il contient une classe en plus qui se base sur le Unet d'origine mais permettra d'avoir un nombre de couches flexible pour pouvoir segmenter comme on souhaite

Le fichier unet.sh contient les flags à utiliser pour correspondre au mieux au papier "Diffusion models beat gans on image synthesis" ( https://arxiv.org/pdf/2105.05233.pdf )
Il faut l'utiliser avec la commande "source"

L'entrainement se fait donc d'abord en 2 temps
Dans un premier temps on entraine avec des données non labellisées le DDPM grace à la bibliothèque guided diffusion

(Eventuellement il faudra peut etre retaper les arguments à la main ..)
OPENAI_LOGDIR est le dossier où s'écrivent les poids
data_dir est le chemin vers le dossier contenant les data non labellisées

OPENAI_LOGDIR=$OPENAI_LOGDIR python scripts/image_train.py --data_dir ~/Desktop/Bitewings_Resized_256 $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS

Une fois ce modèle entrainé, nous allons l'utiliser pour faire de la segmentation.
Grace à la diffusion, le modèle aura appris les features importantes de ces images et aura besoin de moins de données (voire très peu, c'est l'objectif) pour apprendre à segmenter
Les masques de segmentation sont sous en nuances de gris, avec les valeurs allant de 0 à C (nombres de classes)
A l'oeil nu on ne voit donc pas grand chose sur ces images ...

Ces masques de segmentation ont été créés à l'aide du script rgb2tif.py, je définis dans ce script les couleurs à garder, il faudra adapter pour les segmentations futures (ou refaire un autre script)
Les données de segmentation sont dans le dossier Dataset_segmentation, le split train/val a déjà été fait dans Split et Split2 (l'un fait du 50/50, l'autre s'approche du papier d'Allisone.ai où il avait moins d'une dizaine d'images en train)
De la même manière dans le predict.py, le dictionnaire categories sera à adapter

On entraine ensuite le modèle de segmentation avec 

python finetune.py checkpoints/model410000.pt ~/Desktop/Dataset_segmentation/Split/TRAIN/IMG ~/Desktop/Dataset_segmentation/Split/TRAIN/MASK ~/Desktop/Dataset_segmentation/Split/VAL/IMG ~/Desktop/Dataset_segmentation/Split/VAL/MASK $(eval echo $MODEL_FLAGS) $(eval echo $TRAIN_FLAGS) --lr 1e-4 --image_size 256  --out_channels 7 --epochs 200 --weights_save segm.pth --batch_size 2 --unfreeze_decoder 1. 

