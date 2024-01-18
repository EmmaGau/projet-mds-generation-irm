# projet-mds-generation-irm

## Clone guided-diffusion

https://github.com/openai/guided-diffusion and rename the folder guided_diffusion

## Scripts

### Train

J'ai déjà entrainé les modèles ils sont dans checkpoints. Mais si jamais pour lancer un entraînement il faut utiliser les scripts bash:

- Entrainement du modèle de diffusion: `./unet_paper.sh`
- Finetuning: `./unet_finetune.sh`

### Predict

- `./unet_predict.sh`

Hésites pas à changer les bash, en particulier predict si tu veux générer d'autres segmentations (arguments 2 (input) et 3 (output))


## Data

- Bitewings_Resized_256: 1791 images non labelisées pour entrainer le modèle de diffusion
- Dataset_segmentation: 50 images labélisées pour le finetuning. Il y a deux split, j'ai utilisé le premier pour l'entrainement (dans les splits les masques apparaissent noirs parce que les valeurs sont dans [1,6]). Les GT avec les couleurs sont dans Full/Mask. Dans les results j'ai mis les outputs pour Split/VAL

## Notes

- Vu que l'entrainement prend pas mal de temps je l'ai fait pour des images de 64x64 (faudra juste resize les gt pour comparer les resultats du coup)
- Toute la théorie est expliquée dans l'article
- Si je trouve le temps j'aimerais bien réussir à voir la tête des sorties du modèle de diffusion de base (juste une reconstruction de l'image de dent du coup), mais vas y j'ai pas trouvé comment faire, faut appliquer un post-processing à la sortie et je sais pas lequel. Déjà on a les dents segmentées, on est pas mal haha
- Dans le dossier guided-diffusion, tous les fichiers exeptés LICENSE, model-card.py, README.md et setup sont des trucs rajoutés (image_train vient du dossier script, c'est dans le package de base, mais les imports relatifs marchaient pas de le lancer depuis le dossier scipts du coup je l'ai mis dehors)