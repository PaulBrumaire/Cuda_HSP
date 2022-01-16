# Cuda_HSP

## Implémentation d'un CNN - LeNet-5 sur GPU

### LeNet-5

L'objectif à terme de ces 4 séances est d'implémenter l'inférence dun CNN très claissque : LeNet-5.
Le travail devra être réalisé en langage C afin de pouvoir utiliser cuda et donc de paralleliser les calculs en travaillant avec le GPU.

![LeNet-5](https://user-images.githubusercontent.com/74967118/149662004-7c525c9d-e9c1-406f-8419-5766b7682711.png)

Nous avons décider de travailler avec l'IDE VSCode qui nous est tout les deux familière depuis deux ans maintenant.

## Partie 1 - Prise en main de Cuda : Multiplication de matrices

Afin de prendre en main toutes les subtilités de CUDA, nous allons commencer par coder quelques opérations simples sur les matrices. Nous avons donc réaliser des [additions](matrix_add.cu) et des [multiplications](matrix_mul.cu) à la fois en C et en cuda. Ainsi cela nous a permis de voir qu'en parallelisant les opérations on obtenait des temps de calculs plus petits, ce qui est l'effet recherché.

![image](https://user-images.githubusercontent.com/74967118/149662339-10c2fed6-aa15-4202-b3d8-d120b3ea3ee5.png)

En effet, on constate sur l'image ci-dessus qu'une boucle for n'est pas nécessaire pour parcourir les différents indices de la matrice, c'est en fait chaque cellule du GPU qui va effectuer un seul calcul en même temps que tout les autres cellules, et c'est cela qui rend le calcul plus rapide. 

## Partie 2 - Premières couches du réseau de neurone LeNet-5 : Convolution 2D et subsampling

Une fois qu'on a pris en main le langage cuda, il fallait s'attaquer aux différentes couches présentes dans le réseau.
L'ensemble du travail réalisé dans le langage cuda ce situe dans ce [fichier](conv.cu) pour la suite des TP.

![image](https://user-images.githubusercontent.com/74967118/149662460-f5f34125-9b0e-4641-8842-323ee2b9b6cb.png)
