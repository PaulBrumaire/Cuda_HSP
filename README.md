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

La difficulté principale pour réaliser les fonctions de convolutions et subsampling était de bien additioner les bons indices entre eux. En effet entre le fait qu'on travaille avec des matrices de matrices et qu'il était plus facile en C de représenter cette matrice de matrice sous la forme d'un unique vecteur, il était très facile d'avoir les mauvais indices.

Ci-dessous vous pouvez voir un exemple de fonction codé en cuda, la concolution2D. On peut notamment observer qu'il y a bien du parallélisme dans les opérations effectués mais qu'une double boucle for est quand même nécessaire pour parcourir l'entiereté des kernels qui eux aussi sont des matrices de matrices.

![image](https://user-images.githubusercontent.com/74967118/149662460-f5f34125-9b0e-4641-8842-323ee2b9b6cb.png)

Enfin il fallait veiller à utiliser les bonnes fonctions d'activation, on peut facilement savoir laquelle il faut utiliser en allant chercher le modèle sur internet.

On a également tester notre travail en permanence pour avoir une idée de ce que le programme faisait. Cela explique la présence de fonction d'initialisations de matrices.

## Partie 3 - Un peu de Python

Dans cette partie on passe au langage python afin d'entraîner le modèle est derécupérer les poids associer aux différents kernels. Il est en effet plus aisé de faire cela sur python où des bibliothèques comme tensorflow ou keras nous facilitent grandement le travail.

Une fois le modèle entrainé, nous avons du réordonner les poids (grâce notamment à la fonction reshape), puis nous les avons sauvegarder dans des fichiers .h que nous pourrons facilement récupérer dans notre code cuda plus tard.

## Partie 4 - Finalisation du modèle et résultats

Pour finaliser le modèle il manquait deux étapes, créer les dernières couches, ce qui était assez facile vu que ce sont des couches "dense" (c'est à dire tout les neuronnes de sorties sont reliés à tout les neuronnes d'entrées) qui ne nécessitent donc que de simples opérations et multiplications. Et ajouter les bons poids aux kernels initialiser de façon aléatoire jusqu'a maintenant.

Une fois ce travail fait on a pu tester notre modèle, malheureusement celui-ci n'arriver pas à classifier correctement les chiffres présents sur la base de données MNIST. On obtenait des probabilités très proches pour les différents chiffres. On s'est donc interrogé sur d'où pouvait provenir cette erreur et certains points nous paraissent plus probables que d'autres. Tout d'abord la façon dont on a réordonner les poids des kernels pourrait être source de l'erreur et sinon l'erreur proviendrait de nos convolutions où nous pourrions avoir des problèmes d'indice.
