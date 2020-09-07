# Parcours Data Analyst : Projet 8

Ce projet était un projet libre. J'ai décidé de travailler sur la modélisation du prix du baril *WTI* (référence pour le brut nord américain). Mon objectif était de voir s'il était possible de trouver un modèle donnant de **meilleurs résultats que le modèle naïf assumant un prix constant**. Ce dernier, bien qu'extrêmement simple est le plus utilisé pour la valorisation d'actifs pétroliers.

Les données ont été récupérés sur le site de l'agence américaine de l'énergie **U.S. Energy Information Administration**.

L'analyse et les résultats sont présentés sous forme d'un notebook jupyter.  
Les librairies python nécessaires pour pouvoir lancer le notebook sont regroupées dans le fichier *requirements.txt*.

Toutes les fonctions créées afin de mener à bien le projet ont été regroupées dans le fichier *Project8.py*.

Le prix pétrole ayant connu de nombreuses crises depuis les débuts de son exploitation il y a plus d'un siècle, il fut nécessaire d'analyser minutieusement l'évolution des cours afin de trouver une période pouvant être considérée comme représentative de la période actuelle. Le but étant d'éviter d'inclure dans mon analyse des variations anomaliques liées à des évènements historiques, évènements difficilement reproductible de nos jours. **C'est la période de 2010 à fin 2018 qui fut choisie.**

Après avoir établi la période sur laquelle j'allais pouvoir travailler, il a fallu trouver les variables les plus à même de modéliser le prix du baril. 
Les variables les plus prometteuses étaient :
- La consommation de produits pétroliers;
- La variation des stocks;
- La production pétrolière décalée de 12 mois;

J'ai pu alors tester une regression Ridge à partir de ces variables afin de modéliser les prix. Les résultats fut convaincants et j'ai donc décidé de poursuivre avec ce modèle. J'envisage de tester d'autres modèles plus complexes (Random Forest, Gradient Boosting, XGBoost, MLP) pour voir s'il est possible d'affiner encore un peu plus résultats. J'envisage aussi de mettre la période (notamment la date de fin) en hyperparamètre.

Pour finir, j'ai utilisé l'algorithme SARIMA (présence d'une saisonnalité) pour prédire l'évolution des stocks ainsi que la consommation (il n'était pas nécessaire de prédire la production car cette dernière était décalée de 12 mois). Les résultats du SARIMA, ont montré **une bonne prédiction jusqu'à 6 mois**.

Le modèle sélectionné précédemment ainsi que les prédictions de la consommation & des stocks (obtenus par SARIMA) ont été utilisés afin de prédire le *WTI*. Le graphique ci-dessous montre les résultats de la prédicition par rapport aux données réelles.

![WTI Forecast](/charts/WTI_results.png "WTI Forecast")
