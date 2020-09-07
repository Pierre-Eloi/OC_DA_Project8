# coding: utf-8

"""Contient toutes les fonctions de visualisation qui seront utilisées dans le projet 8."""

import numpy as np
import matplotlib.pyplot as plt

def series_plot(data, x=None, series_name=None, kind="line", colors=None, legend=True, change_ymin=False,
                x_label="", y_label="", source="", title_s="", subtitle_s="", file_s=""):
    """Tracer une série.
    -----------
    Paramètres :
    data : jeu de données sous forme de DataFrame
    x : le nom de la colonne des abscisses, si rien n'est renseigné prend les index
    series_name : Liste des noms des séries temporelles
    kind : type de graphique
    colors : une liste des couleurs devant être utilisée (optionnel)
    legend : Booléen, précise si on doit 
    x_label : nom de l'axe des abscisses
    y_label : nom de l'axe des ordonnées
    kind : type de graphique que l'on souhait obtenir
    source : nom de la source
    title_s : texte du titre
    title_s : texte du sous-titre
    file_s : Sous quel nom sauvegarder le graphique
    -----------
    Retourne :
    axes : Matplotlib axis object
    """
    # On configure les paramètres du graphique
    plt.rcParams.update(plt.rcParamsDefault)
    plt.style.use('bmh')

    # On trace le graphique
    if colors:
        if legend:
            plot = data.plot(x=x, y=series_name, kind=kind, figsize=(8, 5),
                         linewidth=2.0, color=colors)
        else:
            plot = data.plot(x=x, y=series_name, kind=kind, figsize=(8, 5),
                         linewidth=2.0, color=colors, legend=False)
    else:
        if legend:
            plot = data.plot(x=x, y=series_name, kind=kind, figsize=(8, 5),
                         linewidth=2.0)
        else:
            plot = data.plot(x=x, y=series_name, kind=kind, figsize=(8, 5),
                         linewidth=2.0, legend=False)
    # On modifie les axes     
    plot.tick_params(axis="both", which="both", labelsize=10, colors="#333333")
    x_min, x_max = plot.get_xlim()
    y_min, y_max = plot.get_ylim()
    if change_ymin:
        y_min -= (y_max-y_min) * 0.075
        plot.set_ylim(bottom=y_min)
    plot.set_xlabel(x_label, color="#333333", weight="bold", fontsize=10)  
    plot.set_ylabel(y_label, color="#333333", weight="bold", fontsize=10)
    # On ajoute une signature en dessous
    txt_x = x_min - (x_max-x_min)*.05
    txt_y = y_min - (y_max-y_min)*.15
    plot.text(x=txt_x, y=txt_y, s=" PE.RAGETLY", fontsize=9, color="#333333")
    plot.text(x=x_max+(x_max-x_min)*.012, y=txt_y, s="SOURCE : "+source.upper(),
              fontsize=8, color="#333333", horizontalalignment="right")
    # On ajoute une ligne horizontale au-dessus de la signature
    arr_x = x_min - (x_max-x_min)*.06
    arr_y = y_min - (y_max-y_min)*.12
    plot.text(x=arr_x, y=arr_y, s=" "*(167), fontsize=9,
              bbox=dict(edgecolor="#333333", facecolor="#333333", pad=-4.5, linewidth=.5))   
    # Ajout d'un titre et d'un sous-titre
    title_y = y_max + (y_max-y_min)*.13
    subtitle_y = y_max + (y_max-y_min)*.07
    plot.text(x=txt_x, y=title_y, s=title_s, fontsize=15,
                weight="bold", color="#333333")
    plot.text(x=txt_x, y=subtitle_y, s=subtitle_s, fontsize=10,
                color="#333333")    
    # Sauvegarde du graphique
    plt.savefig(file_s + ".png", bbox_inches = "tight")
    plt.show()
    
def r2_score(y_true, y_pred):
    """Donne le R^2 d'une régression
    -----------
    Paramètres :
    y_true : Valeurs réelles (array)
    y_pred : Valeurs prédites (array)
    -----------
    Retourne :
    float : R^2 score
    """
    mu = np.mean(y_true)
    TSS = np.sum((y_true-mu)**2)
    TSM = np.sum((y_pred-mu)**2)
    r2 = TSM/TSS
    return round(r2, 3)

def rmse(y_true, y_pred):
    """Donne le RMSE (Root Mean Square Error) d'une prédiction
    -----------
    Paramètres :
    y_true : Valeurs réelles (array)
    y_pred : Valeurs prédites (array)
    -----------
    Retourne :
    float : RMSE score
    """
    rmse = np.sqrt(np.mean((y_true-y_pred)**2))
    return round(rmse, 2)
    
def acf_plot(series, nlags=None, partial=False, unbiased=False):
    """Autocorrelation plot pour Séries temporelles.
    -----------
    Paramètres :
    series (series) : Série temporelle
    nlags (int, optionnal) : nombre de retard pris en compte pour l'acf
    unbiased (bool, optionnal) : Si vrai, utilisation de la version non biaisée pour les autocorrélations simples
    partial (bool, optionnal): Si vrai fonction d'autocorrélation partielle
    kwds : keywords
    Options to pass to matplotlib plotting method
    -----------
    Retourne :
    axes : Matplotlib axis object
    """
    if nlags is None:
        nlags = series.size - 1
    from statsmodels.tsa.stattools import acf, pacf
    n = series.size
    data = np.asarray(series)
    z95 = 1.96
    x = np.arange(nlags+1)
    y_acf = acf(data, unbiased, nlags)
    
    plt.rcParams.update(plt.rcParamsDefault)
    plt.style.use('bmh')
    if partial:
        y_pacf = pacf(data, nlags)
        fig = plt.figure(figsize=(5,8))
        # autocorrélogramme simple
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.bar(x, y_acf, color='#348ABD', width = 0.2)
        ax1.axhline(y=z95 / np.sqrt(n), color='#666666', linestyle='--', linewidth=1.5)
        ax1.axhline(y=0.0, color='black', linewidth=1.5)
        ax1.axhline(y=-z95 / np.sqrt(n), color='#666666', linestyle='--', linewidth=1.5)
        ax1.tick_params(axis='both', which='major', labelsize=10, colors='#666666')
        ax1.set_ylim(-1, 1)
        ax1.set_xlabel("Lag", color='#333333', weight='bold', fontsize=10)
        ax1.set_ylabel("ACF", color='#333333', weight='bold', fontsize=10)
        ax1.set_title("Autocorrélations simples", color='#333333', weight='bold', fontsize=15)
        # autocorrélogramme partiel
        ax2 = fig.add_subplot(2, 1, 2)
        ax2.bar(x, y_pacf, color='#348ABD', width = 0.2)
        ax2.axhline(y=z95 / np.sqrt(n), color='#666666', linestyle='--', linewidth=1.5)
        ax2.axhline(y=0.0, color='black', linewidth=1.5)
        ax2.axhline(y=-z95 / np.sqrt(n), color='#666666', linestyle='--', linewidth=1.5)
        ax2.tick_params(axis='both', which='major', labelsize=10, colors='#666666')
        ax2.set_ylim(-1, 1)
        ax2.set_xlabel("Lag", color='#333333', weight='bold', fontsize=10)
        ax2.set_ylabel("PACF", color='#333333', weight='bold', fontsize=10)
        ax2.set_title("Autocorrélations partielles", color='#333333', weight='bold', fontsize=15)
    else:
        (fig, ax) = plt.subplots(1, 1, figsize=(5,4))
        ax.bar(x, y_acf, color='#348ABD', width = 0.2)
        ax.axhline(y=z95 / np.sqrt(n), color='#666666', linestyle='--', linewidth=1.5)
        ax.axhline(y=0.0, color='black', linewidth=1.5)
        ax.axhline(y=-z95 / np.sqrt(n), color='#666666', linestyle='--', linewidth=1.5)
        ax.tick_params(axis='both', which='major', labelsize=12, colors='#666666')
        ax.set_xlabel("Lag", color='#333333', weight='bold', fontsize=10)
        ax.set_ylabel("ACF", color='#333333', weight='bold', fontsize=10)
        ax.set_ylim(-1, 1)
        ax.set_title("Autocorrélations simples", color='#333333', weight='bold', fontsize=15)
    plt.tight_layout()
    plt.savefig("Charts/ACF_plot.png", bbox_inches = 'tight')
    plt.show()