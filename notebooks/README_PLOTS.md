# Generation de Graphiques pour les Notebooks

## Generation automatique des graphiques

Pour generer tous les graphiques PNG automatiquement, executez:

```bash
python generate_plots.py
```

Ce script va:
- Charger le dataset
- Generer tous les graphiques des notebooks
- Sauvegarder les PNG dans `outputs/figures/`

## Graphiques generes

### Notebook 01 - Exploration
- `01_exploration_01_histograms.png` - Distributions des metriques
- `01_exploration_02_temporal_patterns.png` - Courbes temporelles moyennes
- `01_exploration_03_heatmap_comparison.png` - Heatmap de comparaison
- `01_exploration_04_correlation_heatmaps.png` - Matrices de correlation
- `01_exploration_05_series_with_anomalies.png` - Series avec anomalies
- `01_exploration_06_profile_distributions.png` - Distributions par profil
- `01_exploration_07_attack_type_distributions.png` - Distributions par type d'attaque

### Notebook 02 - Feature Engineering
- `02_feature_engineering_01_rolling_features.png` - Features de rolling
- `02_feature_engineering_02_pca.png` - Visualisation PCA
- `02_feature_engineering_03_distributions.png` - Distributions des features
- `02_feature_engineering_04_importance_heatmap.png` - Heatmap d'importance

### Notebook 04 - Evaluation
- `04_evaluation_01_roc_pr_curves.png` - Courbes ROC et PR
- `04_evaluation_02_score_distributions.png` - Distributions des scores

## Execution des notebooks

Pour executer les notebooks manuellement dans Jupyter:

1. Ouvrez Jupyter: `jupyter notebook`
2. Ouvrez les notebooks dans l'ordre:
   - `01_exploration.ipynb`
   - `02_feature_engineering.ipynb`
   - `03_modeling.ipynb`
   - `04_evaluation.ipynb`

Les graphiques seront affiches dans les cellules et sauvegardes automatiquement en PNG.

## Note importante

Les notebooks utilisent `%matplotlib inline` pour afficher les graphiques directement dans les cellules. Tous les graphiques sont aussi sauvegardes automatiquement dans `outputs/figures/` avec `plt.savefig()`.

