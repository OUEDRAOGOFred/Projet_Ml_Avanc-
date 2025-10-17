# Application Streamlit — Projet ML Avancé

Ce dépôt contient une application Streamlit simple pour charger le modèle `best_cnn_model_final.h5`, uploader des images et visualiser `resultats_comparaison_modeles.csv`.

Prérequis
- Python 3.10+ recommandé

Installation (PowerShell)

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

Lancer l'app

```powershell
streamlit run app.py
```

Conseils
- Placez `best_cnn_model_final.h5` et `resultats_comparaison_modeles.csv` à la racine du projet (déjà présent normalement)
- Pour améliorer l'apparence, modifiez `assets/style.css`
