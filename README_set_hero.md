How to set the homepage (hero) image for the Streamlit app

1) Copy your pneumonia X-ray image to the project and run the helper script:

```powershell
python .\tools\set_hero.py "C:\full\path\to\your\pneumonia_image.png"
```

2) Alternatively, start the Streamlit app and upload the image from the Home page, then click "Enregistrer comme image héros".

The script writes the image to `assets/hero_xray.png` and the app will pick it automatically on next load.
