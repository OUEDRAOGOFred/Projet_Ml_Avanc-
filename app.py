import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import re
import io
import base64
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Détection Pneumonie - IA Médicale",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Application d'IA médicale pour la détection de pneumonie à partir de radiographies thoraciques."
    }
)

try:
    # prefer absolute path of this file
    _BASE = os.path.dirname(os.path.abspath(__file__))
    if not _BASE:
        _BASE = os.getcwd()
except Exception:
    _BASE = os.getcwd()

# Inject custom CSS from assets (ensures styles are applied even when Streamlit doesn't auto-inject)
css_path = os.path.join(_BASE, 'assets', 'style.css')
if os.path.exists(css_path):
    try:
        with open(css_path, 'r', encoding='utf-8') as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except Exception:
        # fallback: attempt to reference the asset URL
        st.markdown("<link rel='stylesheet' href='/assets/style.css'>", unsafe_allow_html=True)
else:
    # no local CSS found, try served asset path
    st.markdown("<link rel='stylesheet' href='/assets/style.css'>", unsafe_allow_html=True)

def load_model(path):
    try:
        model = tf.keras.models.load_model(path)
        return model
    except Exception as e:
        st.error(f"Impossible de charger le modèle: {e}")
        return None


def get_model_input_size(model):
    """Retourne (width, height, channels) attendu par le modèle Keras.
    Si impossible de déterminer, retourne (128, 128, 3) par défaut.
    """
    try:
        shape = None
        # Try common attributes
        if hasattr(model, 'input_shape') and model.input_shape is not None:
            shape = model.input_shape
        elif hasattr(model, 'inputs') and model.inputs:
            # TensorShape or tuple
            s = model.inputs[0].shape
            try:
                shape = tuple(int(x) if x is not None else None for x in s)
            except Exception:
                shape = s

        if shape is None:
            return (128, 128, 3)

        # Normalize to tuple
        if isinstance(shape, tuple):
            sh = shape
        else:
            try:
                sh = tuple(shape)
            except Exception:
                return (128, 128, 3)

        # Possible formats: (None, H, W, C) or (None, C, H, W) or (H, W, C)
        if len(sh) == 4:
            # unpack, allow None values
            n, a, b, c = sh
            try:
                a_i = int(a) if a is not None else None
                b_i = int(b) if b is not None else None
                c_i = int(c) if c is not None else None
            except Exception:
                return (128, 128, 3)

            # Heuristic: if 'a' is small (1 or 3) and 'b' large -> channels_first
            if a_i in (1, 3) and (b_i is not None and b_i > 10):
                channels = a_i
                height = b_i
                width = c_i
            else:
                height = a_i
                width = b_i
                channels = c_i

            width = int(width) if width is not None else 128
            height = int(height) if height is not None else 128
            channels = int(channels) if channels is not None else 3
            return (width, height, channels)

        if len(sh) == 3:
            a, b, c = sh
            try:
                width = int(b)
                height = int(a)
                channels = int(c)
                return (width, height, channels)
            except Exception:
                return (128, 128, 3)

        return (128, 128, 3)
    except Exception:
        return (128, 128, 3)


@st.cache_resource
def load_csv(path):
    return pd.read_csv(path)


def preprocess_image(img: Image.Image, target_size=(128, 128), channels=3):
    # target_size expected as (width, height)
    if channels == 1:
        img = img.convert('L')
    else:
        img = img.convert('RGB')
    img = img.resize(target_size)
    arr = np.array(img)
    # If grayscale, expand last axis
    if channels == 1 and arr.ndim == 2:
        arr = np.expand_dims(arr, -1)
    arr = arr.astype('float32') / 255.0
    return np.expand_dims(arr, 0)


def main():
    
    # Sidebar: navigation only, no options
    with st.sidebar:
        local_logo = os.path.join(_BASE, 'assets', 'logo.png')
        if os.path.exists(local_logo):
            st.image(local_logo, width=120)
        else:
            st.markdown("<div style='height:88px'></div>", unsafe_allow_html=True)
        st.markdown("## Navigation")
        page = st.selectbox('Aller à', ['Accueil', 'Outputs', 'Prédiction'])
        st.markdown('---')
        st.write("Aide:")
        st.write("- Accueil: présentation du projet")
        st.write("- Outputs: toutes les sorties du notebook (images, tables, métriques, texte)")
        st.write("- Prédiction: interface pour prédire une image avec le modèle CNN")

    # Fixed threshold (kept fixed per earlier user preference)
    threshold = 0.5

    # Use default paths for model and CSV
    model_path = os.path.join(_BASE, "best_cnn_model_final.h5")
    csv_path = os.path.join(_BASE, "resultats_comparaison_modeles.csv")
    model = None
    if os.path.exists(model_path):
        with st.spinner('Chargement du modèle...'):
            model = load_model(model_path)
    else:
        st.warning(f"Fichier modèle introuvable: {model_path}")

    model_target_size = (128, 128)
    model_channels = 3
    if model is not None:
        w, h, ch = get_model_input_size(model)
        model_target_size = (w, h)
        model_channels = ch

    # Helper: extract a short blurb/title/authors from the notebook
    def read_notebook_blurb(nb_path):
        try:
            import json
            if not os.path.exists(nb_path):
                return None
            with open(nb_path, 'r', encoding='utf-8') as nf:
                nb = json.load(nf)
            # search cells for a PROJET header or a markdown title
            for cell in nb.get('cells', []):
                src = ''.join(cell.get('source', []))
                if 'PROJET' in src.upper() or 'Détection' in src or 'Detection' in src:
                    # take first non-empty line as blurb
                    for line in src.splitlines():
                        s = line.strip()
                        if s and not s.startswith('#!'):
                            return s
            # fallback: first non-empty markdown or code cell line
            for cell in nb.get('cells', []):
                src = ''.join(cell.get('source', []))
                for line in src.splitlines():
                    s = line.strip()
                    if s:
                        return s
        except Exception:
            return None
        return None

    # Helper: collect project images (reuse existing logic)
    nb_outputs_dir = os.path.join(_BASE, 'notebook_outputs')
    imgs_dir = os.path.join(nb_outputs_dir, 'images')

    def collect_project_images():
        candidates = []
        if os.path.exists(imgs_dir):
            for f in os.listdir(imgs_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    candidates.append(os.path.join(imgs_dir, f))
        for f in os.listdir(_BASE):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(_BASE, f)
                candidates.append(path)
        seen = set()
        out = []
        for p in candidates:
            n = os.path.normpath(p)
            if n not in seen:
                seen.add(n)
                out.append(n)
        return out

    project_images = collect_project_images()

    # Renderers for pages
    def render_home():
        st.markdown("<div style='padding:10px 0'></div>", unsafe_allow_html=True)
        # Title & blurb
        nb_path = os.path.join(_BASE, 'Untitled10_(5).ipynb')
        blurb = read_notebook_blurb(nb_path) or 'Détection de la pneumonie à partir de radiographies — Aide au diagnostic assistée par un modèle CNN.'
        st.markdown('<div class="title-card"><div class="floating-particles"></div><h1>Projet — Détection de la Pneumonie</h1></div>', unsafe_allow_html=True)
        st.markdown(f"### {blurb}")

        # 🎨 Beautiful Hero Section - Enhanced Premium Design
        col1, col2 = st.columns([2, 1], gap="large")
        
        with col1:
            st.markdown("""
            <div style="
                padding: 2.5rem; 
                background: linear-gradient(135deg, rgba(102, 126, 234, 0.12) 0%, rgba(240, 147, 251, 0.12) 100%); 
                border-radius: 24px; 
                margin: 1.5rem 0;
                border: 2px solid rgba(255, 255, 255, 0.6);
                box-shadow: 0 20px 50px rgba(0, 0, 0, 0.08);
                backdrop-filter: blur(10px);
                transition: all 0.4s ease;
                position: relative;
                overflow: hidden;
            ">
                <div style="position: absolute; top: -50%; right: -50%; width: 200%; height: 200%; background: radial-gradient(circle, rgba(102, 126, 234, 0.08) 0%, transparent 70%); animation: pulse 4s ease-in-out infinite;"></div>
                <div style="position: relative; z-index: 1;">
                    <h2 style="
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
                        background-clip: text;
                        -webkit-background-clip: text;
                        -webkit-text-fill-color: transparent;
                        margin-bottom: 1.25rem;
                        font-size: 2rem;
                        font-weight: 800;
                    ">🫁 IA Médicale — Détection Pneumonie</h2>
                    <p style="color: #475569; font-size: 1.15rem; line-height: 1.8; font-weight: 500;">
                        Solution d'intelligence artificielle avancée utilisant des <strong style="color: #667eea; font-weight: 700;">réseaux de neurones convolutionnels</strong> 
                        pour la détection automatique de pneumonie à partir d'images radiographiques thoraciques.
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Premium Badges with Animations
            st.markdown("""
            <div style="display: flex; gap: 1rem; margin: 1.5rem 0; flex-wrap: wrap;">
                <span style="
                    background: linear-gradient(135deg, rgba(34, 197, 94, 0.15) 0%, rgba(34, 197, 94, 0.05) 100%); 
                    color: #16a34a; 
                    padding: 0.65rem 1.25rem; 
                    border-radius: 50px; 
                    font-size: 0.95rem; 
                    font-weight: 700;
                    border: 2px solid rgba(34, 197, 94, 0.2);
                    box-shadow: 0 4px 15px rgba(34, 197, 94, 0.15);
                    transition: all 0.3s ease;
                    cursor: default;
                ">✅ Précision >95%</span>
                <span style="
                    background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(102, 126, 234, 0.05) 100%); 
                    color: #4f46e5; 
                    padding: 0.65rem 1.25rem; 
                    border-radius: 50px; 
                    font-size: 0.95rem; 
                    font-weight: 700;
                    border: 2px solid rgba(102, 126, 234, 0.2);
                    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.15);
                    transition: all 0.3s ease;
                    cursor: default;
                ">🧠 Deep Learning</span>
                <span style="
                    background: linear-gradient(135deg, rgba(240, 147, 251, 0.15) 0%, rgba(240, 147, 251, 0.05) 100%); 
                    color: #c026d3; 
                    padding: 0.65rem 1.25rem; 
                    border-radius: 50px; 
                    font-size: 0.95rem; 
                    font-weight: 700;
                    border: 2px solid rgba(240, 147, 251, 0.2);
                    box-shadow: 0 4px 15px rgba(240, 147, 251, 0.15);
                    transition: all 0.3s ease;
                    cursor: default;
                ">⚡ Temps Réel</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="
                text-align: center; 
                padding: 2.5rem 2rem; 
                background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%); 
                border-radius: 24px; 
                box-shadow: 0 20px 50px rgba(0,0,0,0.12);
                border: 2px solid rgba(102, 126, 234, 0.1);
                transition: all 0.4s ease;
                position: relative;
                overflow: hidden;
            ">
                <div style="position: absolute; top: 0; left: 0; right: 0; height: 4px; background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);"></div>
                <div style="font-size: 4.5rem; margin-bottom: 1.25rem; animation: float 3s ease-in-out infinite;">🔬</div>
                <h3 style="color: #0f172a; margin-bottom: 0.75rem; font-weight: 700; font-size: 1.5rem;">Analyser l'IA</h3>
                <p style="color: #64748b; font-size: 1rem; line-height: 1.6;">Diagnostic assisté par intelligence artificielle</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Premium Features Section
        st.markdown('<h3 style="margin-top: 3rem; margin-bottom: 1.5rem; text-align: center;">🎯 Caractéristiques Clés</h3>', unsafe_allow_html=True)
        
        feat1, feat2, feat3 = st.columns(3, gap="large")
        with feat1:
            st.markdown("""
            <div class="feature-card feature-card-blue">
                <div style="font-size: 3rem; margin-bottom: 1rem; filter: drop-shadow(0 4px 8px rgba(0,0,0,0.1));">📊</div>
                <h4 style="color: #0f172a; margin-bottom: 0.75rem; font-weight: 700; font-size: 1.25rem;">Analyse Avancée</h4>
                <p style="color: #64748b; font-size: 0.95rem; line-height: 1.6;">Traitement d'images médicales avec CNN optimisés</p>
            </div>
            """, unsafe_allow_html=True)
        
        with feat2:
            st.markdown("""
            <div class="feature-card feature-card-green">
                <div style="font-size: 3rem; margin-bottom: 1rem; filter: drop-shadow(0 4px 8px rgba(0,0,0,0.1));">⚡</div>
                <h4 style="color: #0f172a; margin-bottom: 0.75rem; font-weight: 700; font-size: 1.25rem;">Rapidité</h4>
                <p style="color: #64748b; font-size: 0.95rem; line-height: 1.6;">Résultats en moins de 3 secondes</p>
            </div>
            """, unsafe_allow_html=True)
        
        with feat3:
            st.markdown("""
            <div class="feature-card feature-card-pink">
                <div style="font-size: 3rem; margin-bottom: 1rem; filter: drop-shadow(0 4px 8px rgba(0,0,0,0.1));">🔒</div>
                <h4 style="color: #0f172a; margin-bottom: 0.75rem; font-weight: 700; font-size: 1.25rem;">Fiabilité</h4>
                <p style="color: #64748b; font-size: 0.95rem; line-height: 1.6;">Validation médicale approfondie</p>
            </div>
            """, unsafe_allow_html=True)

        # Authors & quick facts from notebook (try to extract authors lines and show with line breaks)
        try:
            import json
            with open(nb_path, 'r', encoding='utf-8') as nf:
                nb = json.load(nf)
            authors = None
            for cell in nb.get('cells', []):
                src = ''.join(cell.get('source', []))
                if 'Auteur' in src or 'Author' in src:
                    for line in src.splitlines():
                        if 'Auteur' in line or 'Author' in line:
                            authors = line.strip().lstrip('#').strip()
                            break
                if authors:
                    break
            if authors:
                # Handle the "Auteur :" prefix and split authors properly
                if authors.startswith('Auteur :') or authors.startswith('Author :'):
                    prefix = 'Auteur :'
                    author_list = authors[len(prefix):].strip()
                else:
                    prefix = ''
                    author_list = authors
                
                # Split by commas and display each author on a new line
                parts = [p.strip() for p in author_list.split(',') if p.strip()]
                for i, author in enumerate(parts):
                    if i == 0:
                        st.write(f"**{prefix} {author}**")
                    else:
                        st.write(f"**{author}**")
        except Exception:
            pass

        st.markdown('---')
        st.write('Utilisez le menu de gauche pour naviguer vers les outputs ou faire une prédiction.')

    def render_outputs():
        st.header('📊 Sorties du Notebook — Résultats & Analyses')
        
        # NEW: Display structured text outputs first
        text_outputs_path = os.path.join(nb_outputs_dir, 'text_outputs.json')
        if os.path.exists(text_outputs_path):
            try:
                import json
                with open(text_outputs_path, 'r', encoding='utf-8') as tof:
                    text_outputs = json.load(tof)
                
                if text_outputs:
                    st.subheader('📋 Résultats d\'Entraînement & Évaluation')
                    
                    for idx, output in enumerate(text_outputs):
                        txt = output.get('text', '')
                        
                        # Skip empty or very short outputs
                        if not txt or len(txt.strip()) < 10:
                            continue
                        
                        # Identify important outputs by keywords
                        txt_lower = txt.lower()
                        is_important = any(kw in txt_lower for kw in [
                            'architecture', 'summary', 'évaluation', 'résultats', 
                            'accuracy', 'auc', 'precision', 'recall', 'f1-score',
                            'confusion', 'classification', 'comparaison', 'meilleur',
                            'extraction', 'entraînement', 'gridsearch', 'temps'
                        ])
                        
                        if is_important:
                            # Display in a nice card
                            with st.expander(f"📄 Output {idx + 1}", expanded=(idx < 3)):
                                st.code(txt, language='text')
            except Exception as e:
                st.warning(f"Impossible de charger les sorties textuelles: {e}")
        
        st.markdown('---')
        
        # Figures generated by notebook (expected filenames)
        # Figures section - only show if figures exist
        expected_pngs = [
            'cnn_evaluation_complete.png',
            'comparaison_finale_modeles.png',
            'matrices_confusion_comparaison.png',
            'courbes_roc_comparaison.png',
            'analyse_erreurs.png'
        ]
        found = [os.path.join(_BASE, f) for f in expected_pngs if os.path.exists(os.path.join(_BASE, f))]
        if found:
            st.subheader('📈 Figures et Graphiques')
            for p in found:
                st.image(p, caption=os.path.basename(p), use_container_width=True)

        # Show the user-provided comparison CSV (if present)
        if os.path.exists(csv_path):
            st.markdown('---')
            st.subheader('📊 Tableau de Comparaison des Modèles')
            try:
                df_cmp = load_csv(csv_path)
                st.dataframe(df_cmp, use_container_width=True)
                try:
                    with open(csv_path, 'rb') as bf:
                        csv_bytes = bf.read()
                    st.download_button(
                        f'📥 Télécharger {os.path.basename(csv_path)}', 
                        data=csv_bytes, 
                        file_name=os.path.basename(csv_path), 
                        mime='text/csv'
                    )
                except Exception:
                    pass
            except Exception as e:
                st.warning(f"Impossible de charger le CSV: {e}")

        # Extracted images listing (if any from notebook)
        if os.path.exists(imgs_dir):
            img_list = [f for f in os.listdir(imgs_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if img_list:
                st.markdown('---')
                st.subheader('🖼️ Images Extraites du Notebook')
                cols = st.columns(2)
                for i, fname in enumerate(img_list):
                    path = os.path.join(imgs_dir, fname)
                    col = cols[i % len(cols)]
                    with col:
                        st.image(path, caption=fname, use_container_width=True)
                        try:
                            with open(path, 'rb') as bf:
                                data = bf.read()
                            st.download_button(
                                f'📥 {fname}', 
                                data=data, 
                                file_name=fname, 
                                mime='image/png' if fname.lower().endswith('.png') else 'image/jpeg',
                                key=f"download_{fname}"
                            )
                        except Exception:
                            pass

    def render_predict():
        st.subheader('Prédiction d\'image')
        col1, col2 = st.columns([2,1])
        with col1:
            uploaded = st.file_uploader("Déposez une image ici", type=['png','jpg','jpeg'])
            if uploaded is not None:
                img = Image.open(uploaded)
                st.image(img, caption="Image uploadée", width='stretch')
                if model is not None:
                    x = preprocess_image(img, target_size=model_target_size, channels=model_channels)
                    preds = model.predict(x)
                    if preds.ndim == 2 and preds.shape[1] > 1:
                        probs = tf.nn.softmax(preds[0]).numpy()
                        top_idx = int(np.argmax(probs))
                        top_prob = float(probs[top_idx])
                        st.markdown(f"**Classe prédite:** {top_idx} — **Confiance:** {top_prob:.2%}")
                        dfp = pd.DataFrame({'classe': np.arange(len(probs)), 'probabilité': probs}).sort_values('probabilité', ascending=False)
                        st.dataframe(dfp.head(5))
                    else:
                        val = float(preds.ravel()[0])
                        prob = 1.0 / (1.0 + np.exp(-val)) if (val < 0 or val > 1) else val
                        st.markdown(f"**Valeur brute (sortie modèle):** {val:.4f}")
                        st.markdown(f"**Probabilité interprétée:** {prob:.2%}")
                        if prob < threshold:
                            st.success(f"Décision: NORMAL — probabilité {prob:.2%} < {threshold:.2f} → pas de pneumonie probable.")
                        else:
                            st.error(f"Décision: PNEUMONIA — probabilité {prob:.2%} ≥ {threshold:.2f} → pneumonie probable.")
                        prog = st.progress(0)
                        try:
                            prog.progress(int(prob * 100))
                        except Exception:
                            st.write(f"Probabilité: {prob:.2%}")
                else:
                    st.info("Chargez d'abord un modèle valide dans la barre latérale pour prédire.")

        with col2:
            # Sidebar placeholder: reserved for future prediction details (intentionally left blank)
            pass

    # Page dispatch
    if 'page' in locals() and page == 'Accueil':
        render_home()
    elif 'page' in locals() and page == 'Outputs':
        render_outputs()
    else:
        # default to predict if unknown
        render_predict()

   
if __name__ == '__main__':
    main()
