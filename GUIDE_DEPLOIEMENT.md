# Mettre le chatbot Lockall en ligne (gratuit)

## 1. Dans ton dossier de projet
Copie ces 4 fichiers à côté de `chatbot_lockall_propre.py` :
`app.py`, `requirements.txt` (remplace l'ancien), `.gitignore`, `.streamlit/secrets.toml.example`

Ton dossier doit contenir au final :
- app.py, chatbot_lockall_propre.py, vectorize_data.py
- lockall_articles.json, lockall_horaires.json
- requirements.txt, .gitignore
- le dossier faiss_index/ (voir étape 2)

NE PAS publier : `.env` et `venv/` (déjà exclus par le .gitignore).

## 2. Générer l'index puis le tester en local
    python vectorize_data.py          # crée le dossier faiss_index/
    streamlit run app.py              # teste dans le navigateur (utilise ton .env)

Commit le dossier `faiss_index/` (2 fichiers : index.faiss et index.pkl).
Si tu l'oublies, l'app le reconstruit au premier lancement (quelques centimes de Mistral).

## 3. Mettre sur GitHub
Crée un dépôt, puis pousse le dossier. Vérifie sur GitHub que `.env` n'y est PAS.

## 4. Déployer (choisis UNE plateforme)

### A. Streamlit Community Cloud (le plus simple)
1. Va sur share.streamlit.io et connecte ton compte GitHub.
2. "Create app" > choisis ton dépôt, la branche, et le fichier principal `app.py`.
3. "Advanced settings" > Secrets : colle
       MISTRAL_API_KEY = "ta_cle"
       MAX_QUESTIONS = "20"
       ACCESS_CODE = "un_code"      (optionnel)
4. Deploy. Tu obtiens un lien en .streamlit.app.

### B. Hugging Face Spaces
1. Crée un Space, SDK "Streamlit", visibilité Public.
2. Envoie les fichiers (git push ou upload), le fichier principal doit s'appeler app.py.
3. Settings > Variables and secrets > New secret : MISTRAL_API_KEY (et optionnellement
   MAX_QUESTIONS, ACCESS_CODE).

## 5. Protège ta clé Mistral (important)
Le lien est public : chaque question consomme TON crédit Mistral.
- Dans la console Mistral, fixe une limite de dépense mensuelle.
- Garde MAX_QUESTIONS bas (20 par session). Attention : un visiteur peut rafraîchir
  la page pour repartir à zéro, la vraie protection est la limite de dépense.
- Pour un test avec des personnes précises, utilise ACCESS_CODE.
- Si la clé a fuité un jour (GitHub, capture d'écran), régénère-la dans la console.

## Limites à connaître
- Les offres gratuites mettent l'app en veille après un temps sans visite : le premier
  visiteur attend quelques secondes. Vérifie les conditions actuelles sur leurs sites.
- Le modèle utilisé est mistral-medium-latest (comme ton script). Il coûte plus cher
  que mistral-small : tu peux le changer dans get_llm() de app.py.
- Le bandeau "projet de démonstration non officiel" est volontaire : le bot parle au
  nom de Lockall avec de vraies coordonnées. Garde-le visible.
