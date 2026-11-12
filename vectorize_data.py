"""
Script de vectorisation et de création de l'index FAISS pour Lockall.
Ce script charge les articles de support et les horaires,
les transforme en vecteurs via Mistral AI, et sauvegarde l'index localement.
"""

import json
import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS

# Chargement des variables d'environnement (clé API)
load_dotenv()

def create_and_save_vector_store():
    print("📂 Chargement des données Lockall depuis les fichiers JSON...")
    
    # 1. Charger les deux fichiers JSON
    with open("lockall_articles.json", "r", encoding="utf-8") as f:
        articles = json.load(f)
        
    with open("lockall_horaires.json", "r", encoding="utf-8") as f:
        horaires = json.load(f)
        
    # 2. Combiner toutes les données dans une seule liste
    all_data = articles + horaires
    documents = []

    # 3. Formater chaque élément en un Document LangChain
    for item in all_data:
        titre = item.get('titre', '')
        categorie = item.get('categorie', '')
        contenu = item.get('contenu', '')
        item_id = item.get('id', '')

        # On prépare le texte que l'IA va lire et vectoriser
        content = f"Titre: {titre}\nCatégorie: {categorie}\nContenu: {contenu}"

        # On garde l'ID et la catégorie en métadonnées pour le suivi
        metadata = {
            "id": item_id,
            "titre": titre,
            "categorie": categorie
        }

        # On ajoute le document tel quel (sans le découper !)
        documents.append(Document(page_content=content, metadata=metadata))

    print(f"✅ {len(all_data)} éléments préparés pour la vectorisation.")

    # 4. Vectorisation + indexation FAISS
    print("🧠 Envoi à Mistral pour vectorisation (mistral-embed)...")
    embeddings = MistralAIEmbeddings(model="mistral-embed")
    vector_store = FAISS.from_documents(documents, embeddings)

    # Vérification
    print(f"🔍 {vector_store.index.ntotal} vecteurs créés dans l'index.")

    # On écrase l'ancien dossier faiss_index (celui de Brest) avec le nouveau
    vector_store.save_local("faiss_index")
    print("🎉 Base vectorielle sauvegardée dans le dossier 'faiss_index' !")
    
    return vector_store

def search_lockall(vector_store, query, k=3):
    """Recherche les 'k' documents les plus pertinents pour la question."""
    return vector_store.similarity_search(query, k=k)

if __name__ == "__main__":
    # 1. On lance la création de l'index
    mon_vector_store = create_and_save_vector_store()
    
    # 2. On lance un test rapide de vérification
    print("\n🔎 Test de recherche : 'Je n'arrive pas à ouvrir mon box avec OneAccess'")
    test_results = search_lockall(mon_vector_store, "Je n'arrive pas à ouvrir mon box avec OneAccess", k=2)
    
    for i, doc in enumerate(test_results, 1):
        print(f"{i}. [{doc.metadata.get('id')}] {doc.metadata.get('titre')} (Catégorie: {doc.metadata.get('categorie')})")