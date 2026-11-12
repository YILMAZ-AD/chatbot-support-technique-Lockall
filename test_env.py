# test_env.py
try:
    import langchain
    import faiss
    import mistralai # (Le package Python pour l'API Mistral s'appelle souvent mistralai)
    print("Succès ! Toutes les bibliothèques sont bien installées et prêtes à être utilisées.")
    print(f"Version de LangChain : {langchain.__version__}")
    
except ImportError as e:
    print(f"Erreur d'importation : {e}")