import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
# -------------------------------------------------------------------------------------------------------------------
def init_model():
    # le modèle
     MODEL_NAME = "chemin/vers/le/model/Llama-3.2-1B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else {"": "cpu"},
        trust_remote_code=True
    )

    print(f"Modèle chargé sur le device : {device}")
    return tokenizer, model, device

# ----------------------------------------------------------------------------------------------------

def call_LLM(prompt, tokenizer, model, device):
    """Génère une réponse à partir d'un prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True, top_p=0.9, temperature=0.2)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -------------------------------------------------------------------------------------------------------
def detect_intent(diag, tokenizer, model, device):
    # mappage des mots-clés pour intention
    keywords_mapping = {
        ("organisation", "structure", "hiérarchie"): "1. Organisation de la société médiévale",
        ("seigneur", "seigneurs", "noble", "vassal", "chevalerie"): "2. Mode de vie des seigneurs et chevaliers",
        ("paysan", "paysans", "agriculture", "fermier", "terres"): "3. Vie des paysans et artisans",
        ("château", "forteresse", "remparts", "murailles", "tour de guet"): "4. Fonctionnement des châteaux forts",
        ("religion", "catholicisme", "église", "clergé", "hérésie", "papauté", "pape", "foi", "croyance"): "5. Religions et croyances",
        ("éducation", "scolaire", "enseignement", "école", "université", "apprentissage"): "6. Apprentissage et éducation",
        ("commerce", "marché", "échanges", "monnaie", "marchands"): "7. Commerce et échanges économiques",
        ("maladie", "médecine", "peste", "épidémie", "remède", "guérison", "soins", "médecins", "santé"): "8. Maladies et médecine médiévale",
        ("femme", "femmes", "noble", "paysanne", "mère", "épouse", "féminité", "genre"): "9. Rôles et place des femmes",
        ("tournoi", "joute", "chevaliers", "combat", "jeu", "bataille", "lance"): "10. Tournois et divertissements",
        ("guerre", "armée", "bataille", "stratégie", "militaire", "conquête", "siège"): "11. Guerre et stratégies militaires",
        ("justice", "tribunal", "loi", "châtiment", "serment", "jurisprudence", "droit", "féodal"): "12. Justice et droit féodal",
        ("ville", "village", "agglomération", "cité", "banlieue", "faubourg", "place", "ruelles"): "14. Villes et villages médiévaux",
        ("cathédrale", "église", "abbaye", "chapelle", "architecture religieuse", "sanctuaire", "cloître"): "15. Construction des cathédrales et églises",
        ("nourriture", "cuisine", "aliment", "repas", "viande", "pain", "soupe", "ragoût", "produits", "ingrédient"): "16. Nourriture et cuisine médiévale",
        ("moine", "clergé", "abbaye", "moines", "religieux", "monastère", "vie religieuse"): "17. Vie quotidienne des moines et moniales",
        ("système féodal", "feudalisme", "seigneurie", "serf", "vassal", "fief", "relation féodale"): "18. Fonctionnement du système féodal",
        ("pirate", "bandit", "brigand", "corsaire", "vol", "pillages", "routes dangereuses", "mer"): "19. Pirates et bandits",
        ("superstition", "magie", "sorcellerie", "astrologie", "astrologue", "magicien", "rituels", "divination", "sorts"): "20. Superstitions et magie",
        ("art", "musique", "peinture", "sculpture", "littérature", "poème", "chant", "instrument", "création artistique"): "21. Art et musique médiévale",
        ("épidémie", "peste", "famine", "maladie", "réaction", "disette", "déclin", "mortalité"): "22. Épidémies et famines",
        ("mariage", "rituel", "fiancée", "époux", "noble", "dot", "alliances", "cérémonie", "mariés", "coutumes"): "23. Coutumes et mariages",
        ("autre", "divers", "questions générales", "non spécifié", "inconnu"): "25. Autre intention"
    }

    # convertir le texte en minuscules pour une correspondance insensible à la casse
    diag_lower = diag.lower()

    # recherche du mot-clé correspondant
    for keywords, intent in keywords_mapping.items():
        if any(keyword in diag_lower for keyword in keywords):
            result = intent
            break
    else:
        result = "25. Autre intention"

    print(f"Intent trouvé : {result}")

    # le numéro de l'intention
    intent_number = result.split(".")[0]
    print("Numéro d'intention extrait :", intent_number)

    return result


# ------------------------------------------------------------------------------------------------
def extract_slot(slot, intent, tokenizer, model, device):
    """extraction de la valeur des slots à partir de l'intention détectée """
    if intent in database:
        if slot in database[intent]:
            return database[intent][slot]
        else:
            return f"Le slot '{slot}' n'existe pas pour l'intention '{intent}'."
    else:
        return f"L'intention '{intent}' n'existe pas dans la base de données."


# ------------------------------------------------------------------------------------------------
def evaluate_similarity(response, expected_response):
    """similarité cosinus pour calculer la similarité entre la sortie du modèle et les réponses attendues"""
    vectorizer = CountVectorizer().fit_transform([response, expected_response])
    vectors = vectorizer.toarray()
    similarity = cosine_similarity(vectors)
    return similarity[0][1]


# ------------------------------------------------------------------------------------------------
def generate_full_narrative_response(intent_name, slots, tokenizer, model, device):
    """
    le LLM génère une réponse narrative complète en utilisant toutes les valeurs des slots pour l'intention donnée
    """
    prompt = f"Donne un aperçu complet sur {intent_name}. "

    print("Slots disponibles : ", slots)

    for slot in slots:
        description = extract_slot(slot, intent_name, tokenizer, model, device)
        reponses_attendues = description
        #print("réponses attendues : ", reponses_attendues)
        prompt += f"{slot}: {description}. "

    response = call_LLM(prompt, tokenizer, model, device)

    #stocker la réponse générée par le LLM dans la variable llm_response
    llm_response = response
    #print("réponses du modèle : ", llm_response)

    # évaluer la similarité entre la réponse du modèle et la réponse attendue
    similarity_score = evaluate_similarity(llm_response, reponses_attendues)
    print(f"Similarité entre la réponse du modèle et la réponse attendue : {similarity_score:.2f}")

    # retourner la réponse complète générée
    return llm_response

# ------------------------------------------------------------------------------------------------
def save_conversation_to_file(user_input, model_response, filename="/content/MyDrive/MyDrive/Agent/Ressources/conversation_log.txt"):
    """enregistrer les derniers chats pour analyse"""
    with open(filename, 'a') as f:
        f.write(f"Utilisateur: {user_input}\n")
        f.write(f"Réponse du modèle: {model_response}\n")
        f.write("-" * 80 + "\n")


# -----------------------------------------------------------------------------------------
def main():
    intent_to_slots_and_keywords = {
        1: ("Organisation de la société médiévale", ["Seigneur", "Chevalier", "Paysan", "Moine", "Artisan"]),
        2: ("Mode de vie des seigneurs et chevaliers", ["Alimentation", "Logement", "Activités quotidiennes"]),
        3: ("Vie des paysans et artisans", ["Travail", "Outils", "Conditions de vie"]),
        4: ("Fonctionnement des châteaux forts", ["Défense", "Architecture", "Pièce principale"]),
        5: ("Religions et croyances", ["Croyances", "Religieux", "Pratiques"]),
        6: ("Apprentissage et éducation", ["Écoles monastiques", "Universités", "Maîtres et élèves"]),
        7: ("Commerce et échanges économiques", ["Monnaies", "Routes commerciales", "Marchés médiévaux"]),
        8: ("Maladies et médecine médiévale", ["Types de maladies", "Remèdes", "Médecins et guérisseurs"]),
        9: ("Rôles et place des femmes", ["Noblesse", "Paysannerie", "Monastères", "Métiers"]),
        10: ("Tournois et divertissements", ["Joutes", "Banquets", "Jeux populaires"]),
        11: ("Guerre et stratégies militaires", ["Armures", "Armes", "Techniques de combat"]),
        12: ("Justice et droit féodal", ["Tribunaux", "Châtiments", "Serments"]),
        13: ("Religion et hérésie", ["Croisades", "Inquisition", "Saints et pèlerinages"]),
        14: ("Villes et villages médiévaux", ["Urbanisme", "Rôles des corporations", "Artisanat"]),
        15: ("Construction des cathédrales et églises", ["Architecture", "Art gothique", "Métiers impliqués"]),
        16: ("Nourriture et cuisine médiévale", ["Plats typiques", "Ingrédients utilisés", "Techniques de cuisson"]),
        17: ("Vie quotidienne des moines et moniales", ["Règles monastiques", "Travail", "Prière"]),
        18: ("Fonctionnement du système féodal", ["Suzeraineté", "Hommage", "Fiefs"]),
        19: ("Pirates et bandits au Moyen Âge", ["Routes dangereuses", "Répression des brigands"]),
        20: ("Superstitions et magie", ["Sorcellerie", "Astrologie", "Amulettes"]),
        21: ("Art et musique médiévale", ["Instruments", "Peinture", "Littérature"]),
        22: ("Épidémies et famines", ["Peste noire", "Disettes", "Réactions des populations"]),
        23: ("Coutumes et mariages", ["Mariage nobles vs paysans", "Traditions", "Dotes"]),
        24: ("Sortir de la conversation", []),
    }

    tokenizer, model, device = init_model()

    print("Détection d'intention et extraction de slots avec le modèle.")
    print("Tapez 'quit' ou 'exit' pour terminer.\n")

    while True:
        diag = input("Posez votre question (ou tapez 'quit' pour arrêter) : ")

        if diag.lower() in ["quit", "exit"]:
            print("Fin du programme.")
            break

        intent = detect_intent(diag, tokenizer, model, device)
        intent_num = int(intent.split(".")[0])
        intent_name, slots = intent_to_slots_and_keywords.get(intent_num, ("", []))

        response = generate_full_narrative_response(intent_name, slots, tokenizer, model, device)
        print("Réponse générée : ", response)

        # enregistrer la conversation
        save_conversation_to_file(diag, response)

        print("-" * 80)
if __name__ == "__main__":
    main()
