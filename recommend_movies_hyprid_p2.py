import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


try:

    nlp = spacy.load("en_core_web_lg")
except OSError:
    print("The 'en_core_web_sm' model is not installed. Please run:")
    print("python -m spacy download en_core_web_lg")
    exit()


movies = {
    "The Matrix": "A computer programmer awakens to a simulated reality. He joins a group of individuals who resist sentient machines controlling humanity's perception.",
    "Star Wars": "A moisture farm inhabitant on Tatooine is swept into a galactic conflict after acquiring droids with secret plans. He meets an exiled Jedi and a smuggler, assisting a rebellion.",
    "Inception": "A corporate extractor enters dreams to steal information from subconscious minds. He accepts a final assignment to plant an idea instead of extracting one.",
    "A New Hope": "A farm inhabitant discovers a message from a princess in two droids. He allies with a Jedi master and smugglers to rescue her from a superweapon's crew.",
    "Avatar": "A paraplegic marine is dispatched to Pandora, a moon with a lush biosphere. He utilizes an engineered body to interact with the native species, but becomes conflicted between his mission and protecting their land.",
    "The Dark Knight": "A vigilante faces an anarchistic mastermind who terrorizes a city. He struggles to uphold order against a criminal whose only motivation is chaos.",
    "Lord of the Rings": "A hobbit inherits a powerful ring and embarks on a journey with companions from various races. Their goal is to travel to a volcano to destroy the artifact.",
    "Forrest Gump": "A low-IQ individual recounts his life, which intersects with major American events. He achieves extraordinary success while remaining devoted to his childhood companion.",
    "Titanic": "A struggling artist wins a ticket aboard a famous ship's inaugural journey. He forms a bond with an upper-class passenger, a relationship complicated by social divisions and an inevitable disaster.",
    "Jaws": "A chief of police, a marine scientist, and a seafarer hunt a massive, predatory shark terrorizing a coastal community.",
    "The Lion King": "A lion cub flees after his uncle murders his father and frames him. He returns later to challenge his relative and reclaim his rightful position.",
    "Jurassic Park": "Geneticists clone prehistoric creatures for a theme park. A security system failure leads to the escape of the animals, resulting in a survival scenario."
}

def get_processed_text():
    """Converts movie plots to a list of processed strings (lemmas)."""
    processed_plots = []
    for plot in movies.values():
        doc = nlp(plot)
        processed_plot = " ".join([token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct])
        processed_plots.append(processed_plot)
    return processed_plots

def recommend_movies_final_hybrid(target_movie_title):
    
    processed_plots = get_processed_text()
    movie_titles = list(movies.keys())


    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(processed_plots)
    

    target_index = movie_titles.index(target_movie_title)
    

    cosine_sim = cosine_similarity(tfidf_matrix[target_index:target_index+1], tfidf_matrix)
    

    sim_scores = list(enumerate(cosine_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    print(f"\nRecommendations for '{target_movie_title}':")
    for i, score in sim_scores[1:]:
        title = movie_titles[i]
        print(f"- {title} (Similarity: {score:.2f})")


recommend_movies_final_hybrid("The Matrix")