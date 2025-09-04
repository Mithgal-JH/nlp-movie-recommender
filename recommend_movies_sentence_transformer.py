from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the Sentence-Transformer model
# This model is designed to find semantic similarity between sentences
model = SentenceTransformer('all-MiniLM-L6-v2')

# The movie dataset
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

def recommend_movies_final_solution(target_movie_title):
    # Get the plots as a list of strings
    movie_plots = list(movies.values())
    
    # Encode all movie plots into a single NumPy array of vectors
    movie_vectors = model.encode(movie_plots)

    movie_titles = list(movies.keys())
    
    try:
        target_index = movie_titles.index(target_movie_title)
        target_vector = movie_vectors[target_index].reshape(1, -1)
    except ValueError:
        print(f"The movie '{target_movie_title}' is not in the list.")
        return

    # Calculate similarity between the target movie and all others
    sim_scores = cosine_similarity(target_vector, movie_vectors)
    
    # Get a list of (index, score) pairs
    sim_scores = list(enumerate(sim_scores[0]))
    
    # Sort the list by similarity score in descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Print the top recommendations (excluding the target movie itself)
    print(f"\nRecommendations for '{target_movie_title}':")
    for i, score in sim_scores[1:]:
        title = movie_titles[i]
        print(f"- {title} (Similarity: {score:.2f})")
# Run the program with a test movie
recommend_movies_final_solution("The Matrix")