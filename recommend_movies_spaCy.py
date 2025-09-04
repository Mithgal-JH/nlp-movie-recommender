import spacy 


nlp= spacy.load("en_core_web_lg")

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


def recommend_movies(target_movie_title):
    
    if target_movie_title not in movies:
        print(f"Sorry, {target_movie_title} not exist.")
        return []
    
    
    target_polt = nlp(movies[target_movie_title])
    
    recommendations=[]
    
    for title,plot in movies.items():
        if title.lower() == target_movie_title.lower():
            continue
        
        other_plot = nlp(plot)
        
        similarity_score= target_polt.similarity(other_plot)
        
        recommendations.append((title,similarity_score))
    recommendations= sorted(recommendations,key=lambda x: x[1],reverse=True)
    print(f"\nRecommendations for '{target_movie_title}':")
    for title,score in recommendations:
        print(f"- {title} (Similarity: {score:.2f})")
            




recommend_movies("The Matrix")
