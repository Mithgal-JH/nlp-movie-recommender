# Movie Recommendation System – Learning Project

## Overview
This project is a personal learning experiment to understand **Natural Language Processing (NLP)**, **Machine Learning (ML)**, and **Deep Learning (DL)** concepts by building a movie recommendation system. The goal was to compare different methods of computing similarity between movie plots and observe how results change based on the chosen approach.

The project uses three sets of movie plot summaries:  
1. **First Data** – short summaries  
2. **Second Data** – detailed summaries  
3. **Third Data** – summaries without common words (stop words removed)

---

## Methods Used

### 1. **spaCy Embeddings Cosine**
- Uses `spaCy` pre-trained word embeddings (`en_core_web_lg`) to create vector representations of movie plots.  
- Cosine similarity is used to compare the semantic meaning of each plot.  
- **Pros:** Captures meaning beyond exact words; understands synonyms and context.  
- **Cons:** Sometimes gives similar similarity scores for very different plots, because embeddings are broad.

### 2. **TF-IDF Cosine (Lemmatized)**
- Preprocesses text with **lemmatization**, removes stop words, and converts plots into TF-IDF vectors.  
- Cosine similarity compares frequency-based significance of words.  
- **Pros:** Focuses on distinguishing words, highlights unique plot features.  
- **Cons:** Can underestimate similarity if two plots use different words for the same concept.

### 3. **Sentence Transformers (Optional, Future Extension)**
- Could be used for even better semantic similarity.  
- Produces context-aware embeddings at sentence or paragraph level.

### 4. **Hybrid Approaches**
- Combine methods (e.g., embeddings + TF-IDF) for more robust similarity scoring.  
- Weighted averages allow balancing semantic meaning with distinguishing keywords.

---

## Differences Between Methods
- **Short summaries vs. detailed summaries:**  
  Detailed summaries increase the importance of context, giving more meaningful similarity rankings in spaCy embeddings, but may dilute TF-IDF if unique words are rare.
  
- **Embeddings vs. TF-IDF:**  
  Embeddings capture meaning even if wording differs. TF-IDF captures exact word importance. This explains why similarity scores vary between methods.

- **Hybrid methods:**  
  Aim to balance strengths of both approaches. Without proper weighting, results may still favor one method (e.g., embeddings dominating or TF-IDF giving near-zero scores).

---

## Observations
- Movies with similar themes but different wording (e.g., *The Matrix* and *Inception*) score high in spaCy embeddings.  
- Movies sharing key terms (e.g., *Star Wars* and *A New Hope*) score higher in TF-IDF cosine.  
- Short summaries give less nuanced similarity than longer or cleaned summaries.  
- Hybrid methods can improve overall ranking but require proper weighting for best results.

---

## Recommendations Table Example for *The Matrix* (First Data)

| Method | Top 5 Recommendations | Similarity Scores |
|--------|----------------------|-----------------|
| spaCy Embeddings | Inception, Avatar, Lord of the Rings, A New Hope, The Dark Knight | 0.90, 0.90, 0.88, 0.85, 0.85 |
| TF-IDF Cosine | Star Wars, A New Hope, Lord of the Rings, Avatar, Inception | 0.06, 0.05, 0.05, 0.05, 0.00 |
| Hybrid (Embeddings + Cosine) | Inception, Avatar, Lord of the Rings, A New Hope, The Dark Knight | 0.90, 0.90, 0.88, 0.85, 0.85 |

> Note: The full note.txt contains similarity scores for all 11 movies, across all datasets and methods.

---

## Conclusion
This project helped to:  
- Explore different NLP techniques for text similarity.  
- Understand the trade-offs between semantic embeddings and word frequency methods.  
- Learn how preprocessing (lemmatization, stop-word removal) affects results.  
- Highlight why a “one-size-fits-all” approach doesn’t exist—each method provides unique insights.  

The final output is stored in **note.txt** showing similarity scores for each movie compared to others, across all datasets and methods.

---

## How to Run
1. Install required packages:
```bash
pip install spacy scikit-learn numpy
python -m spacy download en_core_web_lg
