import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


ratings = pd.read_csv("ratings.csv")  
movies = pd.read_csv("movies.csv")   


data = pd.merge(ratings, movies, on="movieId")


pivot_table = data.pivot_table(index="userId", columns="title", values="rating")


pivot_table.fillna(0, inplace=True)


user_similarity = cosine_similarity(pivot_table)
user_similarity_df = pd.DataFrame(user_similarity, index=pivot_table.index, columns=pivot_table.index)

def get_recommendations(user_id, num_recommendations=5):
    
    similar_users = user_similarity_df[user_id]

    
    similar_users_ratings = pivot_table.loc[similar_users.index].T

    
    weighted_ratings = similar_users_ratings.dot(similar_users) / similar_users.sum()

    
    recommendations = weighted_ratings.sort_values(ascending=False)
    return recommendations.head(num_recommendations)


recommended_movies = get_recommendations(user_id=1)
print("Recommended Movies:\n", recommended_movies)

def plot_star_ratings(ratings, movie_titles):
    
    plt.figure(figsize=(10, 6))
    
    
    def star_rating(rating):
        return '★' * int(rating) + '☆' * (5 - int(rating))
    
    
    stars = [star_rating(rating) for rating in ratings]
    
    
    plt.barh(movie_titles, ratings, color='skyblue')
    for i, star in enumerate(stars):
        plt.text(ratings[i] + 0.05, i, star, va='center', fontsize=12)

    plt.xlabel('Average Rating')
    plt.title('Movie Ratings with Stars')
    plt.xlim(0, 5)
    plt.grid(axis='x')
    plt.show()


average_ratings = data.groupby('title')['rating'].mean().sort_values()


plot_star_ratings(average_ratings.values, average_ratings.index)


plt.figure(figsize=(10, 6))
plt.barh(average_ratings.index, average_ratings.values, color='skyblue')
plt.xlabel('Average Rating')
plt.title('Average Movie Ratings')
plt.xlim(0, 5)  
plt.grid(axis='x')


plt.show()
