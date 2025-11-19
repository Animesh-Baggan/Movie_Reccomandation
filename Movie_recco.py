# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import zipfile
with zipfile.ZipFile("tmdb_5000_credits.csv.zip", "r") as z:
    z.extractall()

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv("tmdb_5000_credits.csv")

# movies.head()

# movies.shape

# credits.head()

# credits.shape

# Merge movies and credits dataframes on the 'title' column
merged_df = movies.merge(credits, left_on='title', right_on='title')

# Display the head of the merged dataframe
# merged_df.head()

movies = merged_df
import ast

# Function to extract genre names from the string in 'genres'
def extract_genres(genre_str):
    try:
        genres_list = ast.literal_eval(genre_str)
        return ", ".join([g['name'] for g in genres_list])
    except:
        return ""

# Apply the function to the 'genres' column
movies['genres'] = movies['genres'].apply(extract_genres)

# Display the head of the dataframe to verify changes
# movies[['title', 'genres']].head()

# movies['genres']

# movies.head()

# Function to extract keyword names from the string in 'keywords'
def extract_keywords(keyword_str):
    try:
        keywords_list = ast.literal_eval(keyword_str)
        return ", ".join([k['name'] for k in keywords_list])
    except:
        return ""

# Apply the function to the 'keywords' column
movies['keywords'] = movies['keywords'].apply(extract_keywords)
# movies.head()

# Function to extract cast names from the string in 'cast'
def extract_cast(cast_str):
    try:
        cast_list = ast.literal_eval(cast_str)
        # Let's take top 3 cast members for brevity, can adjust as needed
        return ", ".join([c['name'] for c in cast_list[:3]]) 
    except:
        return ""

# Apply the function to the 'cast' column
movies['cast'] = movies['cast'].apply(extract_cast)
# movies.head()

# Function to extract crew names or specific roles from the 'crew' column
def extract_crew(crew_str):
    try:
        crew_list = ast.literal_eval(crew_str)
        # Option 1: Extract only directors
        directors = [c['name'] for c in crew_list if c.get('job') == 'Director']
        return ", ".join(directors)
    except:
        return ""

# Apply the function to the 'crew' column
movies['crew'] = movies['crew'].apply(extract_crew)
# movies.head()

def genres_to_list(genres_str):
    # Converts 'Action, Adventure, Fantasy, Science Fiction' to ['Action', 'Adventure', 'Fantasy', 'ScienceFiction']
    if not isinstance(genres_str, str):
        return []
    genre_list = [g.replace(" ", "") for g in genres_str.split(",")]
    return genre_list



movies['cast'] = movies['cast'].apply(genres_to_list)
movies['crew'] = movies['crew'].apply(genres_to_list)
movies['genres'] = movies['genres'].apply(genres_to_list)
movies['keywords'] = movies['keywords'].apply(genres_to_list)

# movies.head()

movies['overview'] = movies['overview'].apply(lambda x: x.split() if isinstance(x, str) else [])

# movies['overview'].head()

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# Keep poster_path and homepage columns for displaying images
columns_to_drop = ['overview','genres','keywords','cast','crew']
# Only drop columns that exist
columns_to_drop = [col for col in columns_to_drop if col in movies.columns]
new = movies.drop(columns=columns_to_drop)
#new.head()

new['tags'] = new['tags'].apply(lambda x: " ".join(x))
# new.head()

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')

vector = cv.fit_transform(new['tags']).toarray()

# vector.shape

from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(vector)

# similarity.shape

# new[new['title'] == 'The Lego Movie'].index[0]

def recommend(movie):
    index = new[new['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:6]:
        print(new.iloc[i[0]].title)

# recommend('Gandhi')

# recommend('Avatar')

import streamlit as st

# Page configuration (must be first Streamlit command)
st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache the similarity matrix and processed data for better performance
@st.cache_data
def get_processed_data():
    return new, similarity

# Function to get movie poster image URL
def get_poster_url(poster_path):
    """Convert poster_path to full TMDB image URL"""
    if pd.isna(poster_path) or not poster_path:
        return None
    # TMDB image base URL
    base_url = "https://image.tmdb.org/t/p/w500"
    if str(poster_path).startswith('/'):
        return f"{base_url}{poster_path}"
    return f"{base_url}/{poster_path}"

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .movie-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .recommendation-title {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 1rem;
    }
    .similarity-score {
        color: #27ae60;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🎬 Movie Recommender System</h1>', unsafe_allow_html=True)
st.markdown("---")

# Get cached data
movies_df, similarity_matrix = get_processed_data()

def recommend_movies_streamlit(movie, movies_df, similarity_matrix):
    try:
        index = movies_df[movies_df['title'] == movie].index[0]
        distances = sorted(list(enumerate(similarity_matrix[index])), reverse=True, key=lambda x: x[1])
        recommended_movies = []
        for i in distances[1:6]:
            movie_idx = i[0]
            similarity_score = i[1]
            movie_data = movies_df.iloc[movie_idx]
            recommended_movies.append({
                'title': movie_data.title,
                'similarity': similarity_score,
                'data': movie_data
            })
        return recommended_movies
    except IndexError:
        st.error(f"Movie '{movie}' not found in the dataset.")
        return []

# Sidebar for movie selection
with st.sidebar:
    st.header("🎯 Select Your Movie")
    st.markdown("---")
    
    # Searchable selectbox
    movie_list = sorted(movies_df['title'].values)
    selected_movie = st.selectbox(
        'Choose a movie you like:',
        movie_list,
        help="Type to search for a movie"
    )
    
    # Show selected movie info if available
    if selected_movie:
        movie_info = movies_df[movies_df['title'] == selected_movie].iloc[0]
        st.markdown("### Selected Movie Info")
        
        # Display movie poster if available
        poster_url = None
        if 'poster_path' in movie_info and pd.notna(movie_info['poster_path']):
            poster_url = get_poster_url(movie_info['poster_path'])
        
        if poster_url:
            st.image(poster_url, width=200, caption=selected_movie)
        
        st.write(f"**Title:** {selected_movie}")
        if 'release_date' in movie_info and pd.notna(movie_info['release_date']):
            st.write(f"**Release Date:** {movie_info['release_date']}")
        if 'vote_average' in movie_info and pd.notna(movie_info['vote_average']):
            st.write(f"**Rating:** ⭐ {movie_info['vote_average']:.1f}/10")
        if 'homepage' in movie_info and pd.notna(movie_info['homepage']):
            st.write(f"**Homepage:** {movie_info['homepage']}")
        st.markdown("---")

# Main content area
st.markdown("### 🎯 How to Use")
st.info("👈 Select a movie from the sidebar, then click the button below to get personalized recommendations!")

col1, col2 = st.columns([1, 3])

with col1:
    if st.button('🔍 Get Recommendations', type="primary", use_container_width=True):
        st.session_state['show_recommendations'] = True
        st.session_state['selected_movie'] = selected_movie

with col2:
    st.write("")  # Spacing

# Display recommendations
if st.session_state.get('show_recommendations', False) and st.session_state.get('selected_movie'):
    recommendations = recommend_movies_streamlit(st.session_state['selected_movie'], movies_df, similarity_matrix)
    
    if recommendations:
        st.markdown("---")
        st.markdown(f'<div class="recommendation-title">🎯 Recommendations for "{st.session_state["selected_movie"]}"</div>', unsafe_allow_html=True)
        st.write("")
        
        # Display recommendations in a grid
        cols = st.columns(5)
        
        for idx, rec in enumerate(recommendations):
            with cols[idx % 5]:
                with st.container():
                    st.markdown(f'<div class="movie-card">', unsafe_allow_html=True)
                    
                    # Display movie poster if available
                    movie_data = rec['data']
                    poster_url = None
                    if 'poster_path' in movie_data and pd.notna(movie_data['poster_path']):
                        poster_url = get_poster_url(movie_data['poster_path'])
                    
                    if poster_url:
                        st.image(poster_url, width=150, caption=rec['title'])
                    
                    # Movie title
                    st.markdown(f"### {idx + 1}. {rec['title']}")
                    
                    # Similarity score
                    similarity_percent = rec['similarity'] * 100
                    st.markdown(f'<p class="similarity-score">Similarity: {similarity_percent:.1f}%</p>', unsafe_allow_html=True)
                    
                    # Additional movie info
                    if 'release_date' in movie_data and pd.notna(movie_data['release_date']):
                        st.caption(f"📅 {movie_data['release_date']}")
                    if 'vote_average' in movie_data and pd.notna(movie_data['vote_average']):
                        st.caption(f"⭐ {movie_data['vote_average']:.1f}/10")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
        
        # Detailed view in expandable sections
        st.markdown("---")
        st.markdown("### 📋 Detailed Recommendations")
        
        for idx, rec in enumerate(recommendations, 1):
            with st.expander(f"{idx}. {rec['title']} (Similarity: {rec['similarity']*100:.1f}%)"):
                movie_data = rec['data']
                
                # Display poster in expandable section
                poster_url = None
                if 'poster_path' in movie_data and pd.notna(movie_data['poster_path']):
                    poster_url = get_poster_url(movie_data['poster_path'])
                
                col_poster, col_info1, col_info2 = st.columns([1, 2, 2])
                
                with col_poster:
                    if poster_url:
                        st.image(poster_url, width=200, caption=rec['title'])
                    else:
                        st.write("No poster available")
                
                with col_info1:
                    st.write("**Movie Details:**")
                    if 'release_date' in movie_data and pd.notna(movie_data['release_date']):
                        st.write(f"📅 Release Date: {movie_data['release_date']}")
                    if 'vote_average' in movie_data and pd.notna(movie_data['vote_average']):
                        st.write(f"⭐ Rating: {movie_data['vote_average']:.1f}/10")
                    if 'vote_count' in movie_data and pd.notna(movie_data['vote_count']):
                        st.write(f"👥 Votes: {int(movie_data['vote_count']):,}")
                    if 'runtime' in movie_data and pd.notna(movie_data['runtime']):
                        st.write(f"⏱️ Runtime: {int(movie_data['runtime'])} min")
                    if 'homepage' in movie_data and pd.notna(movie_data['homepage']):
                        st.write(f"🌐 [Homepage]({movie_data['homepage']})")
                
                with col_info2:
                    st.write("**Similarity Metrics:**")
                    similarity_percent = rec['similarity'] * 100
                    st.metric("Match Score", f"{similarity_percent:.2f}%")
                    
                    # Progress bar for similarity
                    st.progress(rec['similarity'])
                
                # Show tags if available (first 100 chars)
                if 'tags' in movie_data and pd.notna(movie_data['tags']):
                    tags_preview = movie_data['tags'][:200] + "..." if len(str(movie_data['tags'])) > 200 else movie_data['tags']
                    st.write("**Tags:**")
                    st.caption(tags_preview)
