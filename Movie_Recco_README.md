# 🎬 Movie Recommender System

A content-based movie recommendation system built with Streamlit that suggests movies based on similarity in genres, keywords, cast, crew, and overview. The system uses cosine similarity to find movies with similar content features.

## ✨ Features

- **Interactive Web Interface**: Beautiful and user-friendly Streamlit web application
- **Content-Based Filtering**: Recommends movies based on:
  - Movie genres
  - Keywords
  - Cast members
  - Crew (directors)
  - Movie overview/description
- **Visual Movie Cards**: Displays movie posters, ratings, release dates, and similarity scores
- **Detailed Recommendations**: Expandable sections with comprehensive movie information
- **Search Functionality**: Easy-to-use searchable dropdown to select movies
- **Similarity Scores**: Shows percentage match for each recommendation

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repository-url>
   cd Projects
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r movie_recco_requirements.txt
   ```

4. **Download the dataset**
   - Download the TMDB 5000 Movies dataset from [Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
   - You'll need two CSV files:
     - `tmdb_5000_movies.csv`
     - `tmdb_5000_credits.csv`
   - Update the file paths in `Movie_recco.py` (lines 19-20) to point to your dataset location

### Running the Application

1. **Start the Streamlit app**
   ```bash
   streamlit run Movie_recco.py
   ```

2. **Open in browser**
   - The app will automatically open in your default browser
   - Or manually navigate to `http://localhost:8501`

## 📖 How to Use

1. **Select a Movie**: Use the sidebar dropdown to search and select a movie you like
2. **View Movie Info**: See the selected movie's poster, rating, and release date in the sidebar
3. **Get Recommendations**: Click the "🔍 Get Recommendations" button
4. **Explore Results**: Browse through the top 5 recommended movies with:
   - Movie posters
   - Similarity scores
   - Release dates and ratings
   - Detailed information in expandable sections

## 🛠️ Technologies Used

- **Python 3**: Programming language
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning library
  - `CountVectorizer`: Text vectorization
  - `cosine_similarity`: Similarity calculation

## 📊 How It Works

1. **Data Processing**:
   - Merges movie and credits datasets
   - Extracts genres, keywords, cast, and crew information
   - Combines overview, genres, keywords, cast, and crew into tags

2. **Feature Extraction**:
   - Converts text tags into numerical vectors using CountVectorizer
   - Creates a 5000-feature vector representation for each movie

3. **Similarity Calculation**:
   - Computes cosine similarity between all movie vectors
   - Creates a similarity matrix

4. **Recommendation**:
   - For a selected movie, finds the 5 most similar movies
   - Displays recommendations with similarity scores

## 📁 Project Structure

```
Projects/
│
├── Movie_recco.py                    # Main application file
├── movie_recco_requirements.txt      # Python dependencies
├── README.md                         # Project documentation
└── data/                             # Dataset directory (create this)
    ├── tmdb_5000_movies.csv
    └── tmdb_5000_credits.csv
```

## 🔧 Configuration

Before running the application, update the CSV file paths in `Movie_recco.py`:

```python
movies = pd.read_csv('path/to/your/tmdb_5000_movies.csv')
credits = pd.read_csv('path/to/your/tmdb_5000_credits.csv')
```

## 📝 Requirements

See `movie_recco_requirements.txt` for the complete list of dependencies:
- numpy>=1.21.0
- pandas>=1.3.0
- scikit-learn>=1.0.0
- streamlit>=1.28.0

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available for educational purposes.

## 🙏 Acknowledgments

- Dataset: [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) from Kaggle
- Movie posters and metadata: The Movie Database (TMDB)

## 📧 Contact

For questions or suggestions, please open an issue in the repository.

---

**Enjoy discovering new movies! 🍿**

