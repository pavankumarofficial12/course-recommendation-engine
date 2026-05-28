import logging
import os
from typing import List, Dict
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# ========================= CONFIGURATION =========================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database Configuration from .env
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    logger.critical("DATABASE_URL is not set in .env file")
    raise Exception("DATABASE_URL environment variable is required")

# Initialize FastAPI
app = FastAPI(
    title="AI Hybrid Course Recommendation System",
    description="Personalized course recommendations using Hybrid Filtering",
    version="1.0.0"
)

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # Change this in production to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database Engine
engine = create_engine(DATABASE_URL, pool_recycle=3600)

# ========================= DATA FETCHING FUNCTIONS =========================
def get_user():
    return pd.read_sql_query("SELECT * FROM user", engine)

def get_course():
    query = """
    SELECT 
        id AS course_id,
        course_title AS courseTitle,
        COALESCE(course_description, '') AS courseDescription,
        course_type AS courseType,
        CAST(level AS UNSIGNED) AS level,
        course_fee AS courseFee,
        course_image AS courseImage,
        course_language AS courseLanguage,
        course_rating AS courseRating,
        user_id AS creator_id
    FROM course
    """
    return pd.read_sql_query(query, engine)

def get_enrollment():
    return pd.read_sql_query("SELECT * FROM enrollment", engine)

def get_search():
    return pd.read_sql_query("SELECT * FROM search", engine)

def get_personal_details():
    return pd.read_sql_query("SELECT * FROM personal_details", engine)

def fetch_data():
    return get_user(), get_course(), get_enrollment(), get_search(), get_personal_details()

# ========================= HELPER FUNCTIONS =========================
def get_user_language(user_id: int):
    user_df = get_user()
    user = user_df[user_df['id'] == user_id]
    if user.empty:
        raise ValueError(f"User with id {user_id} not found")
    return user['user_language'].values[0]

def get_creator_name(creator_id: int):
    user_df = get_user()
    user = user_df[user_df['id'] == creator_id]
    if user.empty:
        return "Unknown"
    return user['name'].values[0]

# ========================= CONTENT-BASED RECOMMENDATIONS =========================
def content_based_recommendations(user_id: int, top_n: int = 10):
    user_df, course_df, _, search_df, personal_details_df = fetch_data()

    user_language = get_user_language(user_id)
    user = user_df[user_df['id'] == user_id]
    if user.empty:
        raise ValueError("User not found")

    # Build user profile
    user_skills = personal_details_df[personal_details_df['user_id'] == user_id]['skills'].values
    user_skills = user_skills[0] if len(user_skills) > 0 else ""
    user_searches = search_df[search_df['user_id'] == user_id]['course_content'].tolist()
    user_profile = f"{user['user_description'].values[0]} {user_skills} {' '.join(user_searches)}"

    # TF-IDF
    course_df['courseDescription'] = course_df['courseDescription'].fillna('')
    tfidf = TfidfVectorizer(stop_words='english')
    course_tfidf_matrix = tfidf.fit_transform(course_df['courseDescription'])
    user_profile_tfidf = tfidf.transform([user_profile])

    cosine_sim = cosine_similarity(user_profile_tfidf, course_tfidf_matrix).flatten()
    course_df = course_df.copy()
    course_df['similarity'] = cosine_sim

    # Filter by language
    filtered_courses = course_df[course_df['courseLanguage'] == user_language]
    if filtered_courses.empty:
        logger.info("No courses in user language. Using all courses.")
        filtered_courses = course_df.copy()

    # Sort and get top courses
    recommended = filtered_courses.sort_values(by=['courseRating', 'similarity'], ascending=[False, False]).head(top_n)
    recommended['creatorName'] = recommended['creator_id'].apply(get_creator_name)

    # Format rating
    recommended['courseRating'] = recommended['courseRating'].apply(lambda x: int(x) if x in [1,2,3,4,5] else 0)

    return recommended[['course_id', 'courseTitle', 'courseDescription', 'courseType', 'level',
                        'courseFee', 'courseImage', 'courseRating', 'creatorName']]

# ========================= COLLABORATIVE FILTERING =========================
def collaborative_filtering_recommendations(user_id: int, top_n: int = 10):
    _, course_df, enrollment_df, _, _ = fetch_data()

    user_language = get_user_language(user_id)

    user_enrollments = enrollment_df[enrollment_df['user_id'] == user_id]
    if user_enrollments.empty:
        return pd.DataFrame()

    similar_users = enrollment_df[enrollment_df['course_id'].isin(user_enrollments['course_id'])]
    similar_users = similar_users[similar_users['user_id'] != user_id]

    recommended_course_ids = similar_users['course_id'].value_counts().index
    recommended_courses = course_df[course_df['course_id'].isin(recommended_course_ids)]

    filtered_courses = recommended_courses[recommended_courses['courseLanguage'] == user_language]
    if filtered_courses.empty:
        filtered_courses = recommended_courses.copy()

    filtered_courses = filtered_courses.sort_values(by='courseRating', ascending=False).head(top_n)
    filtered_courses['creatorName'] = filtered_courses['creator_id'].apply(get_creator_name)
    filtered_courses['courseRating'] = filtered_courses['courseRating'].apply(lambda x: int(x) if x in [1,2,3,4,5] else 0)

    return filtered_courses[['course_id', 'courseTitle', 'courseDescription', 'courseType', 'level',
                             'courseFee', 'courseImage', 'courseRating', 'creatorName']]

# ========================= HYBRID RECOMMENDATIONS =========================
def hybrid_recommendations(user_id: int, top_n: int = 5):
    logger.info(f"Generating hybrid recommendations for user_id: {user_id}")
    
    content_recs = content_based_recommendations(user_id, top_n * 2)
    collab_recs = collaborative_filtering_recommendations(user_id, top_n * 2)

    all_recs = pd.concat([content_recs, collab_recs]).drop_duplicates(subset=['course_id']).reset_index(drop=True)
    final_recs = all_recs.sort_values(by='courseRating', ascending=False).head(top_n)

    return final_recs

# ========================= REQUEST MODEL =========================
class RecommendationRequest(BaseModel):
    user_id: int

# ========================= API ENDPOINT =========================
@app.post('/course')
async def get_recommendations(request: RecommendationRequest):
    try:
        recommendations = hybrid_recommendations(request.user_id, top_n=5)
        recommendations_json = recommendations.to_dict(orient='records')

        return {
            "status": "success",
            "user_id": request.user_id,
            "recommendations": recommendations_json,
            "message": "Recommendations generated successfully"
        }
    except ValueError as ve:
        logger.warning(f"Validation error: {ve}")
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error. Please try again later.")

# Health Check
@app.get('/health')
async def health_check():
    return {"status": "healthy", "service": "Course Recommendation System"}

# ========================= RUN SERVER =========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
