import streamlit as st
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_courses(csv_file):
    courses = []
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            courses.append({
                "id": row["course_id"],
                "name": row["course_name"],
                "tags": row["tags"],
                "rating": float(row["rating"])
            })
    return courses


def vectorize_tags(courses):
    tag_list = [course["tags"] for course in courses]
    vectorizer = CountVectorizer()
    tag_matrix = vectorizer.fit_transform(tag_list)
    return tag_matrix, vectorizer


def search_courses(user_query, courses, tag_matrix, vectorizer):
    query_vector = vectorizer.transform([user_query])
    similarity_scores = cosine_similarity(query_vector, tag_matrix)
    scores = list(enumerate(similarity_scores[0]))
    scores.sort(key=lambda x: (x[1], courses[x[0]]["rating"]), reverse=True)

    results = []
    for index, score in scores:
        if score > 0:
            course = courses[index]
            results.append({
                "id": course["id"],
                "name": course["name"],
                "tags": course["tags"],
                "rating": course["rating"],
                "similarity": round(score, 2)
            })
    return results


st.title(" Course Recommendation System")
st.markdown("Type a topic to search for relevant courses!")

courses = load_courses("courses.csv")
tag_matrix, vectorizer = vectorize_tags(courses)

query = st.text_input("🔍 Search by topic (e.g., python, ai, sql)")

if query:
    results = search_courses(query, courses, tag_matrix, vectorizer)
    if results:
        st.subheader("Recommended Courses:")
        for course in results:
            st.markdown(f"**{course['name']}** ({course['id']})")
            st.markdown(f"-> Tags: `{course['tags']}`")
            st.markdown(f"-> Rating: {course['rating']}")
            st.markdown(f"-> Similarity: {course['similarity']}")
            st.markdown("---")
    else:
        st.warning("No matching courses found.")
