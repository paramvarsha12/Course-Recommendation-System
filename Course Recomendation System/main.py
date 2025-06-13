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


def display_courses(courses):
    print("\nAvailable Courses:\n")
    for course in courses:
        print(f"{course['id']} - {course['name']}")
        print(f"Tags: {course['tags']}")
        print(f"Rating: {course['rating']}")
        print("-" * 40)


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

    print("\nTop Matching Courses:\n")
    found = False
    for index, score in scores:
        if score > 0:
            found = True
            course = courses[index]
            print(f"{course['id']} - {course['name']}")
            print(f"Tags: {course['tags']}")
            print(f"Rating: {course['rating']}")
            print(f"Similarity Score: {round(score, 2)}")
            print("-" * 40)

    if not found:
        print("No matching courses found.")

# --- MAIN PROGRAM ---
courses = load_courses("courses.csv")
display_courses(courses)
tag_matrix, vectorizer = vectorize_tags(courses)
user_input = input("\nSearch for a topic: ")
search_courses(user_input, courses, tag_matrix, vectorizer)
