from sklearn.feature_extraction.text import CountVectorizer

courses = [
    "python data science",
    "web html css javascript",
    "java oop development"
]

vectorizer = CountVectorizer()

vector_matrix = vectorizer.fit_transform(courses)

print("Vocabulary:", vectorizer.get_feature_names_out())
print("Vector shape:", vector_matrix.shape)
print("Vector array:\n", vector_matrix.toarray())
