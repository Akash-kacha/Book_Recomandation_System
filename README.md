# Book Recommendation System
# **Content-Based Book Recommendation System**

## **Project Overview**
This project focuses on building a **personalized book recommendation system** using **Content-Based Filtering**. By analyzing book attributes such as **plot summary, author, and genre**, the system generates relevant book recommendations tailored to user preferences.

---

## **Business Problem**
**How can book attributes (e.g., genre, author, keywords) be leveraged to generate relevant recommendations?**

This project aims to help users **discover books similar to their interests** by using text similarity techniques.

---

## **Dataset Description**
The dataset contains **16,559 books** with the following attributes:
- `Book_title` - Name of the book
- `Author` - Name of the author
- `genres` - Book genre (e.g., Fiction, Science Fiction)
- `Plot_summary` - Description of the book
- `Wikipedia article ID`
- `Freebase ID`
- `Publication date`

---

## **Methodology**
### **1. Data Preprocessing**
- **Selected relevant features** (`Book_title`, `Author`, `genres`, `Plot_summary`)
- **Dropped missing values** in critical columns
- **Extracted genre names** from JSON-formatted data
- **Cleaned plot summaries**:
  - Removed special characters, numbers
  - Converted text to lowercase
  - Removed stopwords
  - Applied stemming using `PorterStemmer`

### **2. Feature Engineering**
- **Created a new "tags" column**:
  ```python
  books['tags'] = books['summary'] + books['Author_'] + books['genre_new_']
  ```
- **Vectorized text data** using `CountVectorizer`:
  ```python
  vectors = cv.fit_transform(new_df['tags']).toarray()
  ```
- **Computed Cosine Similarity**:
  ```python
  similarity = cosine_similarity(vectors)
  ```

### **3. Building the Recommendation System**
A function `recommend(book)` suggests **5 similar books** based on textual similarity.

```python
def recommend(book):
    book_index = new_df[new_df['Book_title'] == book].index[0]
    distances = similarity[book_index]
    book_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    for i in book_list:
        print('Book-title:', new_df.iloc[i[0]].Book_title)
        print('Author:', new_df.iloc[i[0]].Author)
        print('Genre:', new_df.iloc[i[0]].genre_new)
        print('Book-Summary:', new_df.iloc[i[0]].Plot_summary)
        print()
```

## **Future Improvements**
- **Use TF-IDF Vectorization** to improve accuracy
- **Implement a Hybrid System** (Content-Based + Collaborative Filtering)
- **Deploy as a Web Application** using Streamlit or Flask


## **Author**
**Akash Kacha**
- **Mathematician & Data Analyst**
- **LinkedIn:** [https://www.linkedin.com/in/akash-kacha-a98883228/](#)
- **GitHub:** [https://github.com/Akash-kacha](#)


⭐ **Feel free to star this repository if you find it useful!** ⭐

