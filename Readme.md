# 🧭 WOOFI ML

**WOOFI ML** is a machine learning–based project designed to deliver personalized tourism destination recommendations in Indonesia. The system leverages user preferences and tourism content descriptions to generate intelligent, relevant, and engaging travel exploration experiences. By integrating user profiling and content analysis, WOOFI ML enhances tourism discovery through data-driven decision-making.

🌐 Website: https://www.woofi.web.id/

---

## 📊 Dataset

### 1️⃣ `Tempat_Wisata.csv`

- **Data Source**: Google Maps  
- **Collection Method**: Manual data scraping of popular tourist destinations from multiple regions in Indonesia, including South Sumatra, Lampung, West Java, and other provinces.

#### Features Collected:
- Tourist Destination Name  
- Rating  
- Category  
- Province  
- Google Maps Link  
- Description (English)  
- Images  

---

### 2️⃣ `test.json`

- **Data Source**: User dataset for recommendation system evaluation  
- **Description**: Simulated user data used to test and validate the recommendation pipeline. Each entry represents a user profile containing demographic attributes, interests, and historical search behavior.

---

## 🤖 Machine Learning & Technology

### 🔹 Models

- **Supervised Classification Model**  
  Predicts suitable tourist destinations based on user gender, age, and interests.

- **Content-Based Filtering**  
  Uses TF-IDF vectorization and cosine similarity to measure textual similarity between user preferences and destination descriptions.

### 🔹 Hybrid Recommendation Approach

The system combines:
- Supervised learning (user demographic-based prediction)
- Content-based filtering (text similarity-based recommendation)

This hybrid approach improves accuracy, personalization quality, and relevance of recommendations.

---

## 🌍 Key Features

- Personalized destination recommendations
- Support for new and returning users:
  - **New users**: recommendations based on profile (gender, age, interests)
  - **Returning users**: recommendations based on search history and demonstrated preferences
- Multilingual search integration using Google Translate API
- Search and filtering by province and category

---

## 🛠️ Installation

Install required dependencies using:

```bash
pip install -r requirements.txt
```

---

## 🎯 Project Goal

To develop an intelligent tourism recommendation system that delivers personalized, data-driven travel suggestions and enhances user exploration experiences across Indonesia.
