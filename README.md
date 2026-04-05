# YouTube Engagement Analysis & Insights  
Final Project – Introduction to Data Science

---

## 📌 Project Overview

This project analyzes YouTube data from channels related to Data Science, AI, and Machine Learning to identify **key factors influencing video engagement**.

The goal is to extract **actionable insights** that can help content creators optimize their publishing strategy and improve audience interaction.

---

## 🎯 Business Objective

- Understand what drives higher engagement (views, likes, comments)
- Identify optimal publishing strategies (time, tags, content patterns)
- Support data-driven decision making for content creators

---

## 👥 Team Information

Group ID: 04

| Name               | Student ID |
|--------------------|-----------|
| Võ Duy Anh         | 21127221  |
| Nguyễn Mậu Gia Bảo | 21127583  |
| Lê Mỹ Khánh Quỳnh  | 21127681  |
| Vũ Minh Phát       | 21127739  |

---

## 📊 Data Collection

Data was collected using the YouTube API from multiple content channels.

Key features include:

- Engagement metrics: viewCount, likeCount, commentCount  
- Content metadata: title, description, tags  
- Publishing information: publishedAt, duration, caption  

---

## 🧹 Data Preparation

- Removed duplicate records  
- Handled missing values  
- Converted datetime and duration formats  
- Engineered additional features:
  - hour, day, month, year  
  - cleaned and formatted tags  
  - processed text fields (title, description)  

---

## 🔍 Exploratory Data Analysis

The analysis focused on answering key business questions:

- Does video length affect engagement?
- What types of tags are associated with high-performing videos?
- When is the best time to publish content?
- Do captions influence user interaction?
- What keywords are commonly used in successful video titles?
- How has user interest evolved over time?

---

## 💡 Key Insights

- **Video duration has minimal impact on engagement**, indicating that content quality or topic relevance is more important than length  

- **Tags related to AI, Machine Learning, and Data Science** appear more frequently in high-performing videos  

- Videos published between **14:00 – 18:00** tend to receive higher engagement  

- **Weekend uploads (especially Sunday)** show stronger performance compared to weekdays  

- User interest in data-related content has **increased over time**, though with fluctuations  

---

## 📈 Business Recommendations

Based on the analysis:

- Focus on high-interest topics such as AI, ML, and Data Science  
- Schedule video publishing during peak hours (afternoon & weekends)  
- Optimize titles using commonly occurring keywords in high-performing videos  
- Prioritize content relevance over video duration  

---

## 🤖 Supporting Analysis (Modeling)

Machine learning models were used to explore predictive patterns:

- Classified videos into trending levels (low / medium / high)  
- Applied feature engineering and model tuning (GridSearchCV)  

Note: Modeling is used as a **supporting tool**, not the main objective of the project.

---

## 🧠 Recommendation System

A simple recommendation system was developed to suggest videos based on user input text:

- Text preprocessing  
- TF-IDF vectorization  
- Clustering (KMeans, MiniBatchKMeans, BisectingKMeans)  

---

## 📓 Analysis Approach

The main analysis was conducted using **Jupyter Notebook**, including:

- Data cleaning and preprocessing  
- Exploratory data analysis (EDA)  
- Visualization and interpretation  
- Feature engineering  
- Model experimentation  

---

## 🖥️ Demo Application (Optional)

A lightweight Flask application is provided to demonstrate the recommendation system.

Run:

```bash
python src/app.py
```

Access:

```
http://127.0.0.1:5000/
```

Note: This is for demonstration only. The core value lies in data analysis and insights.

---

## 🛠️ Tools & Technologies

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib  
- TF-IDF  
- Jupyter Notebook  

---

## ⚠️ Limitations

- Limited dataset size  
- Memory constraints  
- Simplified scoring approach  
- Limited vocabulary coverage  

---

## 🚀 Future Improvements

- Expand dataset for better generalization  
- Improve recommendation model  
- Develop more robust scoring metrics  
- Explore deep learning approaches  

---

## 📚 References

See:

```
src/list_references.md
```

---

## 📌 Note

This project was developed as part of the  
**Introduction to Data Science course**, with a focus on applying data analysis techniques to real-world scenarios.
