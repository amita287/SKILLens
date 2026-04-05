SKILLens — Resume Skill Matcher

SKILLens is a domain-independent NLP-based system that analyzes resumes, extracts skills, compares them with a job description, and provides match scores with improvement suggestions.


🚀 Features
- Upload PDF/DOCX resumes
- AI-based skill extraction (domain independent)
-  Match score using TF-IDF + skill overlap
-  Skill gap analysis (matched, missing, bonus)
- Visual insights (charts, word cloud, associations)
-  Optional AI suggestions for resume improvement
- Setup
pip install -r requirements.txt
python -m spacy download en_core_web_sm
streamlit run app.py

Open: http://localhost:8501

🧩 Tech Stack
Frontend: Streamlit
NLP: spaCy, NLTK
ML: TF-IDF, cosine similarity
Data Mining: Apriori (mlxtend)
Visualization: Plotly, Matplotlib
