
# Resume Matcher

**Resume Matcher** is a scalable, Spark-powered, Streamlit-based application designed to analyze how well a resume matches a given job description using natural language processing techniques. It supports large-scale data (63.5 GB) and provides annotated keyword comparisons, network visualizations, and similarity scoring—all through a clean, interactive interface.

---

## Project Overview

- Processes 63.5 GB of data (58 GB job descriptions, 5.5 GB resumes)
- Runs without HDFS or distributed cloud systems
- Real-time similarity scoring using NLP
- Interactive dashboards built with Plotly and NetworkX
- Professional UI built with Streamlit and custom CSS

---

## Features

- Resume and JD selection from dataset
- NLTK-based keyword extraction
- Annotated highlights of overlapping keywords
- Visual keyword network graph (NetworkX)
- Plotly treemaps and tables for term frequency
- Similarity scoring based on vector comparison

---

## Technologies Used

- Python 3.10+
- Apache Spark (for large-scale processing)
- Streamlit (web frontend)
- NLTK (text cleaning and stopword removal)
- Pandas (data manipulation)
- Plotly (visualizations)
- NetworkX (graph generation)

---

## Directory Structure

```
Resume-Matcher/
├── Assets/
├── Data/
│   ├── Big/
│   │   ├── BigJobDescriptions.csv
│   │   └── BigResumeData.csv
│   ├── Processed/
│   ├── Raw/
│   └── Demo/
├── Outputs/
│   ├── Common_Keywords_Heatmap.png
│   ├── Keyword_Venn.png
│   ├── Top_Keywords_Barplot.png
│   ├── Resume_Top_Keywords.parquet
│   └── JD_Top_Keywords.parquet
├── resume_matcher/
│   ├── dataextractor/
│   ├── scripts/
│   └── main.py
├── KeyTermExtractor.py
├── cleaner.py
├── process_bigdata_with_spark.py
├── streamlit_app.py
├── TextCleaner.py
├── app.log
├── .gitignore
└── README.md
```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/NityaRondla/Big-Data---DATA-228.git
cd Resume-Matcher
```

### 2. Create and Activate Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Place Datasets

Ensure that the following files exist inside `Data/Raw/`:

- `UpdatedResumeDataSet.csv`
- `JobDescriptions.csv`

### 5. Run the Application

```bash
streamlit run app.py
```

---

## Output and Visuals

The app produces:

- Annotated resume and JD with highlighted keywords
- NetworkX star graphs showing keyword relations
- Treemap keyword frequency charts using Plotly
- Tabular summaries of key terms
- Final similarity score for resume–job alignment

---

## Performance Notes

- Efficiently handles 63.5 GB of text data using Spark
- Matches resumes to JDs within seconds
- Runs completely on a local machine—no HDFS or cloud required

---

## License

This project is for educational and academic use only. All rights reserved.
