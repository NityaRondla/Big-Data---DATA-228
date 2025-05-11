from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, split, lower, col
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn2

# Stopwords to exclude
stopwords = set([
    "and", "the", "a", "to", "in", "of", "is", "for", "on", "with", "at", "by",
    "an", "this", "that", "i", "you", "we", "from", "as", "it", "be", "are", "or"
])

# Spark session
spark = SparkSession.builder \
    .appName("Keyword Comparison Visualization") \
    .config("spark.driver.bindAddress", "127.0.0.1") \
    .getOrCreate()

# Load datasets
df_resume = spark.read.option("header", True).csv("Data/Big/BigResumeData.csv")
df_jd = spark.read.option("header", True).csv("Data/Big/BigJobDescriptions.csv")

# Extract keywords
resume_words = df_resume.select(explode(split(lower(col("Resume")), "\\W+")).alias("word"))
jd_words = df_jd.select(explode(split(lower(col("Job Description")), "\\W+")).alias("word"))

# Group and filter
resume_keywords = resume_words.groupBy("word").count().orderBy(col("count").desc())
jd_keywords = jd_words.groupBy("word").count().orderBy(col("count").desc())

# Filter stopwords and save top 20 in Parquet
resume_keywords.filter(~col("word").isin(stopwords)).limit(20) \
    .write.mode("overwrite").parquet("Outputs/Resume_Top_Keywords.parquet")

jd_keywords.filter(~col("word").isin(stopwords)).limit(20) \
    .write.mode("overwrite").parquet("Outputs/JD_Top_Keywords.parquet")

# Convert for plotting in Pandas:
df_resume_keywords = spark.read.parquet("Outputs/Resume_Top_Keywords.parquet").toPandas()
df_jd_keywords = spark.read.parquet("Outputs/JD_Top_Keywords.parquet").toPandas()

# BAR PLOT - Top Keywords
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.barplot(data=df_resume_keywords, x="count", y="word", palette="Blues_d")
plt.title("Top 20 Resume Keywords")
plt.xlabel("Frequency")

plt.subplot(1, 2, 2)
sns.barplot(data=df_jd_keywords, x="count", y="word", palette="Purples_d")
plt.title("Top 20 JD Keywords")
plt.xlabel("Frequency")

plt.tight_layout()
plt.savefig("Outputs/Top_Keywords_Barplot.png")
plt.show()

# VENN DIAGRAM - Overlap
resume_set = set(df_resume_keywords["word"])
jd_set = set(df_jd_keywords["word"])

plt.figure(figsize=(7, 7))
venn2([resume_set, jd_set], set_labels=('Resume', 'Job Description'))
plt.title("Keyword Overlap (Top 20)")
plt.savefig("Outputs/Keyword_Venn.png")
plt.show()

# HEATMAP of top overlapping terms
common = list(resume_set & jd_set)
resume_common = df_resume_keywords[df_resume_keywords["word"].isin(common)].set_index("word")
jd_common = df_jd_keywords[df_jd_keywords["word"].isin(common)].set_index("word")

heat_df = pd.concat([resume_common["count"], jd_common["count"]], axis=1)
heat_df.columns = ["Resume", "Job Description"]
heat_df.fillna(0, inplace=True)

plt.figure(figsize=(8, 6))
sns.heatmap(heat_df, annot=True, fmt="g", cmap="YlGnBu")
plt.title("Frequency Heatmap of Common Keywords")
plt.savefig("Outputs/Common_Keywords_Heatmap.png")
plt.show()

spark.stop()
print(" All visualizations completed and saved in 'Outputs/'")
