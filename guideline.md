FINAL PROJECT GUIDELINES
Course: Sentiment Analysis for Business Analytics
Credits: 2 Credits
Weight: 60% of the total course grade
Format: Group work (3-4 students/group) - 15-minute presentation + 5-minute Q&A.
I. Project Overview
As the most critical assessment of this course, the Final Project requires students to apply
the entire Sentiment Analysis pipeline (similar to the CRISP-DM lifecycle) from data
acquisition to decision-making.
Groups will act as an independent Data Analytics Agency. Your mission is to transform
a raw text dataset into a strategic business narrative (Data Storytelling) and provide
direct consulting to the Board of Directors (C-level executives) of a business based on
actual customer feedback.
II. Detailed Requirements
The project must encompass the following 4 core phases:
1. Data Acquisition & Preprocessing
• Acquisition: Automate data collection using web scraping tools (Playwright,
Selenium, BeautifulSoup) to extract reviews from platforms like Shopee,
TripAdvisor, App Store, YouTube, etc. (Minimum 2,000 rows of data). If using
an existing dataset (e.g., from Kaggle), it must contain noisy data and closely
reflect a real-world business scenario.
• Data Cleaning: Normalize text, remove emojis (if necessary), handle
abbreviations, internet slang, and missing values.
• EDA (Exploratory Data Analysis): Conduct an in-depth descriptive statistical
analysis of the dataset (e.g., average review length, star rating distribution,
common vocabulary before and after cleaning).
2. Modeling & Evaluation
• Modeling: Build a Sentiment Analysis pipeline. You must compare at least 2
methods (e.g., Traditional Machine Learning like Naive Bayes/SVM vs. Deep
Learning/Zero-shot Classification like PhoBERT/DeBERTa).
• Evaluation: The use of a Confusion Matrix is mandatory. You must clearly
explain the business implications of the trade-off between Precision and Recall
in the context of your chosen topic (e.g., the financial or brand impact of False
Positives vs. False Negatives).
3. Business Insights & Visualization
• Calculate the Net Sentiment Score (NSS) and plot sentiment trends over time
(Trend Analysis).
• Use Word Clouds (separating Positive and Negative groups) or Topic Modeling
to extract the specific reasons behind customer praise or complaints.
• Visualization: Create professional charts using Python libraries (Seaborn/Plotly)
or export your data to BI tools (Tableau/PowerBI) to build a Dashboard.
• Provide at least 3 Actionable Recommendations of strategic value for the
business based on your data findings.
4. Deliverables
1. Source Code: A smoothly running Jupyter Notebook (.ipynb) file with a clear,
logical structure and comprehensive explanatory comments.
2. Presentation Slides: Maximum 15 slides, heavily focused on Data Storytelling.
Absolutely do not include long blocks of code on the slides.
3. Comprehensive Project Report: A PDF document ranging from 10 to 15 pages
(excluding title page, table of contents, and appendices). The report should not
just be a brief summary but must thoroughly document your entire process:
• Introduction: The business problem and project objectives.
• Data Strategy: Detailed presentation of your EDA, cleaning steps, and
challenges encountered during text processing.
• Methodology: Justification for the chosen AI models and hyperparameter
tuning methods.
• Model Evaluation: Deep dive into the metrics (Confusion Matrix, Precision,
Recall, F1) and Error Analysis (why did the model fail on certain specific
sentences?).
• Business Insights: Presentation of visualizations (NSS, Word Clouds) and
logical deduction of your business recommendations.
III. Team Guidelines
To prevent one person from "carrying the team" and to practice real-world teamwork
skills, groups must clearly divide roles. One student can take on 1-2 roles:
• Data Engineer: Searches for/crawls data, cleans, and normalizes the text.
• AI Developer / Data Scientist: Writes the code to train NLP models, fine-tunes
parameters, and outputs technical evaluation reports.
• Business Analyst: Designs the Dashboard, calculates NSS, analyzes charts, and
writes business recommendations.
• Project Manager (Team Leader): Creates the timeline, integrates the code,
writes the Comprehensive Report (10-15 pages), designs the Slides, and manages
overall progress.
Timeline:
• Week 5: Finalize the topic and data source (Proposal).
• Week 6-7: Data collection, cleaning, and EDA.
• Week 8-9: Train Machine Learning/Deep Learning models and evaluate results
(Error Analysis).
• Week 10: Create visual charts, extract Business Insights, and finalize the 10-15
page Report and Slides.
• Week 11: Final Project Defense Presentation.
