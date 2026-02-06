Fake News Detection Challenge 


 NLP Project Project OverviewThe goal of this project is to develop a robust Natural Language Processing (NLP) pipeline to classify news articles as either Real or Fake.
 
  On this This project we explored different approaches for text representation and classification like Word2Vec with XGBoost and TF-IDF with Naive Bayes.
  
  We start by loading the datasets (Training and Unlabeled Challenge data). Since the source files use a TSV (Tab-Separated Values) structure without headers, we manually map the columns into label and text to ensure data integrity.
  
  Before processing, we analyze the dataset to: 
  Check the Class Balance.
  ensuring a fair distribution between "Fake" and "Real" labels.
  Identify Missing Values.
  Detecting null entries or empty strings that could disrupt the pipeline.
  Confirming the scale of the training vs. testing sets.
 
  Text Preprocessing & Normalization  reduce noise and improve model focus, we apply a dedicated cleaning function:Case Folding: Converting all text to lowercase (already present in the source, but reinforced).
  
  Regex Filtering: Removing special characters and numbers to keep only alphabetical characters.
  
  Whitespace Stripping: Removing redundant spaces and trimming edges to clean the vocabulary.
  
Word2Vec + XGBoostVectorization: We use a Word2Vec model to create dense word embeddings, capturing semantic relationships between words. Sentences are converted into fixed-length vectors by averaging word embeddings.Classification: We use XGBoost (Extreme Gradient Boosting) with hardware acceleration (CUDA/GPU).

The model is optimized with a low learning rate (0.02) and 100 estimators to prevent overfitting while capturing complex patterns.

TF-IDF + Naive BayesVectorization: We use the TfidfVectorizer (limited to 5,000 features) to represent text based on word importance relative to the corpus.
Classification: A Multinomial Naive Bayes model is implemented. This is a highly efficient baseline for text classification that performs exceptionally well with sparse TF-IDF matrices. Evaluation & Strategy 

We use a Stratified Train-Test Split (80/20) to ensure that both sets reflect the same class proportions. Performance is measured primarily through Accuracy Score.

Pipeline Consistency: We ensure that the .transform() method (and not .fit_transform()) is used on the validation data to avoid Data Leakage.

Reproducibility: A fixed random_state=42 is used throughout the project to ensure results are consistent across different runs.

The final model generates predictions on the unlabeled challenge dataset. Results are exported in a standardized TSV format (label\ttext) without headers or indices, ready for evaluation.