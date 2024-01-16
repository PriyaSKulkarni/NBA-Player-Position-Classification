# NBA-Player-Position-Classification
CSE 5334: Data Mining


**Project Title: NBA Player Position Classification**

**Task Description:**
Implemented a classification model to categorize NBA players into five positions (SG, PG, SF, PF, C) based on per-game average performance in the 2020-2021 season. Utilized the MLPClassifier from scikit-learn, applying various methods and parameter tuning for optimal results.

**Implementation Steps:**

1. Loaded and processed the dataset using pandas.
2. Utilized the MLPClassifier for classification, experimenting with different parameter combinations.
3. Split the data into 75% for training and 25% for testing, calculating and printing out the accuracy on both sets.
4. Printed the confusion matrix for the multi-class classification problem.
5. Employed 10-fold stratified cross-validation to evaluate the model's accuracy across different folds.
6. Documented feature selection considerations, insights from basketball positions, and handling of NULL values.
7. Adjusted the model to consider the impact of limited minutes played on player statistics.
8. Explored custom distance functions for distance-based classification methods.
9. Achieved a balance between accuracy and realistic expectations given the nature of the dataset.

**Documentation:**
To enhance accuracy, I considered feature importance and relevance to basketball positions, avoiding redundant and less informative features. Domain knowledge of basketball positions influenced the model's understanding of player similarities. I addressed NULL values, replacing them with relevant defaults. Recognizing the impact of limited minutes played, I adjusted the model's expectations for bench players. Additionally, I experimented with custom distance functions and tuned parameters for optimal performance. This comprehensive approach contributed to achieving a robust classification model.
