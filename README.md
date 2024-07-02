# Social_Media_Impact_on_Mental_Health

## Depression Detection in Tweets

This Streamlit app analyzes tweets to detect signs of depression using machine learning algorithms. It provides insights into the sentiment analysis and classification of tweets as depressed or not depressed.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project aims to demonstrate the use of natural language processing and machine learning techniques to analyze tweets for signs of depression. It includes sentiment analysis using the TextBlob library and classification using various algorithms like K-Nearest Neighbors (KNN), Logistic Regression, and Random Forest.

## Features

- Upload a JSON file containing tweets for analysis.
- Perform sentiment analysis to determine the polarity of tweets.
- Classify tweets as depressed or not depressed based on sentiment analysis.
- Compare the accuracy of different machine learning algorithms.
- Display model performance metrics such as accuracy, confusion matrix, and classification report.
- Download datasets of depressed and not depressed tweets.
- Visualize counts, percentages, and comparisons of depressed and not depressed tweets.

## Dependencies

Ensure you have the following dependencies installed:

- streamlit
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- textblob

You can install them using `pip` with the provided `requirements.txt` file.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/PunuNGowda/Social_Media_Impact_on_Mental_Health.git
   cd Social_Media_Impact_on_Mental_Health

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

3. Usage
To run the Streamlit app locally:
   ```bash
   streamlit run streamlit_app.py
• Upload a JSON file containing tweets when prompted.
• Click "Submit" to analyze the uploaded tweets.
• Explore the results and visualizations provided by the app.

4. Deployment
The app can be deployed using Streamlit Cloud or any other hosting service that supports Streamlit apps. Ensure your requirements.txt and streamlit_app.py are properly configured.

5. Contributing
Contributions are welcome! Here's how you can contribute to this project:

• Fork the repository.
• Create a new branch (git checkout -b feature/your-feature).
• Make your changes.
• Commit your changes (git commit -am 'Add new feature').
• Push to the branch (git push origin feature/your-feature).
• Create a new Pull Request.

6. LICENSE

   (`[LICENSE]`)
