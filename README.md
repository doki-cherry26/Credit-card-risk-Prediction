<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Credit Card Prediction Project</title>
    <style>
        body {
            font-family: Arial, Helvetica, sans-serif;
            margin: 40px;
            line-height: 1.6;
            background-color: #f9f9f9;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        ul {
            margin-left: 20px;
        }
        .section {
            background: #ffffff;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 0 8px rgba(0,0,0,0.08);
        }
        img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 6px;
            margin-top: 10px;
        }
        code {
            background-color: #eef;
            padding: 3px 6px;
            border-radius: 4px;
        }
    </style>
</head>
<body>

<h1>Credit Card Prediction Project</h1>

<div class="section">
    <h2>Project Overview</h2>
    <p>
        This project focuses on predicting credit card outcomes using Machine Learning.
        The complete implementation is done using <strong>classes, objects, and functions</strong>,
        following a structured ML pipeline from data preprocessing to model deployment.
    </p>
</div>

<div class="section">
    <h2>Data Preprocessing</h2>
    <ul>
        <li>Checked for <strong>null values</strong> and removed them.</li>
        <li>Identified <strong>missing values</strong> and handled them using <strong>random sampling techniques</strong>.</li>
        <li>Detected <strong>outliers</strong> in the dataset and handled them appropriately.</li>
        <li>Removed unnecessary columns using <strong>feature selection techniques</strong>.</li>
        <li>Balanced the dataset to handle <strong>class imbalance</strong>.</li>
    </ul>
</div>

<div class="section">
    <h2>Models Used</h2>
    <p>The cleaned and processed data was trained on the following Machine Learning models:</p>
    <ul>
        <li>K-Nearest Neighbors (KNN)</li>
        <li>Naive Bayes</li>
        <li>Logistic Regression</li>
        <li>Decision Tree</li>
        <li>Random Forest</li>
        <li>AdaBoost</li>
    </ul>
</div>

<div class="section">
    <h2>Model Evaluation</h2>
    <p>Each model was evaluated using:</p>
    <ul>
        <li>Test Accuracy</li>
        <li>Classification Report</li>
        <li>Confusion Matrix</li>
        <li>ROC Curve</li>
        <li>AUC-ROC Curve</li>
    </ul>

    <h3>Confusion Matrix Example</h3>
    <img src="images/confusion_matrix.png" alt="Confusion Matrix">

    <h3>ROC Curve</h3>
    <img src="images/roc_curve.png" alt="ROC Curve">
</div>

<div class="section">
    <h2>Best Model Selection</h2>
    <p>
        Based on the comparison of <strong>ROC and AUC-ROC curves</strong>,
        <strong>Logistic Regression</strong> was selected as the best-performing model.
    </p>

    <img src="images/model_comparison.png" alt="Model Comparison">
</div>

<div class="section">
    <h2>Model Saving</h2>
    <p>
        The finalized Logistic Regression model was saved using a
        <code>pickle</code> file for later use in deployment.
    </p>
</div>

<div class="section">
    <h2>Deployment</h2>
    <p>
        The trained model was deployed as a web application.
        A user-friendly web page was created to interact with the model
        and make predictions based on user input.
    </p>

    <h3>Web Application Interface</h3>
    <img src="images/webpage.png" alt="Web Page Screenshot">
</div>

<div class="section">
    <h2>Technologies Used</h2>
    <ul>
        <li>Python</li>
        <li>Panda
