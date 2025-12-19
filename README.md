<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    
</head> 
<body>

    <h1>💳 Credit Card Fraud Prediction Project</h1>

    <div class="section">
        <h2>📌 Project Overview</h2>
        <p>
            This project focuses on predicting credit card fraud using various machine learning models.
            The complete pipeline is implemented using <strong>classes, objects, and functions</strong>.
            Multiple preprocessing techniques, model training, evaluation, and deployment steps are
            performed to identify the best-performing model.
        </p>
    </div>

    <div class="section">
        <h2>📂 Dataset Description</h2>
        <p>
            The dataset contains credit card transaction records with both legitimate and fraudulent transactions.
            The data is highly imbalanced and requires preprocessing before model training.
        </p>
    </div>

    <div class="section">
        <h2>🧹 Data Preprocessing Steps</h2>
        <ul>
            <li>Checked for <strong>null values</strong> and removed them.</li>
            <li>Identified <strong>missing values</strong> and handled them using <strong>random sampling techniques</strong>.</li>
            <li>Detected <strong>outliers</strong> and handled them appropriately.</li>
            <li>Removed unnecessary columns using <strong>feature selection</strong>.</li>
            <li>Balanced the dataset to handle class imbalance.</li>
        </ul>

        <h3>📊 Data Analysis Example</h3>
        <img src="images/data_analysis.png" alt="Data Analysis Image">
    </div>

    <div class="section">
        <h2>⚙️ Machine Learning Models Used</h2>
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
        <h2>📈 Model Training & Evaluation</h2>
        <p>The following evaluation metrics were used for each model:</p>
        <ul>
            <li>Test Accuracy</li>
            <li>Classification Report</li>
            <li>Confusion Matrix</li>
            <li>AUC-ROC Curve</li>
            <li>ROC Curve</li>
        </ul>

        <h3>📉 Confusion Matrix</h3>
        <img src="images/confusion_matrix.png" alt="Confusion Matrix">

        <h3>📈 ROC Curve</h3>
        <img src="images/roc_curve.png" alt="ROC Curve">
    </div>

    <div class="section">
        <h2>🏆 Best Model Selection</h2>
        <p>
            Based on the comparison of ROC curves and AUC scores,
            <strong>Logistic Regression</strong> was finalized as the best-performing model.
        </p>
    </div>

    <div class="section">
        <h2>💾 Model Saving</h2>
        <p>
            The trained Logistic Regression model and preprocessing objects were saved using
            the <code>pickle</code> file format for future predictions and deployment.
        </p>
    </div>

    <div class="section">
        <h2>🚀 Deployment</h2>
        <p>
            The project was deployed as a web application where users can input transaction data
            and receive fraud prediction results.
        </p>

        <h3>🌐 Web Application Interface</h3>
        <img src="images/web_app.png" alt="Web Application UI">
    </div>

    <div class="section">
        <h2>🛠️ Technologies Used</h2>
        <ul>
            <li>Python</li>
            <li>Pandas, NumPy</li>
            <li>Scikit-learn</li>
            <li>Matplotlib, Seaborn</li>
            <li>Pickle</li>
            <li>Flask / Web Framework</li>
            <li>HTML & CSS</li>
        </ul>
    </div>

    <div class="section">
        <h2>✅ Conclusion</h2>
        <p>
            This project  complete end-to-end machine learning pipeline,
            from data preprocessing to model deployment. Logistic Regression proved to be
            the most effective model for credit card risk prediction based on ROC and AUC analysis.
        </p>
    </div>

</body>
</html>
