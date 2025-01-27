Developed a high-accuracy machine learning model for digit classification on the MNIST dataset, demonstrating expertise in feature extraction, algorithm design, and model optimization. Key highlights include:

Feature Engineering:

Implemented Gaussian Blur to reduce noise and smooth image data, achieving optimal accuracy with a fine-tuned sigma value of 5.3.
Applied Principal Component Analysis (PCA) to reduce dimensionality, retaining 99.9% of the variance for faster runtime and improved performance.
Normalized image data to mitigate the effects of outliers and improve consistency.
Classifier Design:

Utilized a custom Weighted K-Nearest Neighbors (KNN) classifier with cosine distance to focus on shape similarity rather than spatial size differences.
Introduced weighted voting for neighbors, enhancing prediction accuracy by prioritizing closer matches.
Conducted comparative analysis with other distance metrics (Euclidean, Manhattan) to validate the superiority of cosine distance.
Technical Implementation:

Employed NumPy for numerical computations and scikit-learn for model evaluation.
Designed reusable Python functions for PCA, Gaussian Blur, and classifier training/testing.
Integrated modular design principles, facilitating future extensibility and reusability.
Performance Metrics:

Achieved 99.0% accuracy on noise-affected test data and 97.0% on mask-affected test data.
Leveraged detailed error analysis to refine feature extraction and classifier parameters, significantly improving results.
This project demonstrates proficiency in machine learning, feature extraction, and algorithm optimization for real-world applications.
