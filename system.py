import numpy as np
import math
from utils import save_model, load_model, NullModel


def image_to_reduced_feature(images, split='test', sigma=5.3, variance=0.999):
    """
    Reduces the dimensionality of images using PCA (Principal Component Analysis).

    Parameters:
    - images: np.ndarray, the set of input images to process.
    - split: str, 'test' or 'train', to determine if PCA components should be loaded or calculated.
    - sigma: float, standard deviation for the gaussian blur applied to the images.
    - variance: float, threshold for cumulative explained variance to select PCA components.

    Returns:
    - pca: np.ndarray, reduced-dimensional feature representation of the images.
    """
    # Normalising the images by subtracting the mean
    normalised_images = images - np.mean(images, axis=0)

    # Applying gaussian blur to reduce noise and smooth the images out
    normalised_images = apply_gaussian_blur(normalised_images, sigma=sigma)

    if split == 'test':  # so the images are from one of the test in the dataset

        # we are loading the principal components that we calculated when we were training the model 
        principal_components = load_model('principal_components')
    else:  # if the images are from the train dataset

        # Calculating the covariance matrix of the normalized images
        covariance_of_normalised_images = np.cov(normalised_images, rowvar=False)

        # Getting the eigenvalues and eigenvectors 
        eigenvalues, eigenvectors = np.linalg.eig(covariance_of_normalised_images)

        # Sorting eigenvalues in descending order and then sort eigenvectors based on the ordering of the eigen values
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Here we are trying to see how many principal components we need to keep the ratio in variances to
        # at a certain value
        explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
        cumulative_variance = np.cumsum(explained_variance_ratio)
        variance_threshold = variance
        number_of_top_features = np.argmax(cumulative_variance >= variance_threshold) + 1

        # Selecting the top principal components
        principal_components = eigenvectors[:, :number_of_top_features]

        # Saving the top principal components to use to pca reduced any test data
        save_model(principal_components, 'principal_components')

    # Returning the projection of the images onto the selected principal components
    return np.dot(normalised_images, principal_components)


def training_model(train_features, train_labels):
    return KNNClassifier(train_features, train_labels)


class KNNClassifier(NullModel):
    """
       My custom KNN classifier.

       Parameters:
           - train_features: np.ndarray, the pca reduced training image.
           - train_labels: np.ndarray, the labels corresponding to the pca reduced training data.

       Returns:
           - the KNNClassifier initialised with pca reduced training features and labels.
   """

    def __init__(self, train_features, train_labels):
        """
        Initialising the KNN classifier with normalised training features and labels.

        Parameters:
            - train_features: np.ndarray, the training feature vectors.
            - train_labels: np.ndarray, the corresponding labels for training data.
        """
        super().__init__()
        # Normalise the training features using norm
        self.train_features = train_features / np.linalg.norm(train_features, axis=1, keepdims=True)
        self.train_labels = train_labels

    def predict(self, test_features, k=8):
        """
        Here we are predicting labels for pca reduced test images using weighted KNN with cosine distance.

        Parameters:
            - test_features: np.ndarray, the pca reduced test images.
            - k: int, the number of neighbors to consider.

        Returns:
            - predictions: np.ndarray, predicted labels for pca reduced test images.
        """
        predictions = []

        # Iterating over each pca reduced test image 
        for i in range(test_features.shape[0]):
            # Normalising further the pca reduced test image
            image = test_features[i]
            normalized_image = image / (np.linalg.norm(image) + 1e-100)

            # Calculating the cosine distances between test image and all training images
            # so we could then compare them based on distance
            cosine_distance = self.cosine_distance(normalized_image)

            # Selecting the indexes of the top k smallest distances, (KNN)
            min_indices = np.argpartition(cosine_distance, k, axis=0)[:k]
            min_labels = self.train_labels[min_indices]

            # Adding weights to the distances (Weighted KNN)
            weights = 1 / (cosine_distance[min_indices] + 1e-100)

            # Performing weighted voting to determine the predicted label
            weighted_votes = {}
            for label, weight in zip(min_labels, weights):
                weighted_votes[label] = weighted_votes.get(label, 0) + weight

            # Selecting the label with the highest weight
            predicted_label = max(weighted_votes, key=weighted_votes.get)
            predictions.append(predicted_label)

        return np.array(predictions)

    def cosine_distance(self, test_feature):
        """
        Calculating cosine distances between a test image and all training images.

        Parameters:
        - test_feature: np.ndarray, the test feature vector.

        Returns:
        - cosine_distances: np.ndarray, cosine distances between test and training features.
        """
        # Compute cosine similarity
        cosine_similarities_top_bit = np.dot(test_feature, self.train_features.T)
        cosine_similarities_bottom_bit = np.linalg.norm(test_feature) * np.linalg.norm(self.train_features)

        # Normalize to avoid division errors and compute similarity
        cosine_similarities = np.array(cosine_similarities_top_bit) / (cosine_similarities_bottom_bit + 1e-100)

        # Convert cosine similarity to cosine distance
        return 1 - cosine_similarities


def generate_gaussian_kernel(sigma):
    """
    Generate a 1-dimensional gaussian kernel using a given standard deviation.

    Parameters:
    - sigma: float, the standard deviation of the gaussian distribution.

    Returns:
    - kernel: np.ndarray, the normalised gaussian kernel.
    """
    size = int(2 * (math.ceil(3 * sigma)) + 1)  # The kernel size
    center = size // 2
    x = np.arange(-center, center + 1)
    exponent = (-(x ** 2)) / (2 * sigma ** 2)
    kernel = (np.exp(exponent)) / ((math.sqrt(2 * math.pi)) * sigma)
    return kernel / kernel.sum()


def apply_gaussian_blur(image, sigma=3.55):
    """
    Apply gaussian blur to an image using a separable 1-dimensional gaussian kernel.

    Parameters:
    - image: np.ndarray, the input image (or batch of images).
    - sigma: float, standard deviation for the gaussian kernel.

    Returns:
    - final_result: np.ndarray, the blurred image, finally.
    """
    # Generating the 1-dimensional gaussian kernel
    kernel = generate_gaussian_kernel(sigma)

    # Applying convolution along rows 
    filtered_horizontal = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode='same'), axis=1, arr=image)

    # Applying convolution along columns on the horizontally blurred image
    final_result = np.apply_along_axis(lambda col: np.convolve(col, kernel, mode='same'), axis=0,
                                       arr=filtered_horizontal)
    return final_result
