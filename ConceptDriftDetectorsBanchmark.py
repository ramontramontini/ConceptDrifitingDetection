import warnings
import numpy as np
from abc import ABC, abstractmethod
from scipy.special import rel_entr
from scipy.spatial.distance import jensenshannon
from scipy.stats import chisquare, wasserstein_distance, ks_2samp
from sklearn.metrics import confusion_matrix
from TransformedMNIST import TransformedMNIST
from Report import Report

class DriftDetector(ABC):
    def __init__(self, mnist, trasnformation_type, alpha=0.05):
        print(f"Initializing detector: {self.name()} transformation_type: {trasnformation_type}")
        self.mnist = mnist
        self.alpha = alpha
        self.transformation_type = trasnformation_type
        self.original_images = mnist.training_validation_labels
        if trasnformation_type == "Rotation":
            self.transformed_images = self.mnist.rotationDrifitImagesSet
            self.altered_indices = self.mnist.rotatedImagesIndices
        if trasnformation_type == "Scale":
            self.transformed_images = self.mnist.scaleDrifitImagesSet
            self.altered_indices = self.mnist.scaledImagesIndices
        if trasnformation_type == "Noise":
            self.transformed_images = self.mnist.noisedDrifitImagesSet
            self.altered_indices = self.mnist.noisyImagesIndices
        
    @abstractmethod     
    def name(self):
        pass

    @abstractmethod
    def compute_statistic(self, pixel_reference, pixel_new):
        pass

    def detect_drift(self):
        self.report = Report(self, self.transformation_type)
        self.report.start_detection_timer()
        reference_data = self.mnist.training_validation_images

        if len(reference_data.shape) != 3 or len(self.transformed_images.shape) != 3:
            raise ValueError(f"Data is not in expected 3D format. Reference data shape: {reference_data.shape}, New data shape: {self.transformed_images.shape}")

        num_images = reference_data.shape[0]
        true_labels = np.zeros(num_images, dtype=int)
        predicted_labels = np.zeros(num_images, dtype=int)
        true_labels[self.altered_indices] = 1

        drift_count = 0
        for img_index in range(num_images):
            pixel_reference = reference_data[img_index].flatten()
            pixel_new = self.transformed_images[img_index].flatten()

            if self.compute_statistic(pixel_reference, pixel_new):
                predicted_labels[img_index] = 1
                drift_count += 1

        self.report.stop_detection_timer()

        cm = confusion_matrix(true_labels, predicted_labels)
        drift_percentage = (drift_count / num_images) * 100
        accuracy = (drift_percentage/self.mnist.modification_proportion)
        self.report.set_metrics(cm, accuracy)
        return drift_percentage

class KLDivergence(DriftDetector):
    def name(self):
        return "Kullback-Leibler Divergence"

    def compute_statistic(self, pixel_reference, pixel_new):
        ref_hist, bin_edges = np.histogram(pixel_reference, bins=10, density=True)
        new_hist, _ = np.histogram(pixel_new, bins=bin_edges, density=True)
        kl_div = np.sum(rel_entr(ref_hist, new_hist))
        return kl_div > self.alpha

class ChiSquared(DriftDetector):
    def name(self):
        return "Chi-Squared"

    def compute_statistic(self, pixel_reference, pixel_new):
        bins = np.histogram_bin_edges(np.concatenate((pixel_reference, pixel_new)), bins=10)
        hist_ref, _ = np.histogram(pixel_reference, bins)
        hist_new, _ = np.histogram(pixel_new, bins)
        _, p_val = chisquare(hist_ref, hist_new)
        return p_val < self.alpha
    
class WassersteinDistance(DriftDetector):
    def name(self):
        return "Wasserstein (Earth-Mover Distance)"
    
    def compute_statistic(self, pixel_reference, pixel_new):
        p_val = wasserstein_distance(pixel_reference, pixel_new)
        return p_val > self.alpha

class KolmogorovSmirnov(DriftDetector):
    def name(self):
        return "Kolmogorov-Smirnov"
    
    def compute_statistic(self, pixel_reference, pixel_new):
        _, p_val = ks_2samp(pixel_reference, pixel_new)
        return p_val < self.alpha

class PSI(DriftDetector):
    def __init__(self, mnist, transformation_type, alpha=0.1, bins=10):
        super().__init__(mnist, transformation_type, alpha)
        self.bins = bins
    
    def name(self):
        return "Population Stability Index (PSI)"
    
    def compute_statistic(self, pixel_reference, pixel_new):
        ref_hist, bin_edges = np.histogram(pixel_reference, bins=self.bins)
        new_hist, _ = np.histogram(pixel_new, bins=bin_edges)

        ref_hist = ref_hist / len(pixel_reference)
        new_hist = new_hist / len(pixel_new)

        psi_values = (new_hist - ref_hist) * np.log((new_hist + 1e-5) / (ref_hist + 1e-5))
        psi = np.sum(psi_values)

        return psi > self.alpha
class JensenShannon(DriftDetector):
    def __init__(self, mnist, transformation_type, alpha=0.05):
        super().__init__(mnist, transformation_type, alpha)
    
    def name(self):
        return "Jensen-Shannon Distance"
    
    def compute_statistic(self, pixel_reference, pixel_new):
        # Normalize the pixel values to create probability distributions
        pixel_reference_norm = pixel_reference / np.sum(pixel_reference)
        pixel_new_norm = pixel_new / np.sum(pixel_new)

        # Compute the Jensen-Shannon distance
        js_distance = jensenshannon(pixel_reference_norm, pixel_new_norm)

        return js_distance > self.alpha



# Suppress specific runtime warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, module='scipy.stats._stats_py', lineno=8064)
warnings.filterwarnings('ignore', 'ks_2samp: Exact calculation unsuccessful. Switching to method=asymp.', RuntimeWarning, 'ConceptDriftDetectorsBanchmark', 100)

mnist_dataset = TransformedMNIST()  # Assuming this is already define
transformations = ["Rotation", "Scale", "Noise"]
detector_classes = [KLDivergence, ChiSquared, WassersteinDistance, KolmogorovSmirnov, PSI, JensenShannon]
for detector_class in detector_classes:
    for transformation in transformations:
        detector_class(mnist_dataset, transformation).detect_drift()

Report.print_report()
Report.print_matrices()
mnist_dataset.plot_change_matrix()
