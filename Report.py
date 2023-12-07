import time
import pandas as pd

class Report:
    all_reports = []  # Class-level attribute to store all instances of Report

    def __init__(self, detector, transformation_type):
        self.detector = detector
        self.transformation_type = transformation_type
        self.detection_time = 0
        self.confusion_matrix = None
        self.transformed_dataset_size = 0
        self.transformed_images_count = 0
        self.modification_proportion = detector.mnist.modification_proportion
        
        self.accuracy = 0
        
        Report.all_reports.append(self)  # Add instance to the class-level list

    def start_detection_timer(self):
        self.detection_time = time.time()

    def stop_detection_timer(self):
        self.detection_time = time.time() - self.detection_time

    def set_metrics(self, confusion_matrix, accuracy):
        self.confusion_matrix = confusion_matrix
        self.transformed_dataset_size = len(self.detector.mnist.training_validation_images)
        self.transformed_images_count = self.detector.mnist.transformed_images_count
        self.accuracy = accuracy

    @classmethod
    def print_matrices(cls):
        ("\nConfusion Matrices:")
        for report in cls.all_reports:
            print(f"\nDetector: {report.detector.name()}, Type: {report.transformation_type}")
            print(report.confusion_matrix)

    @classmethod
    def print_report(cls):
        
        print ()
        # Create a dictionary to hold the report data
        report_dict = {}

        # Aggregate the data
        for report in cls.all_reports:
            detector_name = report.detector.name()
            transformation_type = report.transformation_type
            # Initialize the nested dictionary if necessary
            if detector_name not in report_dict:
                report_dict[detector_name] = {}
            if transformation_type not in report_dict[detector_name]:
                report_dict[detector_name][transformation_type] = {'Time': [], 'Accuracy': []}
            # Append the time and accuracy data
            report_dict[detector_name][transformation_type]['Time'].append(f"{report.detection_time:.2f}s")
            report_dict[detector_name][transformation_type]['Accuracy'].append(f"{report.accuracy:.2f}%")

        # Convert the dictionary to a DataFrame
        data_rows = []
        for detector, transformations in report_dict.items():
            for transformation, metrics in transformations.items():
                data_rows.append([
                    transformation,
                    detector, 
                    ', '.join(metrics['Accuracy']),
                    ', '.join(metrics['Time'])
                ])
        df = pd.DataFrame(data_rows, columns=['Detector', 'Transformation', 'Accuracy', 'Time'])
        df_pivot = df.pivot(index='Transformation', columns='Detector', values=['Accuracy', 'Time' ])
        print("Report by Detector:")
        print(df_pivot)