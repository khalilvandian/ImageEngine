import marimo

__generated_with = "0.17.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import csv

    def calculate_metrics():
        # Parse detection results
        detection_results = {}
        with open("detection_results.csv", "r") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                detection_results[row[0]] = row[1] == "Yes"

        # Read ground truth
        ground_truth = {}
        with open("D:/Projects/ImageEngine/Images/Anno/identity_CelebA.txt", "r") as f:
            for line in f:
                parts = line.strip().split()
                ground_truth[parts[0]] = int(parts[1])

        # Calculate metrics
        tp = 0
        fp = 0
        tn = 0
        fn = 0

        for image_name, detected in detection_results.items():
            is_hugh = ground_truth.get(image_name) == 3899
            if detected and is_hugh:
                tp += 1
            elif detected and not is_hugh:
                fp += 1
            elif not detected and not is_hugh:
                tn += 1
            elif not detected and is_hugh:
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0

        print(f"True Positives: {tp}")
        print(f"False Positives: {fp}")
        print(f"True Negatives: {tn}")
        print(f"False Negatives: {fn}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"Accuracy: {accuracy:.2f}")

    if __name__ == "__main__":
        calculate_metrics()
    return


if __name__ == "__main__":
    app.run()
