import csv


def read_csv(csv_path):
    csv_content = []
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            csv_content.append(row)
    return csv_content
