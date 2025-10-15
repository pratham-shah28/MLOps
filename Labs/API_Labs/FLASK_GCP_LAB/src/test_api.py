import requests

url = 'http://127.0.0.1:5000/predict'

# Example: 13 feature inputs for the Wine dataset
data = {
    'alcohol': 13.2,
    'malic_acid': 2.77,
    'ash': 2.51,
    'alcalinity_of_ash': 18.5,
    'magnesium': 98,
    'total_phenols': 2.2,
    'flavanoids': 1.45,
    'nonflavanoid_phenols': 0.42,
    'proanthocyanins': 1.4,
    'color_intensity': 4.0,
    'hue': 1.0,
    'od280/od315_of_diluted_wines': 3.08,
    'proline': 820
}

response = requests.post(url, data=data)

if response.status_code == 200:
    prediction = response.json().get('prediction')
    print('Predicted wine class:', prediction)
else:
    print('Error:', response.status_code)
