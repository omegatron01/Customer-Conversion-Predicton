# Customer Conversion Prediction Model 

This project is a containerized Machine Learning service. It predicts whether a client will subscribe to a term deposit (conversion) based on their profile and interactions.

## Project Structure

* `Dockerfile`: Instructions to build the Linux-based container with Python 3.11 and Gunicorn.
* `predict.py`: Flask application that loads the model and serves predictions.
* `model_1.0.bin`: The trained machine learning model (saved via pickle).
* `requirements.txt`: Python dependencies (Flask, Scikit-Learn, Gunicorn, etc.).
* `predict_test.py`: A script to send a test request to the running container.

## How to Run (with Docker)

### 1. Build the Image
Run the following command in the project root:

docker build -t zoomcamp-model .

### 2. Run the Container
Start the container and map port 9696:

docker run -it -p 9696:9696 zoomcamp-model

Note: The container uses Gunicorn as the production server.

## How to Test
Once the container is running, open a separate terminal and run the test script:

python predict_test.py

Sample Request Payload
The model expects a JSON object similar to this:


{
  "lead_source": "events",
  "industry" : "healthcare",
  "number_of_courses_viewed" : 5,
  "annual_income" : 78796.0,
  "employment_status" : "unemployed", 
  "location" : "australia",
  "interaction_count" : 3,
  "lead_score" : 0.69 
  }


Sample Response
The API returns the probability of conversion:

{
  "conversion": true,
  "conversion_probability": 0.8269
}

###Local Development (Without Docker)
If you wish to run it locally without Docker:

Create a virtual environment: 

python -m venv venv

Activate it: source venv/bin/activate (Linux/Mac) or venv\Scripts\activate (Windows)

Install dependencies: pip install -r requirements.txt

Run the app: python predict.py
