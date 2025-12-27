import requests

url = 'http://localhost:9696/predict'

customer_id = 'xyz-123'

customer = {
            "lead_source": "events",
            "industry" : "healthcare",
            "number_of_courses_viewed" : 5,
            "annual_income" : 78796.0,
            "employment_status" : "unemployed", 
            "location" : "australia",
            "interaction_count" : 3,
            "lead_score" : 0.69 
            }

response = requests.post(url, json=customer)
result = response.json()

print(result)

if result['conversion'] == True:
    print('Sending email to %s' % customer_id)
else:
    print('Not sending email to %s' % customer_id)

