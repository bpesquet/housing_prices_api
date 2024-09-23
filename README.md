# Housing Prices API

A simple [Flask](https://palletsprojects.com/p/flask/) application demonstrating how to deploy and access a trained Machine Learning model.

See [this repository](https://github.com/bpesquet/mlcourse) for details about the training process.

## Prediction API

The `/predict` endpoint expects a **POST** request whose body must contain the input data for a housing district (`population`, `median_income`...) as JSON. See an [example of valid JSON payload](example_request_body.json).

It returns the model's prediction (median price for the housing district) as a JSON object with a `median_housing_price` property.

## Running the server locally

Launch the following command to run the server locally.

```bash
> env FLASK_APP=main.py FLASK_ENV=development flask run
```

You must also update the JavaScript `prediction_url` variable in `templates.index.html` to point to the local web server instead of the Heroku app.

```js
// Switch between local and remote (Heroku) deployment
// const prediction_url = "http://127.0.0.1:5000/predict";
const prediction_url = "https://housing-prices-api.herokuapp.com/predict";
```
