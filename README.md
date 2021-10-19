## Readme - OpenTable Data Science Coding Challenge

Purpose of this file is to give instructions to the project folder as well as providing test commands to run for the three endpoints. 

# Project Folder Structure

The project folder contains several different files:

EmbeddedApp.py
- Top level Python script that must be run to start up the Flask server. Upon startup, the tensorflow model is loaded and distributed to the various endpoint classes. Flask is run in debug mode. 
- Command to run: python3 EmbeddedApp.py
  
App_Modules/EmbeddedAppModules.py
- Helper Python script where the methods to proces input data, perform checks, and produce the results are stored. 

test_app.py
- Python scrip to be run to perform unit testing (8 in total).  
- Command to run: pytest test_app.py

# Endpoint commands

Sample input for GET endpoint (/embeddings)
    curl "http://127.0.0.1:5000/embeddings?sentence=hello+from+the+pther+side"

Sample input for POST endpoint (/embeddings/bulk)
    curl -X POST -H "Content-Type: application/json" -d @payload2.json "http://127.0.0.1:5000/embeddings/bulk"

Sample input for POST endpoint (/embeddings/similarity)
    curl -X POST -H "Content-Type: application/json" -d @payload3.json "http://127.0.0.1:5000/embeddings/similarity"