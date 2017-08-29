# Teradata 360 Installation Guide

## Mongo DB Setup
* Install MongoDB
* Start MongoDB service
* Run the following command in mongo shell to migrate data
```
    db.copyDatabase(fromdb, todb, fromhost)
```

#### MongoDB Instance Running in the staging server (Date: August 29, 2017)

* **VM IP**: 10.25.237.183
* **DB name**: poc_teradata

#### To migrate data from Staging server to local
```
    db.copyDatabase('poc_teradata', 'poc_teradata', '10.25.237.183')
```


## Python environment setup
* Install Python 3.6 (Use of Virtual Environment is preferred)
* Clone the repo
* Run ``` cd reponame ```
* Activate the virtual env, if you are using one.
* Run ```pip install -r requirements.txt```
* Run ```python minerva-webapp-backend.py``` to start the backend
* Use the url http://localhost:5000/sprint/stats to make sure that the backend is running.