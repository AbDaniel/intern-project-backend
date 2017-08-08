import csv

from flask import Flask
from flask import jsonify

from pymongo import MongoClient
import pandas as pd
from datetime import datetime, timedelta

app = Flask(__name__)


def get_db():
    client = MongoClient('localhost:27017')
    db = client.poc_teradata
    return db


@app.route('/')
def hello_world():
    return app.send_static_file('index.html')


@app.route('/sprint/stats')
def sprint_stats():
    db = get_db()
    sprints = db.sprints

    columns = ["allIssuesEstimateSum",
               "completedIssuesEstimateSum",
               "completedIssuesInitialEstimateSum",
               "issuesNotCompletedEstimateSum",
               'issuesNotCompletedInitialEstimateSum',
               "issuesCompletedInAnotherSprintEstimateSum",
               "issuesCompletedInAnotherSprintInitialEstimateSum",
               "puntedIssuesEstimateSum",
               "puntedIssuesInitialEstimateSum"]

    needed_cols = {'_id': 0}
    for col in columns:
        needed_cols[col] = 1

    cursor = sprints.find({}, needed_cols)
    table = pd.DataFrame(list(cursor))
    response = table.describe().to_dict()
    return jsonify(response)


@app.route('/boards')
def get_boards():
    db = get_db()
    boards = db.boards

    cursor = boards.find({})
    response = jsonify((list(cursor)))
    # table = pd.DataFrame(list(cursor))
    # result = table.to_dict()
    # response = jsonify(result)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/build/stats')
def build_stats():
    db = get_db()
    builds = db.builds

    build_collection = []

    for build in builds.find({}, {'_id': 0, 'test_coverage.Classes': 1,
                                  'test_coverage.Packages': 1,
                                  'test_coverage.Files': 1,
                                  'test_coverage.Conditionals': 1,
                                  'test_coverage.Lines': 1}):
        if build != {}:
            build_collection.append(build['test_coverage'])

    table = pd.DataFrame(build_collection)
    response = table.describe().to_dict()
    return jsonify(response)


@app.route('/commit/stats')
def commit_stats():
    db = get_db()
    commits = db.commits

    columns = ["allIssuesEstimateSum",
               "completedIssuesEstimateSum",
               "completedIssuesInitialEstimateSum",
               "issuesNotCompletedEstimateSum",
               'issuesNotCompletedInitialEstimateSum',
               "issuesCompletedInAnotherSprintEstimateSum",
               "issuesCompletedInAnotherSprintInitialEstimateSum",
               "puntedIssuesEstimateSum",
               "puntedIssuesInitialEstimateSum"]

    needed_cols = {'_id': 0}
    for col in columns:
        needed_cols[col] = 1

    cols = ['additions', 'deletions', 'total']

    needed_cols = {'_id': 0}
    for col in cols:
        needed_cols[col] = 1

    cursor = commits.find({}, needed_cols)
    table = pd.DataFrame(list(cursor))
    response = table.describe().to_dict()
    return jsonify(response)


@app.route('/sprint/<int:board_id>/<int:days_before>')
def get_sprints_by_board_id(board_id, days_before):
    start = datetime.now() - timedelta(days_before)
    db = get_db()
    sprints = db.sprints
    filtered_sprints = list(sprints.find({'board_id': board_id, 'startDate': {'$gte': start}},
                                         {"board_id": 1,
                                          "name": 1,
                                          "completedIssuesEstimateSum": 1,
                                          "issuesNotCompletedEstimateSum": 1}))

    response = {"completed_estimate": [], "not_completed_estimate": [], "sprint_name": []}

    for sprint in filtered_sprints:
        not_completed = sprint['issuesNotCompletedEstimateSum']
        completed = sprint['completedIssuesEstimateSum']
        if not_completed is None and completed is None:
            continue
        if not_completed is None:
            not_completed = 0

        response["completed_estimate"].append(completed)
        response["not_completed_estimate"].append(not_completed)
        response["sprint_name"].append(sprint['name'])

    return jsonify(response)


@app.route('/sprint_time_line/<int:board_id>/<int:days_before>')
def get_sprints_by_board_id_with(board_id, days_before):
    start = datetime.now() - timedelta(days_before)
    db = get_db()
    sprints = db.sprints
    filtered_sprints = list(sprints.find({'board_id': board_id, 'startDate': {'$gte': start}},
                                         {"board_id": 1,
                                          "startDate": 1,
                                          "completedIssuesEstimateSum": 1,
                                          "name": 1,
                                          "issuesNotCompletedEstimateSum": 1}))

    result = []

    for sprint in filtered_sprints:
        response_dict = {}
        not_completed = sprint['issuesNotCompletedEstimateSum']
        completed = sprint['completedIssuesEstimateSum']
        if not_completed is None and completed is None:
            continue
        if not_completed is None:
            not_completed = 0

        response_dict["completed_estimate"] = completed
        commited = not_completed + completed
        response_dict["not_completed_estimate"] = commited
        ration = float(completed) / float(commited)
        response_dict["percentage_completed"] = round(100.0 * ration)
        response_dict["name"] = sprint["name"]
        response_dict["mean"] = 75

        start_date = sprint['startDate']

        start_date = start_date.strftime("%Y%m%d")
        response_dict["date"] = start_date
        result.append(response_dict)

    response = jsonify(result)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/board/<int:board_id>')
def get_board_by_id(board_id):
    db = get_db()
    boards = db.boards

    board = boards.find_one({'_id': board_id})
    response = jsonify(board)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


if __name__ == '__main__':
    # get_sprints_by_board_id_with(491, 365)
    # print commit_stats(   )
    app.run('0.0.0.0')
    # commit_stats()
