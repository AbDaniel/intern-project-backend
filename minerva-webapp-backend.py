import csv
from statistics import mean

from flask import Flask
from flask import jsonify
from itertools import groupby

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


def mean_velocity():
    db = get_db()
    boards = db.boards
    sprints = db.sprints

    board_ids = [board['_id'] for board in boards.find({}, {'_id': 1})]

    mean_per_board = []
    for board_id in board_ids:
        filtered_sprints = list(sprints.find({'board_id': board_id, 'state': 'CLOSED'},
                                             {"board_id": 1,
                                              "completedIssuesEstimateSum": 1,
                                              "name": 1,
                                              "issuesNotCompletedEstimateSum": 1}))
        mean_per_sprint = []
        for sprint in filtered_sprints:
            not_completed = sprint['issuesNotCompletedEstimateSum']
            completed = sprint['completedIssuesEstimateSum']
            if not_completed is None and completed is None:
                continue
            if not_completed is None:
                not_completed = 0

            committed = not_completed + completed
            ration = float(completed) / float(committed)
            percentage_completed = 100.0 * ration
            mean_per_sprint.append(percentage_completed)

        mean_per_board.append(mean(mean_per_sprint))
    return round(mean(mean_per_board))


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


@app.route('/commit_timeline/<int:board_id>/<int:days_before>')
def get_commits(board_id, days_before):
    start = datetime.now() - timedelta(days_before)
    db = get_db()
    commits = db.commits
    filtered_commits = commits.find({'board_id': board_id, 'date': {'$gte': start}},
                                    {"additions": 1, "deletions": 1, "date": 1})

    result = []
    for commit in filtered_commits:
        if "date" in commit:
            date = commit["date"]
            date = date.strftime("%Y%m%d")
            response_dict = {"additions": commit["additions"], "deletions": commit["deletions"], "date": date}
            result.append(response_dict)

    sorted_result = sorted(result, key=lambda k: k['date'])
    response = jsonify(sorted_result)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/sprint_time_line/<int:board_id>/<int:days_before>')
def get_sprints_by_board_id_with(board_id, days_before):
    start = datetime.now() - timedelta(days_before)
    db = get_db()
    sprints = db.sprints
    filtered_sprints = list(sprints.find({'board_id': board_id, 'state': 'CLOSED', 'startDate': {'$gte': start}},
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

        response_dict["Completed"] = completed
        commited = not_completed + completed
        response_dict["Committed"] = commited
        ration = float(completed) / float(commited)
        response_dict["Percentage Completed"] = round(100.0 * ration)
        response_dict["name"] = sprint["name"]
        response_dict["Mean"] = 75

        start_date = sprint['startDate']

        start_date = start_date.strftime("%Y%m%d")
        response_dict["date"] = start_date
        result.append(response_dict)

    response = jsonify(result)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/sprint_time_line/velocity/<int:board_id>/<int:days_before>')
def get_velocity(board_id, days_before):
    start = datetime.now() - timedelta(days_before)
    db = get_db()
    sprints = db.sprints

    # mean = compute_mean(board_id)
    filtered_sprints = list(sprints.find({'board_id': board_id, 'state': 'CLOSED', 'startDate': {'$gte': start}},
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

        committed = not_completed + completed
        ration = float(completed) / float(committed)
        response_dict["Percentage Completed"] = round(100.0 * ration)
        response_dict["name"] = sprint["name"]
        response_dict["Mean Baseline"] = mean_velocity()

        start_date = sprint['startDate']

        start_date = start_date.strftime("%Y%m%d")
        response_dict["date"] = start_date
        result.append(response_dict)

    response = jsonify(result)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/sprint_time_line/users/<int:board_id>/<int:days_before>')
def get_sprints_by_board_id_by_users(board_id, days_before):
    start = datetime.now() - timedelta(days_before)
    db = get_db()
    sprints = db.sprints
    filtered_sprints = list(sprints.find({'board_id': board_id, 'state': 'CLOSED', 'startDate': {'$gte': start}},
                                         {"board_id": 1,
                                          "startDate": 1,
                                          "completedIssues": 1,
                                          "name": 1,
                                          }))

    users = set()
    for sprint in filtered_sprints:
        for issue in sprint['completedIssues']:
            user = issue['assigneeName']
            if user not in users:
                estimate = issue['estimateStatistic']['statFieldValue']
                if 'value' in estimate:
                    estimate_value = int(estimate['value'])
                    if estimate_value is not 0:
                        users.add(user)

    result = []
    for sprint in filtered_sprints:
        start_date = sprint['startDate']
        start_date = start_date.strftime("%Y%m%d")

        response_dict = {"name": sprint["name"], "date": start_date}
        for user in users:
            response_dict[user] = 0

        for issue in sprint['completedIssues']:
            estimate = issue['estimateStatistic']['statFieldValue']
            if 'value' in estimate:
                estimate_value = int(estimate['value'])
                if estimate_value is not 0:
                    user = issue['assigneeName']
                    previous_estimate = response_dict[user]
                    response_dict[user] = previous_estimate + estimate_value

        result.append(response_dict)

    sorted_result = sorted(result, key=lambda k: k['date'])
    response = jsonify(sorted_result)
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


@app.route('/jenkins_timeline/<int:board_id>/<int:days_before>')
def get_jenkins_timeline(board_id, days_before):
    start = datetime.now() - timedelta(days_before)
    db = get_db()
    mg_builds = db.builds
    builds = list(mg_builds.find({'board_id': board_id, 'date': {'$gte': start}},
                                 {"board_id": 1,
                                  "date": 1,
                                  "status": 1,
                                  "_id": 0,
                                  }))

    total = 0.0
    success = 0.0

    result = []
    for build in builds:
        status = build['status']
        total += 1.0
        if status == 'SUCCESS':
            success += 1.0
        per = success / total
        success_percentage = round(100.0 * per)
        date = build['date']
        date = date.strftime("%Y%m%d")
        response_dict = {'Success Percentage': success_percentage, 'date': date}
        result.append(response_dict)

    sorted_result = sorted(result, key=lambda k: k['date'])
    response = jsonify(sorted_result)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/commit_timeline/count/<int:board_id>/<int:days_before>')
def get_commits_count(board_id, days_before):
    start = datetime.now() - timedelta(days_before)
    db = get_db()
    mg_commits = db.commits
    commits = list(mg_commits.find({'board_id': board_id, 'date': {'$gte': start}},
                                    {"board_id": 1,
                                     "date": 1,
                                     "author_name": 1,
                                     "_id": 0,
                                     }))

    for commit in commits:
        commit['date'] = commit['date'].date()

    g = groupby(commits, lambda x: x['date'])

    result = []
    for key, group in g:
        count = len(list(group))
        date = key
        date = date.strftime("%Y%m%d")

        response_dict = {'date': date, 'commits': count}
        result.append(response_dict)

    sorted_result = sorted(result, key=lambda k: k['date'])
    response = jsonify(sorted_result)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


if __name__ == '__main__':
    # get_sprints_by_board_id_with(491, 365)
    app.run()
    # commit_stats()
