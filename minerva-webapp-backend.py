import csv
from statistics import mean

from flask import Flask
from flask import jsonify, logging
from itertools import groupby

#from flask_apscheduler import APScheduler
from pymongo import MongoClient
import pandas as pd
from datetime import datetime, timedelta
from collections import Counter
import numpy
import matplotlib.pyplot as plt
import pandas

import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import os.path
import numpy as np
import pickle
import sys

import jobs

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


def toWeek(date):
    '''(date,volume) -> date of the Sunday of that week'''
    sunday = date.strftime('%Y-%U-0')
    return datetime.strptime(sunday, '%Y-%U-%w').strftime('%Y%m%d')


@app.route('/commit_timeline/<int:board_id>/<int:days_before>')
def get_commits(board_id, days_before):
    start = datetime.now() - timedelta(days_before)
    db = get_db()
    commits = db.commits
    filtered_commits = commits.find({'board_id': board_id, 'date': {'$gte': start}},
                                    {"additions": 1, "deletions": 1, "date": 1, "message": 1})

    result_dict = {}
    for commit in filtered_commits:
        merge_commit = commit['message'].split(' ') == 'Merge'
        if "date" in commit and not merge_commit:
            date = commit["date"]
            date = toWeek(date)

            additions = int(commit["additions"])
            deletions = int(commit["deletions"])

            if additions <= 300 and deletions <= 300:
                if date in result_dict:
                    prev_addition = result_dict[date]['additions']
                    prev_deletions = result_dict[date]['deletions']
                    result_dict[date] = {"additions": additions + prev_addition,
                                         "deletions": deletions + prev_deletions, "date": date}
                else:
                    result_dict[date] = {"additions": additions,
                                         "deletions": deletions, "date": date}

    result = []
    for date, addDelTuple in result_dict.items():
        response_dict = {'date': date, 'additions': addDelTuple['additions'], 'deletions': addDelTuple['deletions']}
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
        response_dict["Mean Baseline"] = mean_velocity()

        start_date = sprint['startDate']

        start_date = start_date.strftime("%Y%m%d")
        response_dict["date"] = start_date
        result.append(response_dict)

    response = jsonify(result)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/sprint_time_line/velocity_forecast/<int:board_id>/<int:days_before>')
def get_velocity_forecast(board_id, days_before):
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

    last_sprint = filtered_sprints[len(filtered_sprints) - 1]
    last_date = last_sprint['startDate']
    end_date = last_date + timedelta(days=120)

    end_date = end_date.strftime("%Y%m%d")
    last_date = last_date.strftime("%Y%m%d")

    model_dict = read_or_create_predictions()
    predictions = model_dict[board_id]

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

        start_date = sprint['startDate']

        start_date = start_date.strftime("%Y%m%d")
        response_dict["date"] = start_date
        response_dict["lastDate"] = last_date
        result.append(response_dict)

    response_dict = {"date": end_date, "lastDate": last_date, "Percentage Completed": round(predictions[-1])}
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
    mean_percentage = mean_build_percentage()

    for build in builds:
        status = build['status']
        total += 1.0
        if status == 'SUCCESS':
            success += 1.0
        per = success / total
        success_percentage = round(100.0 * per)
        date = build['date']
        date = date.strftime("%Y%m%d")
        response_dict = {'Success Percentage': success_percentage, 'date': date, 'Mean': mean_percentage}
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
    avg_commit_per_day = mean_commit_count_per_day()

    result = []
    for key, group in g:
        count = len(list(group))
        date = key
        date = date.strftime("%Y%m%d")

        response_dict = {'date': date, 'zAvg Commit Per Day': avg_commit_per_day, 'Commits Count': count}
        result.append(response_dict)

    sorted_result = sorted(result, key=lambda k: k['date'])
    response = jsonify(sorted_result)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/commit_timeline/users/<int:board_id>/<int:days_before>')
def get_count_timeline_users(board_id, days_before):
    start = datetime.now() - timedelta(days_before)
    db = get_db()
    mg_commits = db.commits
    commits = list(mg_commits.find({'board_id': board_id, 'date': {'$gte': start}},
                                   {"board_id": 1,
                                    "date": 1,
                                    "author_name": 1,
                                    "_id": 0,
                                    }))

    users = set()
    for commit in commits:
        user_name = commit['author_name']
        users.add(user_name)

    g = groupby(commits, lambda x: x['date'].strftime("%Y%m%d"))

    result = []
    for key, group in g:
        response_dict = {'date': key}

        for user in users:
            response_dict[user] = 0

        commits = list(group)
        for commit in commits:
            user = commit['author_name']
            response_dict[user] += 1
        result.append(response_dict)

    sorted_result = sorted(result, key=lambda k: k['date'])
    response = jsonify(sorted_result)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/jenkins_timeline/test_coverage/<int:board_id>/<int:days_before>')
def get_test_coverage(board_id, days_before):
    start = datetime.now() - timedelta(days_before)
    db = get_db()
    mg_builds = db.builds
    builds = list(mg_builds.find({'board_id': board_id, 'date': {'$gte': start}},
                                 {"test_coverage": 1,
                                  "date": 1,
                                  "_id": 0,
                                  }))
    result = []

    for build in builds:
        date = build['date'].strftime("%Y%m%d")
        test_coverage = build['test_coverage']
        response_dict = {'date': date}
        if test_coverage:
            for k, v in test_coverage.items():
                response_dict[k] = v
            result.append(response_dict)

    sorted_result = sorted(result, key=lambda k: k['date'])
    response = jsonify(sorted_result)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/bugs/<int:board_id>/<int:days_before>')
def get_bug_timeline(board_id, days_before):
    start = datetime.now() - timedelta(days_before)
    db = get_db()
    mg_issues = db.issues

    issues = list(mg_issues.find({'board_id': board_id, 'created': {'$gte': start}}, {'_id': 0, 'created': 1, 'issueType': 1}))

    for issue in issues:
        issue['created'] = issue['created'].strftime("%Y%m%d")

    sorted_issues = sorted(issues, key=lambda k: k['created'])

    result_dict = {}
    prev_larva = 0
    prev_bug = 0
    for issue in sorted_issues:
        date = issue['created']
        type = issue['issueType']
        if type == 'Bug':
            prev_bug += 1
        elif type == 'Larva':
            prev_larva += 1
        result_dict[date] = {'larva': prev_larva, 'bug': prev_bug}

    result = []
    for date, value in result_dict.items():
        response_dict = {'date': date, 'Bug': value['bug'], 'Larva': value['larva']}
        result.append(response_dict)

    sorted_result = sorted(result, key=lambda k: k['date'])
    response = jsonify(sorted_result)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


def mean_commit_count_per_day():
    db = get_db()
    mg_commits = db.commits
    dates = [commit['date'].date() for commit in list(mg_commits.find({}, {"date": 1, '_id': 0}))]

    return mean(Counter(dates).values())


def mean_build_percentage():
    db = get_db()
    mg_builds = db.builds
    boards = db.boards

    board_ids = [board['_id'] for board in boards.find({}, {'_id': 1})]
    result = []
    for board_id in board_ids:
        builds = list(mg_builds.find({'board_id': board_id},
                                     {"board_id": 1,
                                      "date": 1,
                                      "status": 1,
                                      "_id": 0,
                                      }))

        total = 0.0
        success = 0.0

        for build in builds:
            status = build['status']
            total += 1.0
            if status == 'SUCCESS':
                success += 1.0
            per = success / total
            success_percentage = round(100.0 * per)
            result.append(success_percentage)

    return mean(result)


def get_sprint_details(board_id):
    start = datetime.now() - timedelta(5000)
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

        committed = not_completed + completed
        ration = float(completed) / float(committed)
        response_dict["Percentage Completed"] = round(100.0 * ration)

        start_date = sprint['startDate']

        start_date = start_date
        response_dict["date"] = start_date
        result.append(response_dict)

    sorted_result = sorted(result, key=lambda k: k['date'])
    return sorted_result


def get_predictions(board_id):
    sorted_result = get_sprint_details(board_id)
    dataset = pd.DataFrame(sorted_result)

    if len(sorted_result) <= 3:
        return None

    row = dataset[['Percentage Completed']]
    dataset = row.values.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    d = pd.DataFrame(dataset)
    look_back = 1
    trainX, trainY = create_dataset(dataset, look_back)
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=0)

    prev = np.array([trainX[len(trainX) - 1]])
    next = np.array(model.predict(prev))                    
    print(next)
    testPredict = next
    prev = np.array([next])

    future_predict = 3
    for x in range(future_predict - 1):
        next = np.array(model.predict(prev))
        print(next)
        testPredict = np.concatenate((testPredict, next), axis=0)
        prev = np.array([next])

    testPredict = scaler.inverse_transform(testPredict)
    return testPredict.flatten().tolist()


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)


def read_or_create_predictions():
    db = get_db()
    model_dict = {}
    dir_path = os.path.dirname(os.path.realpath(__file__))

    if not os.path.isfile(dir_path + '/static/predictions.pickle'):
        mg_boards = db.boards

        cursor = mg_boards.find({}, {'_id': 1})
        boards = [board['_id'] for board in cursor]
        for board in boards:
            predictions = get_predictions(board)
            if predictions is not None:
                model_dict[board] = predictions

        with open(dir_path + '/static/predictions.pickle', 'wb') as handle:
            pickle.dump(model_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(dir_path + '/static/predictions.pickle', 'rb') as handle:
            model_dict = pickle.load(handle)
    return model_dict


class Config(object):
    JOBS = [
        {
            'id': 'jira_update',
            'func': jobs.update_jira_data,
            'args': [get_db()],
            'trigger': 'cron',
            'minute': '*/2'
            #'hour': '8,14'
        },
        {
            'id': 'github_update',
            'func': jobs.update_github_data,
            'args': [get_db()],
            'trigger': 'cron',
            'minute': '*/2'
            #'hour': '8,14'
        },
        {
            'id': 'jenkins_update',
            'func': jobs.update_jenkins_data,
            'args': [get_db()],
            'trigger': 'cron',
            'minute': '*/2'
            #'hour': '8,14'
        }
    ]

    SCHEDULER_API_ENABLED = True

if __name__ == '__main__':
    model_dict = read_or_create_predictions()
    app.config.from_object(Config())

    '''
    scheduler = APScheduler()
    scheduler.init_app(app)
    scheduler.start()
    '''

    app.run('0.0.0.0')
    # commit_stats()
