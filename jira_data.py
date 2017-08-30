import pickle
import requests

from dateutil.parser import parse
from jira import JIRA
from pymongo import MongoClient


def get_sprint_url(board_id, sprint_id):
    url = f"https://jira.td.teradata.com/jira/rest/greenhopper/1.0/rapid/charts/sprintreport?rapidViewId={board_id}&sprintId={sprint_id}"
    return url


def get_board_url(board_id):
    url = f"https://jira.td.teradata.com/jira/rest/agile/1.0/board/{board_id}"
    return url


def getEstimateSum(velocity_data, field):
    completed = velocity_data['contents'][field]
    return completed['value'] if 'value' in completed else None


def getIdFromList(list_of_dict):
    return [issue['id'] for issue in list_of_dict]


jira_url = "https://jira.td.teradata.com/jira"
user_name = 'AW186034'
password = 'Ab2Daniel'
mongo_url = 'mongodb://localhost:27017/'


def collected_jira_data(board_ids):
    gh = JIRA(jira_url, basic_auth=(user_name, password))
    client = MongoClient(mongo_url)
    db = client.poc_teradata

    boards_dict_list = []
    for board_id in board_ids:
        response = requests.get(get_board_url(board_id), auth=(user_name, password)).json()
        board_dict = {"name": response["name"], "type": response["type"], "_id": response["id"]}
        boards_dict_list.append(board_dict)

    mg_boards = db.boards
    mg_boards.insert_many(boards_dict_list)

    sprint_data_list = []
    for board in boards_dict_list:
        board_id = board['_id']
        sprints = gh.sprints(board_id)
        for sprint in sprints:
            if sprint.state == 'FUTURE':
                continue

            response = requests.get(get_sprint_url(board_id, sprint.id), auth=('aw186034', password))
            velocity_data = response.json()
            sprint_dict = {'_id': sprint.id, 'board_id': board_id, 'name': sprint.name, 'state': sprint.state,

                           'completeDate': None if sprint.state == 'ACTIVE' else parse(
                               velocity_data['sprint']['completeDate']),
                           'startDate': parse(velocity_data['sprint']['startDate']),
                           'endDate': parse(velocity_data['sprint']['endDate']),
                           'daysRemaining': velocity_data['sprint']['daysRemaining'],

                           'allIssuesEstimateSum': getEstimateSum(velocity_data, 'allIssuesEstimateSum'),
                           'completedIssuesEstimateSum': getEstimateSum(velocity_data, 'completedIssuesEstimateSum'),
                           'completedIssuesInitialEstimateSum': getEstimateSum(velocity_data,
                                                                               'completedIssuesInitialEstimateSum'),
                           'issuesNotCompletedEstimateSum': getEstimateSum(velocity_data,
                                                                           'issuesNotCompletedEstimateSum'),
                           'issuesNotCompletedInitialEstimateSum': getEstimateSum(velocity_data,
                                                                                  'issuesNotCompletedInitialEstimateSum'),
                           'issuesCompletedInAnotherSprintEstimateSum': getEstimateSum(velocity_data,
                                                                                       'issuesCompletedInAnotherSprintEstimateSum'),
                           'issuesCompletedInAnotherSprintInitialEstimateSum': getEstimateSum(velocity_data,
                                                                                              'issuesCompletedInAnotherSprintInitialEstimateSum'),
                           'puntedIssuesEstimateSum': getEstimateSum(velocity_data, 'puntedIssuesEstimateSum'),
                           'puntedIssuesInitialEstimateSum': getEstimateSum(velocity_data,
                                                                            'puntedIssuesInitialEstimateSum'),

                           'puntedIssues': velocity_data['contents']['completedIssues'],
                           'completedIssues': velocity_data['contents']['completedIssues'],
                           'issuesCompletedInAnotherSprint': velocity_data['contents']['completedIssues'],
                           'issuesNotCompletedInCurrentSprint': velocity_data['contents']['completedIssues']}

            sprint_data_list.append(sprint_dict)

    new_sprint_dict_list = []

    for sprint in sprint_data_list:
        sprint_dict = {'_id': {'sprint_id': sprint['_id'], 'board_id': sprint['board_id']}}
        for key in sprint:
            if key == '_id':
                continue
            sprint_dict[key] = sprint[key]
        new_sprint_dict_list.append(sprint_dict)

    mg_sprints = db.sprints
    mg_sprints.insert_many(new_sprint_dict_list)
