import pickle

import requests
from dateutil.parser import parse
from jira import JIRA
from pymongo import MongoClient


from github import Github
import pandas as pd

def create_commit_dict(commit, repo_name, board_id):
    commit_dict = {'_id': commit.sha,
                   'board_id': board_id,
                   'repo_name': repo_name,
                   'message': commit.commit.message,
                   'additions': commit.stats.additions,
                   'deletions': commit.stats.deletions,
                   'total': commit.stats.total,
                   'url': commit.url}

    git_commit = commit.commit
    author = git_commit.author

    commit_dict['author_email'] = author.email
    commit_dict['author_name'] = author.name
    commit_dict['date'] = author.date

    return commit_dict

TOKEN = "3b5ef4b906cf9c7449d82118aba2d99379a90a71"

g = Github(TOKEN, base_url="https://github.td.teradata.com/api/v3")

repo_id_name_dict = {
    419: g.get_organization('open-platform-ag').get_repo('tdc_aws_all'),
    513: g.get_organization('tdmc').get_repo('tdmc-app'),
    491: g.get_organization('tvme-esxi').get_repo('tvme_mpp')
}


# commit_list_per_repo = [repo.get_commits() for repo in repo_dict.values()]

repo_dict = dict()
repo_name_board_id_dict = dict()

for key, repo in repo_id_name_dict.items():
    repo_dict[repo.name] = list(repo.get_commits())
    repo_name_board_id_dict[repo.name] = key

all_commits = []

for repo_name, value in repo_dict.items():
    board_id = repo_name_board_id_dict[repo_name]
    commits = [create_commit_dict(commit, repo_name, board_id) for commit in value]
    all_commits.extend(commits)

retrived_commits = set([commit['_id'] for commit in  all_commits])

for repo_name, value in repo_dict.items():
    board_id = repo_name_board_id_dict[repo_name]
    commits = [create_commit_dict(commit, repo_name, board_id) for commit in value if commit.sha not in retrived_commits]
    all_commits.extend(commits)

client = MongoClient('mongodb://localhost:27017/')
db = client.poc_teradata

mg_commits = db.commits
mg_commits.insert_many(all_commits)