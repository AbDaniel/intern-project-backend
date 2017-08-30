from jira_data import collected_jira_data


def update_jira_data(db):
    boards = db.boards
    board_ids = [board['_id'] for board in boards.find({}, {'_id': 1})]

    collected_jira_data(board_ids)

    return True
