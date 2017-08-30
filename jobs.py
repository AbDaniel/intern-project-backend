from jira_data import collected_jira_data
import sys

def update_jira_data(db):
    boards = db.boards
    board_ids = [board['_id'] for board in boards.find({}, {'_id': 1})]

    # Complete data refresh
    # This can be made less redundant by only downloading new data and
    # not dropping the collections
    db.boards.drop()
    db.sprints.drop()

    print(board_ids)
    sys.stdout.flush()

    # Check if any board_ids currently exist
    if board_ids:
        collected_jira_data(board_ids)

    return True
