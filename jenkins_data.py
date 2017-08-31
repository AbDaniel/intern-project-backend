import requests
from jenkinsapi.jenkins import Jenkins
from pymongo import MongoClient


def get_build_data(build_start_no, build_end_no, J, job_name):
    all_builds = []
    for build_no in range(build_start_no, build_end_no):
        try:
            build = J[job_name].get_build(build_no)
            all_builds.append(build)
        except:
            print("Build not found" + str(build_no))
    return all_builds


def get_test_coverage_url(jenkins_url, job_name, build_no):
    return f"{jenkins_url}job/{job_name}/{build_no}/cobertura/api/json?depth=2"


def get_test_coverage_data(url):
    response = requests.get(url)
    if response.status_code != 200:
        return None
    response = response.json()
    coverage_dict = {}
    for element in response['results']['elements']:
        coverage_dict[element['name']] = element['ratio']

    return coverage_dict


def get_build_data_dict(build, jenkins_url, job_name, board_repo_name, board_id):
    build_details_dict = {}
    test_url = get_test_coverage_url(jenkins_url, job_name, build.get_number())
    build_details_dict['build_no'] = build.get_number()

    build_details_dict['org_name'] = board_repo_name[board_id][0]
    build_details_dict['repository_name'] = board_repo_name[board_id][1]
    build_details_dict['board_id'] = board_id
    build_details_dict['job_name'] = job_name

    build_details_dict['status'] = build.get_status()

    build_details_dict['jenkins_url'] = jenkins_url

    build_details_dict['revision'] = build.get_revision()
    build_details_dict['test_coverage'] = get_test_coverage_data(test_url)
    build_details_dict['date'] = build.get_timestamp()
    return build_details_dict


def collect_jenkins_data(board_id, jenkins_address, job, repo_org_name, repo_name):
    client = MongoClient('mongodb://localhost:27017/')
    db = client.poc_teradata

    board_jenkins = {
        board_id: (jenkins_address, job),
    }

    board_repo_name = {
        board_id: (repo_org_name, repo_name),
    }
    build_dict_list = []
    for board_id, (jenkins_url, job_name) in board_jenkins.items():
        J = Jenkins(jenkins_url)
        first_build_number = J[job_name].get_first_build().get_number()
        last_build_number = J[job_name].get_last_completed_buildnumber()
        all_builds = get_build_data(first_build_number, last_build_number + 1, J, job_name)

        for build in all_builds:
            jenkins_dict = get_build_data_dict(build, jenkins_url, job_name, board_repo_name, board_id)
            build_dict_list.append(jenkins_dict)

    mg_builds = db.builds
    mg_builds.insert_many(build_dict_list)



