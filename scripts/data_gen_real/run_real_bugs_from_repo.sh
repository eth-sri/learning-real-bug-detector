#!/bin/bash

TASK=${1}

OUT_DIR=../../../real-bugs/${TASK}
mkdir -p ${OUT_DIR}

parallel -a ../../../repos/all_py150_repos.txt --jobs 30 --result ${OUT_DIR} --joblog ${OUT_DIR}/joblog.txt --colsep ' ' --timeout 3600 \
"python real_bugs_from_repo.py --task ${TASK} --user {1} --repo {2}"