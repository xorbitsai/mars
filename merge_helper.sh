#!/bin/bash
set -e

if git remote -v | grep -q git@github.com:mars-project/mars.git
  then
  echo "Remote 'community' already exists"
else
  echo "Adding git@github.com:mars-project/mars.git as a new remote 'community'"
  git remote add -t master community git@github.com:mars-project/mars.git
fi

if git branch | grep -q community_master
  then
  echo "Checking out to 'community_master'"
  git checkout -q community_master
else
  echo "Creating branch 'community_master'"
  git fetch -q community master
  git checkout -q FETCH_HEAD
  git checkout -q -b community_master
fi
echo "Updating 'community_master'"
git pull -q community master

echo "Checking out to 'master'"
git checkout -q master
echo "Updating 'master'"
git pull -q origin master

echo "Looking for the community commits to merge"
COUNTER=0
for COMMIT_ID in $(git rev-list community_master)
do
  COMMIT_MSG=$(git show -s --pretty=format:%s "${COMMIT_ID}")
  if git log master | grep -q --fixed-strings "${COMMIT_MSG}"
  then
    echo "Last merged community commit was ${COMMIT_ID} ${COMMIT_MSG}"
    LAST_COMMUNITY_COMMIT_ID=${COMMIT_ID}
    break
  fi
  COUNTER=$((COUNTER+1))
done

if [ -z "${LAST_COMMUNITY_COMMIT_ID}" ]
then
  echo "Last merged community commit not found"
  exit 1
fi

echo "Found ${COUNTER} commit(s) to cherry-pick"
for COMMIT_ID in $(git log --reverse --pretty=format:%h "${LAST_COMMUNITY_COMMIT_ID}"..community_master)
do
  echo "Working on $(git --no-pager show -s --oneline ${COMMIT_ID})}"
  while true
  do
    read -r -p "Please input a command (summary/detail/cp/skip): " CMD
    case ${CMD} in
      cp )
        git cherry-pick -x "${COMMIT_ID}"
        break
      ;;
      skip )
        break
      ;;
      summary )
        git show --stat "${COMMIT_ID}"
      ;;
      detail )
        git show "${COMMIT_ID}"
      ;;
    esac
  done
  echo ""
done
