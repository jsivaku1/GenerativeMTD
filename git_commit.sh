#!/bin/bash

# get the argument message
message="$1"

# If no commit message is passed, use current date time in the commit message
if [[ -z "${message// }" ]]
    then
        message=$(date '+%Y-%m-%d %H:%M:%S')
fi

# stage all changes
git add .
echo "====staged all git files"

# add commit
git commit -m "$message"
echo "====added the commit with message: '$message'"

# get current branch and push
current_branch=$(git branch | sed -n -e 's/^\* \(.*\)/\1/p')
git push origin "$current_branch"
echo "====pushed changes to '$current_branch' branch"
