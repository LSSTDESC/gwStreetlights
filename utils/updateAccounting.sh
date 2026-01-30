#!/bin/bash

# Usage Check
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <username> <old_account> <new_account>"
    echo "Example: $0 jdoe engineering research"
    exit 1
fi

TARGET_USER=$1
OLD_ACCT=$2
NEW_ACCT=$3

echo "Searching for PENDING jobs for user '$TARGET_USER' in account '$OLD_ACCT'..."

# 1. Query squeue for jobs
# -u: Filter by user
# -t PD: Filter by state (Pending)
# -h: No header (data only)
# -o "%A %a": Output format "JobID Account"
jobs_found=0

while read -r job_id current_acct; do
    # 2. Check if the job matches the "old" account group
    if [[ "$current_acct" == "$OLD_ACCT" ]]; then
        echo "Updating Job ID $job_id: Changing account from $OLD_ACCT to $NEW_ACCT"
        
        # 3. Update the job
        # Capture output to check for errors (e.g., permission denied)
        output=$(scontrol update JobId="$job_id" Account="$NEW_ACCT" 2>&1)
        
        if [[ $? -eq 0 ]]; then
            echo "  -> Success"
        else
            echo "  -> Failed: $output"
        fi
        ((jobs_found++))
    fi
done < <(squeue -u "$TARGET_USER" -t PD -h -o "%A %a")

if [[ "$jobs_found" -eq 0 ]]; then
    echo "No matching pending jobs found."
else
    echo "Finished. Updated $jobs_found jobs."
fi
