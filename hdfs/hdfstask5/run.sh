
#! /usr/bin/env bash

real_size="${1}"

################################################################################

# create file with unique name
test_filename_base=".test_filename"
one="1"
status="0"
counter="0"

while [[ "${status}" = "0" ]] && [[ "${counter}" -le "10" ]]; do
    test_filename="${test_filename_base}${counter}"
    hdfs dfs -test -e "${test_filename}"
    status="$?"
    counter="$(( counter + one ))"
done

if [[ "${status}" = "0" ]]; then
   echo "unable to create file, exiting"
   exit
fi

################################################################################

# main logic
dd if=/dev/zero of="${test_filename}"  bs="${real_size}" count=1
hdfs dfs -put "${test_filename}" "${test_filename}"
blk_ids=$(hdfs fsck "/user/${USER}/${test_filename}" -files -blocks -locations | grep -Eo "blk_[0-9]+")

#echo "IDS FOUND:"
#echo "${ids}"

server_size=0
regex="^Block replica on datanode/rack: (.*)/.* is HEALTHY\$"

for blk_id in ${blk_ids}; do
  line_ip=$(hdfs fsck -blockId "$blk_id" | grep -Eom 1 "${regex}")
  [[ ${line_ip} =~ ${regex} ]]
  server="${BASH_REMATCH[1]}"

  cmd='path=$(find / -name '${blk_id}' 2>/dev/null); du -b $path | awk '"'{print \$1}'"
  cur_size=$(sudo -u hdfsuser ssh -l hdfsuser "${server}" "${cmd}")

  server_size="$(( server_size + cur_size ))"
done

echo "$(( server_size - real_size ))"

hdfs dfs -rm "${test_filename}"
