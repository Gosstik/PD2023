#! /usr/bin/env bash

OUT_DIR="vashkevich_mapreduce_task1"
NUM_REDUCERS=8

hdfs dfs -rm -r -skipTrash "${OUT_DIR}" > /dev/null

yarn jar /opt/cloudera/parcels/CDH/lib/hadoop-mapreduce/hadoop-streaming.jar \
    -D mapred.job.name="vashkevich_mapreduce_task1" \
    -D mapreduce.job.reduces="${NUM_REDUCERS}" \
    -files mapper.py,reducer.py \
    -mapper mapper.py \
    -reducer reducer.py \
    -input /data/ids \
    -output "${OUT_DIR}" > /dev/null

cnt=0
for num in $( seq 0 $((NUM_REDUCERS - 1)) )
do
    if [[ ${cnt} -ge 50 ]]; then
      break;
    fi

    sub=$(hdfs dfs -cat "${OUT_DIR}/part-0000${num}" | wc -l)

    if [[ $(( cnt + sub )) -gt 50 ]]; then
      sub=$(( 50 - cnt))
    fi

    hdfs dfs -cat "${OUT_DIR}/part-0000${num}" | head -n ${sub}

    cnt=$((cnt + sub))
done
