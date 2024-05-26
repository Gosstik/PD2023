ADD JAR /opt/cloudera/parcels/CDH/lib/hive/lib/hive-contrib.jar;
ADD jar /opt/cloudera/parcels/CDH/lib/hive/lib/hive-serde.jar;
SET hive.exec.dynamic.partition.mode=nonstrict;
SET hive.exec.max.dynamic.partitions=200;
SET hive.exec.max.dynamic.partitions.pernode=200;
-- USE pd2023a024_test;
USE vashkevicheg;

--------------------------------------------------------------------------------

DROP TABLE IF EXISTS LogsRaw;

CREATE EXTERNAL TABLE
LogsRaw
(
    ip STRING,
    q_time_str STRING, -- STRING is more appropriate to be able to cast it to TIMESTAMP later
    http_query STRING,
    page_size SMALLINT,
    http_status SMALLINT,
    browser STRING
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.RegexSerDe'
WITH SERDEPROPERTIES (
    "input.regex" = '^(\\S*)\\t{3}([0-9]{8})[0-9]{6}\\t(\\S*)\\t(\\S*)\\t(\\S*)\\t(\\S*).*$'
)
LOCATION '/data/user_logs/user_logs_M';

-- select timestamp(regexp_replace(q_time_str,'(\\d{4})(\\d{2})(\\d{2})(\\d{2})(\\d{2})(\\d{2})','$1-$2-$3 $4:$5:$6')) as q_time

--------------------------------------------------------------------------------

DROP TABLE IF EXISTS Logs;

CREATE EXTERNAL TABLE Logs (
    ip STRING,
    http_query STRING,
    page_size SMALLINT,
    http_status SMALLINT,
    browser STRING
)
PARTITIONED BY (q_time_str STRING)
STORED AS TEXTFILE;

INSERT OVERWRITE TABLE Logs PARTITION (q_time_str)
SELECT ip, http_query, page_size, http_status, browser, q_time_str
FROM LogsRaw;

--------------------------------------------------------------------------------

DROP TABLE IF EXISTS Users;

CREATE EXTERNAL TABLE
Users
(
    ip STRING,
    browser STRING,
    sex STRING,
    age TINYINT
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY '\t'
LOCATION '/data/user_logs/user_data_M';

--------------------------------------------------------------------------------

DROP TABLE IF EXISTS IPRegions;

CREATE EXTERNAL TABLE
IPRegions
(
    ip STRING,
    region STRING
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY '\t'
LOCATION '/data/user_logs/ip_data_M';

--------------------------------------------------------------------------------

DROP TABLE IF EXISTS Subnets;

CREATE EXTERNAL TABLE
Subnets
(
    ip STRING,
    mask STRING
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY '\t'
LOCATION '/data/subnets/variant1';

--------------------------------------------------------------------------------

