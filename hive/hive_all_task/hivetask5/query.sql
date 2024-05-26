ADD JAR /opt/cloudera/parcels/CDH/lib/hive/lib/hive-contrib.jar;
ADD FILE ./change_domain.sh;

USE vashkevicheg;

with domain_table as (
    select TRANSFORM(http_query) USING './change_domain.sh' AS http_query_com
    from Logs
)
SELECT
    ip,
    q_time_str,
    http_query_com,
    page_size,
    http_status,
    browser
FROM Logs, domain_table
LIMIT 10;
