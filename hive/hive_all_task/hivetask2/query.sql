USE vashkevicheg;

SELECT q_time_str, count(http_query) as q_count
FROM Logs
GROUP BY q_time_str
ORDER BY q_count DESC;

