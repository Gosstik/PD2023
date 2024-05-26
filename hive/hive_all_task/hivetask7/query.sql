add jar GetHostIpAddress/jar/GetHostIpAddress.jar;

USE vashkevicheg;

create temporary function get_host_ip_address as 'hw.mipt.GetHostIpAddress';

select get_host_ip_address(ip, mask)
from Subnets
limit 100;
