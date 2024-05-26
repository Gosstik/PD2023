add jar ReverseString/jar/ReverseString.jar;

USE vashkevicheg;

create temporary function reverse_string as 'hw.mipt.ReverseString';

select reverse_string(ip)
from Subnets
limit 10;
