USE vashkevicheg;

select region,
       count(case when sex='male' then 1 end) as male_cnt,
       count(case when sex='female' then 1 end) as female_cnt
from Users as u left join IPRegions as ipr on u.ip=ipr.ip
where region is not NULL
group by region;
