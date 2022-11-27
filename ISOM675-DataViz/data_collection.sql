select *
from project
limit 5;

select *
from location l
limit 5;


select p.id
,p.name
,c.name as category 
,c1.name as main_category
,currency 
,DATE(deadline, 'unixepoch') as deadline 
,goal
,DATE(launched_at , 'unixepoch') as lanuched
,pledged
,p.state 
,backers_count 
,l.name as location
,l.state as `location_state`
,l.`type` as location_type
,usd_pledged 
,static_usd_rate * usd_pledged as usd_pledged_real
from project p 
left join category c 
on p.category_id = c.id 
left join category c1
on c.parent_id = c1.id 
left join location l 
on p.location_id = l.id 
where p.country = 'US';