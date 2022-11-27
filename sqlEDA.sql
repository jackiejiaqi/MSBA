#execute the follwing line, and move data to response address to ensure importing process
show variables like '%secure%';

# DROP table simplified_data;
# SELECT * FROM bigdatahw2.simplified_data;
CREATE TABLE `bigdatahw2`.`simplified_data`
 (`id` int, 
 `state` int, 
 `goal` double, 
 `country` text, 
 `usd_pledged` double, 
 `backers_count` int, 
 `comments_count` int, 
 `updates_count` int, 
 `project_duration` int, 
 `profile_state` text, 
 `p_category` text);

LOAD DATA INFILE "C:\\ProgramData\\MySQL\\MySQL Server 8.0\\Uploads\\Simplified_data.csv"
INTO TABLE simplified_data
FIELDS TERMINATED BY ',' 
OPTIONALLY ENCLOSED BY '\"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS; 


-- # total project counts by country,decending order
-- SELECT count(id) as project_count, country 
-- FROM simplified_data
-- GROUP BY country
-- ORDER BY project_count DESC; 

# total successful project counts by country,decending order
SELECT count(id) as project_count, country 
FROM simplified_data
WHERE state=1
GROUP BY country
ORDER BY project_count DESC; 

# total successful project counts by category,decending order
SELECT count(id) as project_count, p_category as category 
FROM simplified_data
WHERE state=1
GROUP BY p_category
ORDER BY project_count DESC; 

#avg amount pledged by country
SELECT ROUND(AVG(usd_pledged),2)as avg_amount, country 
FROM simplified_data
GROUP BY country
ORDER BY avg_amount DESC; 

-- #total amount pledged by country
-- SELECT ROUND(SUM(usd_pledged),2)as avg_amount, country 
-- FROM simplified_data
-- GROUP BY country
-- ORDER BY avg_amount DESC; 

#avg amount pledged by category
SELECT ROUND(AVG(usd_pledged),2)as avg_amount, p_category as category
FROM simplified_data
GROUP BY p_category
ORDER BY avg_amount DESC; 

#try to calculate success rate-failed
-- select (select count(state) as successful
-- from simplified_data
-- WHERE state=1
-- GROUP BY country) /count(id) 
-- as success_rate 
-- from simplified_data
-- GROUP BY country;


-- select count(state)/count(id) as success_rate 
-- from simplified_data
-- WHERE state=1
-- GROUP BY country;

-- select avg(backers_count) as avg
-- from simplified_data;

-- select backers_count as avg
-- from simplified_data
-- ORDER BY backers_count ASC; 