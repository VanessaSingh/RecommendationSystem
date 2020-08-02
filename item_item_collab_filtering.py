import numpy as np
import sys
import pyspark

'''
This code predicts the rating for a given item for all the users. The recommendation system considers only the items and
users left after filtering mentioned in part (a, b, c). 

Output format:
Default output will be saved in / in hdfs. 
For eg:
For the item: '0062082361' the output directory can be listed and printed by:
hadoop fs -ls /
hadoop fs -cat /0062082361/part*
hadoop fs -cat /B00EZPXYP4/part*

The ratings can be directly printed to the screen by uncommenting some lines.
'''

'''
check_sim - calculates the similarities between 2 items - the given item and all the other items
'''


def check_sim(item, vec, chosen_item_map, chosen_item_name):
    count, sim_val = 0, 0
    numer, denom_vec, denom_chosen_item, denom = 0, 0, 0, 0
    nums = len(vec)
    vec_map = {}
    for j in range(nums):
        vec_map[vec[j][0]] = vec[j][1]
    intersecting_users = set(vec_map.keys()).intersection(set(chosen_item_map.keys()))
    # Returning 0 for items that don't satisfy the filter in a)a)ii)
    if len(intersecting_users) < 2:
        return item, 0
    # Returning 0 if item is same as the chosen_item, will get filtered out in the next transformation
    if item == chosen_item_name:
        return item, 0
    # cosine similarity with mean centering
    mean_vec = np.mean(list(vec_map.values()))
    mean_chosen_item = np.mean(list(chosen_item_map.values()))
    for user in intersecting_users:
        numer += (vec_map[user] - mean_vec) * (chosen_item_map[user] - mean_chosen_item)
    for user in vec_map.keys():
        denom_vec += (vec_map[user] - mean_vec) * (vec_map[user] - mean_vec)
    for user in chosen_item_map.keys():
        denom_chosen_item += (chosen_item_map[user] - mean_chosen_item) * (chosen_item_map[user] - mean_chosen_item)
    denom = (np.sqrt(denom_vec) * np.sqrt(denom_chosen_item))
    if denom > 0:
        sim_val = numer / denom
    return item, sim_val


'''
convert - converts tuples to a dictionary
Used to store {user, rating} for the chosen user
Used to store {item, cosine similarity} for all the 50 neighbours
'''


def convert(t, _neigh_dict):
    _neigh_dict.clear()
    for a, b in t:
        _neigh_dict[a] = b
    return _neigh_dict


'''
calculate_rating - calculates rating for the user based on other items rated by the user(these items must be in the 
neighbourhood of the given item). If the user has already rated the given item then it return the original rating
'''


def calculate_rating(user, item_rate_list, chosen_item):
    pred_rating, numer, denom = 0, 0, 0
    # filter in part part b)
    if len(item_rate_list) < 2:
        return user, 0

    for i in range(len(item_rate_list)):
        # If the user has already rated this chosen item before then return that rating
        if item_rate_list[i][0] == chosen_item:
            return user, item_rate_list[i][1]
        # predict new rating
        if item_rate_list[i][0] in neigh_dict.keys():
            sim = neigh_dict[item_rate_list[i][0]]
            numer += (sim * item_rate_list[i][1])
            denom += sim
    if denom > 0:
        pred_rating = numer / denom
    return user, pred_rating


neigh_dict = {}

sc = pyspark.SparkContext.getOrCreate()

given_items = eval(sys.argv[2])
ip = pyspark.SQLContext(sc).read.option("header", "true").json(sys.argv[1])
ip_rdd = ip.select("overall", "reviewerID", "asin").rdd.map(list)
''' 
Filtering - only one rating per user per item
The data is by default in the descending order of review time. Grouping by (item, user) and taking the first rating from 
the list of ratings as that would be the most recent. 
'''
filter_one = ip_rdd.map(lambda x: ((x[2], x[1]), x[0])).combineByKey(lambda x: [x], lambda u, v: u + [v],
                                                                     lambda x, y: x + y)
filtered_data = filter_one.map(lambda x: (x[0][0], x[0][1], list(x[1])[0]))
items_users_group = filtered_data.map(lambda x: (x[0], (x[1], x[2]))).combineByKey(lambda x: [x], lambda u, v: u + [v],
                                                                                   lambda x, y: x + y).map(
    lambda x: (x[0], list(x[1]))).filter(lambda x: len(x[1]) >= 25)
filter_items = set(items_users_group.map(lambda x: x[0]).collect())
# The big dataset might yield more than 1000 items or users that is why not broadcasting the set.

filtered_data_items = filtered_data.filter(lambda x: x[0] in filter_items)
users_items_group = filtered_data_items.map(lambda x: (x[1], (x[0], x[2]))).combineByKey(lambda x: [x],
                                                                                         lambda u, v: u + [v],
                                                                                         lambda x, y: x + y).map(
    lambda x: (x[0], list(x[1]))).filter(lambda x: len(set(x[1])) >= 5)
filter_users = set(users_items_group.map(lambda x: x[0]).collect())

# Data is in the form of item, reviewer, rating
data = filtered_data_items.filter(lambda x: x[1] in filter_users)
# Making a list of (item,[(user, rating), (user, rating)...])
items_users_list = data.map(lambda x: (x[0], (x[1], x[2]))).combineByKey(lambda x: [x], lambda u, v: u + [v],
                                                                         lambda x, y: x + y)

for i in range(len(given_items)):
    chosen_item_vec = items_users_list.filter(lambda x: x[0] == given_items[i]).collect()
    # Making a map for users and their ratings for the given_item
    chosen_vec_map = convert(chosen_item_vec[0][1], dict())
    # Passing this map to each row to compare item-item similarity
    candidates = items_users_list.map(lambda x: check_sim(x[0], list(x[1]), chosen_vec_map, given_items[i]))
    neighbourhood = candidates.filter(lambda x: x[1] > 0).takeOrdered(50, lambda x: -x[1])
    neigh_dict = convert(neighbourhood, neigh_dict)
    # Making a list of (user,[(item, rating), (item, rating)...])
    users_items_list = data.map(lambda x: (x[1], (x[0], x[2]))).combineByKey(lambda x: [x], lambda u, v: u + [v],
                                                                             lambda x, y: x + y)
    user_rating = users_items_list.map(lambda x: calculate_rating(x[0], list(x[1]), given_items[i]))
    fname = "/" + given_items[i]
    user_rating.saveAsTextFile(fname)
    # Uncomment the following lines to print output to the screen
    # ans = user_rating.collect()
    # print("\nITEM : ", given_items[i])
    # print(ans)
