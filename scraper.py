import praw

import json

with open('secret_file.json') as json_data:
    secrets  = json.load(json_data)


ua = 'opioid research project 0.1 by StandardClassroom3'
secret = secrets['secrets']
passwd = secrets['passwd']
client_id = secrets['client_id']

reddit = praw.Reddit(client_id = client_id, 
                     client_secret = secret, 
                     password = passwd, 
                     user_agent = ua,
                     username = 'StandardClassroom3')

##example 
heroin_subreddit = reddit.subreddit('heroin')

for c in heroin_subreddit.comments(limit=25):
    print('author: {} body: {}'.format(c.author,c.body))