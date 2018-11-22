import praw
import requests
import json
import time
from requests.exceptions import RequestException
import os
from PIL import Image
from io import BytesIO

with open('secret_file.json') as json_data:
    secrets  = json.load(json_data)


ua = 'opioid research project 0.1 by StandardClassroom3'
secret = secrets['secret']
passwd = secrets['passwd']
client_id = secrets['client_id']

reddit = praw.Reddit(client_id = client_id,
                     client_secret = secret,
                     password = passwd,
                     user_agent = ua,
                     username = 'StandardClassroom3')

##example
default_subreddits = reddit.subreddit('popular')
seen_posts = []
if os.path.isfile('./default_seen_posts.json'):
    with open('./default_seen_posts.json') as j_file:
        seen_posts = json.load(j_file)

from sqlalchemy import create_engine
engine = create_engine('sqlite:///default_text.db', echo=True)
import sqlorm

if not os.path.isfile('./default_text.db'):
    sqlorm.Base.metadata.create_all(engine)

from sqlalchemy.orm import sessionmaker
Session = sessionmaker(bind=engine)
session  = Session() 

out_dir = './default_photos'


while True:
    print('sampling')
    #sample from heroin_reddits
    submissions = default_subreddits.hot(limit=200)
    for s in submissions:
        if s.id not in seen_posts:
            if s.is_self:
                try:
                    post = sqlorm.Reddit_Self_Post(post_id=s.id,
                                                   title = s.title,
                                                   user_name=s.author.name,
                                                   post_text = s.selftext) 
                except AttributeError:
                    print('author error?')
                    print(s.permalink)
                    post = sqlorm.Reddit_Self_Post(post_id=s.id,
                                                   title = s.title,
                                                   user_name = 'None',
                                                   post_text = s.selftext)
                    
                session.add(post)
                seen_posts.append(s.id)
            else:
                try:
                    r = requests.get(s.url)
                except RequestException as e:
                    print('There was a request exception; {}'.format(e))
                    print('Treating this url as bad')
                    seen_posts.append(s.id)
                    continue

                if r.status_code == 200:
                    if 'image' in r.headers['Content-Type']:
                        fileName = '{}/{}_{}.jpg'.format(out_dir,s.id,s.author.name if s.author is not None else 'None')
                        print(fileName,s.url)
                        image = Image.open(BytesIO(r.content))
                        
                        image = image.convert(mode='RGB')
                        image.save(fileName)
                        seen_posts.append(s.id)

        else:
            print('{} post has been seen already'.format(s.id))
    #buffer maintenance    
    session.commit()
    with open('./default_seen_posts.json','w') as j_file:
        json.dump(seen_posts,j_file)
    
    for c in default_subreddits.comments(limit=200):
        if c.id not in seen_posts:
            try:
                comment = sqlorm.Reddit_Comment(post_id = c.id,
                                                user_name = c.author.name,
                                                post_text = c.body)
            except AttributeError:
                print('attribute_error {}'.format(c.permalink))
                comment = sqlorm.Reddit_Comment(post_id = c.id,
                                                user_name = 'None',
                                                post_text = c.body)
            session.add(comment)
            seen_posts.append(c.id)
        else:
            print('{} comment has been seen already'.format(c.id))

    session.commit()
    with open('./default_seen_posts.json','w') as j_file:
        json.dump(seen_posts,j_file)    

    minutes = 10
    print('sleep for {}minutes'.format(minutes))
    time.sleep(minutes*60)
