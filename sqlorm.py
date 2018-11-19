from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base() 
class Reddit_Self_Post(Base):
    __tablename__='posts'

    pid = Column(Integer,primary_key=True)
    post_id = Column(String)
    user_name = Column(String)
    post_text = Column(String)

    def repr(self): 
        return 'post {} by {} contains {}'.format(self.post_id,self.user_name,self.post_text)

class Reddit_Comment(Base):
    __tablename__='comments'

    pid = Column(Integer,primary_key=True)
    post_id = Column(String)
    user_name = Column(String)
    post_text = Column(String)

    def repr(self): 
        return 'post {} by {} contains {}'.format(self.post_id,self.user_name,self.post_text)