# models.py
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func
from db_setup import Base

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password = Column(String)

class SearchHistory(Base):
    __tablename__ = "search_history"
    id = Column(Integer, primary_key=True)
    username = Column(String)
    query = Column(String)
    timestamp = Column(DateTime, default=func.now())

class Feedback(Base):
    __tablename__ = "feedback"
    id = Column(Integer, primary_key=True)
    username = Column(String)
    product_id = Column(String)
    rating = Column(String)
    timestamp = Column(DateTime, default=func.now())
