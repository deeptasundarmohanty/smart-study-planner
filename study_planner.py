"""
Smart Study Session Planner
============================
Uses Supervised Learning (Linear Regression) to predict recommended study hours
and Unsupervised Learning (KMeans Clustering) to group subjects by priority.

Course: Fundamentals of AI and ML
Project: BYOP – Smart Study Session Planner
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")
