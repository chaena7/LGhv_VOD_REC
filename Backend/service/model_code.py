import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from lightfm import LightFM, cross_validation
from lightfm.data import Dataset
from datetime import timedelta

from scipy.sparse import coo_matrix


class GenreBasedRecommendationModel:
    def __init__(self, data, vod_info, user_info):
        self.train, self.test = self.split_evaluate(data)
        self.vod_info = vod_info
        self.user_info = user_info
        self.kids_program_list = self.kids_program(self.vod_info)
        self.genre_vector, self.genre_similarity_matrix = self.calculate_genre_similarity(self.vod_info)
        self.score_matrix = self.create_score_matrix(data)
        self.score_matrix_evaluate = self.create_score_matrix(self.train)
        # self.precision, self.recall, self.map, self.mar, self.test_diversity, self.user_metrics  = self.evaluate(self.test)
        # self.all_diversity = self.evaluate_all()

    def split_evaluate(self, data):
        train, test = train_test_split(data.dropna(), test_size=0.25, random_state=0)
        train = data.copy()
        train.loc[test.index, 'rating'] = np.nan
        return train, test

    def kids_program(self, vod_info):
        kids_program_list = vod_info[vod_info['ct_cl']=='키즈']['program_id'].values.tolist()
        return kids_program_list

    def create_score_matrix(self, data):
        score_matrix = data.pivot(columns='program_id', index='subsr_id', values='rating')
        return score_matrix

    def calculate_genre_similarity(self, vod_info):
        vod_genre = vod_info.copy()
        vod_genre['program_genre'] = vod_genre['program_genre'].str.split(', ')
        genre_encoding = vod_genre['program_genre'].str.join('|').str.get_dummies()
        genre_vector = pd.concat([vod_genre[['program_id']], genre_encoding], axis=1)
        genre_vector = genre_vector.set_index('program_id')
        genre_similarity = pd.DataFrame(cosine_similarity(genre_vector, genre_vector), index=genre_vector.index, columns=genre_vector.index)
        return genre_vector, genre_similarity

    # 특정 subsr_id 에 대한 top_N 추천 결과
    def recommend(self, subsr_id, score_matrix, kids=0, N=10):
        if kids == 0:
            program_id = score_matrix.loc[subsr_id].sort_values(ascending=False).dropna().index.tolist()[0]
            seen_list =  score_matrix.loc[subsr_id].dropna().index.tolist()
            unseen_list = score_matrix.columns.drop(seen_list).tolist()
            ranking = self.vod_info.merge(self.genre_similarity_matrix.loc[[program_id]][unseen_list].T.sort_values(by=program_id, ascending=False)[:N], how='right', on='program_id')
            #top_N = ranking.sort_values(by=['program_id', '상세보기조회수(관심도)'], ascending=False)[:N]
            top_N = ranking.sort_values(by=[program_id,'release_date', 'click_cnt'], ascending=False)[:N]
            return top_N

        if kids == 1:
            program_id_list = score_matrix.loc[subsr_id].sort_values(ascending=False).dropna().index.tolist()
            # 유사도 측정 대상 vod에 키즈 vod 안포함하도록
            program_id_except_kids = [x for x in program_id_list if x not in self.kids_program_list][0]
            
            seen_list = score_matrix.loc[subsr_id].dropna().index.tolist()
            seen_list_include_kids = list(set(seen_list + self.kids_program_list))
            unseen_list = score_matrix.columns.drop(seen_list).tolist()
            ranking = self.vod_info.merge(self.genre_similarity_matrix.loc[[program_id_except_kids]][unseen_list].T.sort_values(by=program_id_except_kids, ascending=False)[:N], how='right', on='program_id')
            top_N = ranking.sort_values(by=[program_id_except_kids,'release_date', 'click_cnt'], ascending=False)[:N]
            return top_N


    @staticmethod
    def precision_recall_at_k(target, prediction):
        num_hit = len(set(prediction).intersection(set(target)))
        precision = float(num_hit) / len(prediction) if len(prediction) > 0 else 0.0
        recall = float(num_hit) / len(target) if len(target) > 0 else 0.0
        return precision, recall

    @staticmethod
    def map_at_k(target, prediction, k=10):
        num_hits = 0
        precision_at_k = 0.0
        for i, p in enumerate(prediction[:k]):
            if p in target:
                num_hits += 1
                precision_at_k += num_hits / (i + 1)
        if not target.any():
            return 0.0
        return precision_at_k / min(k, len(target))

    @staticmethod
    def mar_at_k(target, prediction, k=10):
        num_hits = 0
        recall_at_k = 0.0
        for i, p in enumerate(prediction[:k]):
            if p in target:
                num_hits += 1
                recall_at_k += num_hits / len(target)
        if not target.any():
            return 0.0
        return recall_at_k / min(k, len(target))

    def evaluate(self, test):
        result = pd.DataFrame()
        precisions = []
        recalls = []
        map_values = []
        mar_values = []
        user_metrics = []

        for idx, user in enumerate(tqdm(test['subsr_id'].unique())):
            if len(self.score_matrix_evaluate.loc[user].dropna()) == 0:
                continue;
            targets = test[test['subsr_id']==user]['program_id'].values
            predictions = self.recommend(user, self.score_matrix_evaluate)['program_id'].values
            precision, recall = self.precision_recall_at_k(targets, predictions)
            map_at_k_value = self.map_at_k(targets, predictions)
            mar_at_k_value = self.mar_at_k(targets, predictions)
            precisions.append(precision)
            recalls.append(recall)
            map_values.append(map_at_k_value)
            mar_values.append(mar_at_k_value)
            user_metrics.append({'subsr_id': user, 'precision': precision, 'recall': recall, 'map_at_k': map_at_k_value, 'mar_at_k': mar_at_k_value})

            result.loc[idx, 'subsr_id'] = user
            for rank in range(len(predictions)):
                result.loc[idx, f'vod_{rank}'] = predictions[rank]

        list_sim = cosine_similarity(result.iloc[:, 1:])
        list_similarity = np.sum(list_sim - np.eye(len(result))) / (len(result) * (len(result) - 1))

        return np.mean(precisions), np.mean(recalls), np.mean(map_values), np.mean(mar_values), 1-list_similarity, pd.DataFrame(user_metrics)

    def evaluate_all(self):
        result = pd.DataFrame()
        for idx, user in enumerate(tqdm(self.user_info['subsr_id'])):
            if len(self.score_matrix.loc[user].dropna()) == 0:
                continue;
            predictions = self.recommend(user, self.score_matrix)['program_id'].values
            result.loc[idx, 'subsr_id'] = user
            for rank in range(len(predictions)):
                result.loc[idx, f'vod_{rank}'] = predictions[rank]

        list_sim = cosine_similarity(result.iloc[:, 1:])
        list_similarity = np.sum(list_sim - np.eye(len(result))) / (len(result) * (len(result) - 1))
        return 1 - list_similarity

    # 신규 user에 대한 추천 결과
    def new_rec(self, genre_list:list, N=10):
        new = pd.DataFrame(columns=self.genre_vector.columns)
        for genre in genre_list:
            new.loc['user_select', genre] = 1
        new = new.fillna(0)
        new_genre_similarity = pd.DataFrame(cosine_similarity(new, self.genre_vector), columns=self.genre_vector.index, index=new.index)
        ranking = self.vod_info.merge(new_genre_similarity.T.sort_values(by='user_select', ascending=False)[:100], how='right', on='program_id')
        top_N = top_N = ranking.sort_values(by=['user_select','release_date', 'click_cnt'], ascending=False)[:N]
        return top_N


#랭킹 - 주간
def trend_vod(vod_log, date:str, N=100):
    now = pd.to_datetime(date) # 실제 현재시각 반영하면 좋을 듯
    week_ago = now - timedelta(weeks=1)
    weekly_vod = vod_log[pd.to_datetime(vod_log['log_dt']) > week_ago]
    weekly_vod = weekly_vod[['subsr_id', 'program_id', 'program_name', 'episode_num', 'log_dt', 'use_tms', 'disp_rtm_sec']]
    weekly_vod['use_tms_ratio'] = weekly_vod['use_tms'] / weekly_vod['disp_rtm_sec']
    # print(weekly_vod.groupby(['program_id', 'program_name']).sum()[['use_tms_ratio']].reset_index())
    trend_vod =  weekly_vod[['program_id', 'program_name','use_tms_ratio']].groupby(['program_id', 'program_name']).sum()[['use_tms_ratio']].reset_index()
    trend_vod = trend_vod.merge(weekly_vod.groupby(['program_id', 'program_name']).count()[['subsr_id']].reset_index())
    trend_vod = trend_vod.rename(columns={'subsr_id':'cnt_seen'})
    result = trend_vod.sort_values(by=['use_tms_ratio','cnt_seen'], ascending=False)[:N]
    return result
    
class LightFM_Model:
    def __init__(self, data, vod_info, user_info, param_loss='logistic', param_components=10, param_epochs=10):
        self.vod_info = vod_info
        self.user_info = user_info
        self.train, self.test = self.split_evaluate(data)
        self.train_interactions, self.train_weights = self.dataset(self.train)
        self.score_matrix = self.create_score_matrix(data)
        self.score_matrix_evaluate = self.create_score_matrix(self.train)
        ####
        self.loss = param_loss
        self.components = param_components
        self.epoch = param_epochs
        self.precision, self.recall, self.map, self.mar, self.test_diversity, self.user_metrics = self.evaluate(self.train_interactions, self.train_weights, self.test)
        self.model = self.train_model(data)
        self.all_diversity = self.evaluate_all(self.model)

    def split_evaluate(self, data):
        #### 수정 완료
        train, test = train_test_split(data[data['시청여부'].notnull()][['subsr_id', 'program_id', 'rating']], test_size=0.25, random_state=0)
        train = data[['subsr_id', 'program_id', 'rating', '시청여부']].copy()
        train.loc[test.index, 'rating'] = np.nan
        return train, test

    #### 수정 완료
    def create_score_matrix(self, data):
        df = data.copy()
        df.loc[df[(df['시청여부']==1) & df['rating'].notnull()].index, 'score_matrix'] = 1
        score_matrix = df.pivot(columns='program_id', index='subsr_id', values='score_matrix')
        return score_matrix

    def dataset(self, train):
        dataset = Dataset()
        dataset.fit(users = train['subsr_id'].sort_values().unique(),
                    items = train['program_id'].sort_values().unique())
        num_users, num_vods = dataset.interactions_shape()
        ####
        train['interaction'] = train['rating'].apply(lambda x: 0 if np.isnan(x) else 1)
        train_dropna = train.dropna()
        train_interactions = coo_matrix((train_dropna['interaction'], (train_dropna['subsr_id'], train_dropna['program_id'])), shape=(train['subsr_id'].max() + 1, train['program_id'].max() +1))
        train_weights = coo_matrix((train_dropna['rating'], (train_dropna['subsr_id'], train_dropna['program_id'])), shape=(train['subsr_id'].max() + 1, train['program_id'].max() +1))
        #(train_interactions, train_weights) = dataset.build_interactions(train[['subsr_id', 'program_id', 'rating']].dropna().values)
        return train_interactions, train_weights

    def fit(self, fitting_interactions, fitting_weights):
        print('training start')
        model = LightFM(random_state=0, loss=self.loss, no_components=self.components)
        model.fit(interactions=fitting_interactions, sample_weight=fitting_weights, verbose=1, epochs=self.epoch)
        return model

    def predict(self, subsr_id, program_id, model):
        pred = model.predict([subsr_id], [program_id])
        return pred

    def recommend(self, subsr_id, score_matrix, model, N):
        # 안 본 vod 추출
        user_rated = score_matrix.loc[subsr_id].dropna().index.tolist()
        user_unrated = score_matrix.columns.drop(user_rated).tolist()
        # 안본 vod에 대해서 예측하기
        predictions = model.predict(int(subsr_id), user_unrated)
        result = pd.DataFrame({'program_id':user_unrated, 'pred_rating':predictions})
        # pred값에 따른 정렬해서 결과 띄우기
        top_N = result.sort_values(by='pred_rating', ascending=False)[:N]
        return top_N

    @staticmethod
    def precision_recall_at_k(target, prediction):
        num_hit = len(set(prediction).intersection(set(target)))
        precision = float(num_hit) / len(prediction) if len(prediction) > 0 else 0.0
        recall = float(num_hit) / len(target) if len(target) > 0 else 0.0
        return precision, recall

    @staticmethod
    def map_at_k(target, prediction, k=10):
        num_hits = 0
        precision_at_k = 0.0
        for i, p in enumerate(prediction[:k]):
            if p in target:
                num_hits += 1
                precision_at_k += num_hits / (i + 1)
        if not target.any():
            return 0.0
        return precision_at_k / min(k, len(target))

    @staticmethod
    def mar_at_k(target, prediction, k=10):
        num_hits = 0
        recall_at_k = 0.0
        for i, p in enumerate(prediction[:k]):
            if p in target:
                num_hits += 1
                recall_at_k += num_hits / len(target)
        if not target.any():
            return 0.0
        return recall_at_k / min(k, len(target))

    def evaluate(self, train_interactions, train_weights, test, N=10):
        evaluate_model = self.fit(train_interactions, train_weights)
        result = pd.DataFrame()
        precisions = []
        recalls = []
        map_values = []
        mar_values = []
        user_metrics = []

        for idx, user in enumerate(tqdm(test['subsr_id'].unique())):
            targets = test[test['subsr_id']==user]['program_id'].values
            predictions = self.recommend(user, self.score_matrix_evaluate, evaluate_model, N)['program_id'].values
            precision, recall = self.precision_recall_at_k(targets, predictions)
            map_at_k_value = self.map_at_k(targets, predictions)
            mar_at_k_value = self.mar_at_k(targets, predictions)
            precisions.append(precision)
            recalls.append(recall)
            map_values.append(map_at_k_value)
            mar_values.append(mar_at_k_value)
            user_metrics.append({'subsr_id': user, 'precision': precision, 'recall': recall, 'map_at_k': map_at_k_value, 'mar_at_k': mar_at_k_value})

            result.loc[idx, 'subsr_id'] = user
            for rank in range(len(predictions)):
                result.loc[idx, f'vod_{rank}'] = predictions[rank]

        list_sim = cosine_similarity(result.iloc[:, 1:])
        list_similarity = np.sum(list_sim - np.eye(len(result))) / (len(result) * (len(result) - 1))

        return np.mean(precisions), np.mean(recalls), np.mean(map_values), np.mean(mar_values), 1-list_similarity, pd.DataFrame(user_metrics)

    def evaluate_all(self, model, N=10):
        result = pd.DataFrame()
        for idx, user in enumerate(tqdm(self.user_info['subsr_id'])):
            predictions = self.recommend(user, self.score_matrix, model, N)['program_id'].values
            result.loc[idx, 'subsr_id'] = user
            for rank in range(len(predictions)):
                result.loc[idx, f'vod_{rank}'] = predictions[rank]

        list_sim = cosine_similarity(result.iloc[:, 1:])
        list_similarity = np.sum(list_sim - np.eye(len(result))) / (len(result) * (len(result) - 1))
        return 1 - list_similarity


    def train_model(self, data):
        # 최종 학습 데이터셋 만들기
        dataset = Dataset()
        dataset.fit(users = data['subsr_id'].sort_values().unique(),
                    items = data['program_id'].sort_values().unique())
        num_users, num_vods = dataset.interactions_shape()
        ####
        data['interaction'] = data['rating'].apply(lambda x: 0 if np.isnan(x) else 1)
        data_dropna = data.dropna()
        train_interactions = coo_matrix((data_dropna['interaction'], (data_dropna['subsr_id'], data_dropna['program_id'])))
        train_weights = coo_matrix((data_dropna['rating'], (data_dropna['subsr_id'], data_dropna['program_id'])))
        #(train_interactions, train_weights) = dataset.build_interactions(data[['subsr_id', 'program_id', 'rating']].dropna().values)
        # fitting
        model = self.fit(train_interactions, train_weights)
        return model
    
#드라마 몰아보기
def watch_series(data, N = 10):
    result = data.groupby('program_id').agg({'program_id': 'value_counts', 'subsr_id': 'nunique'}).reset_index(names= 'pid')
    result['ratio'] = result['program_id']/ result['subsr_id']

    top_N = result.sort_values(by = 'ratio', ascending= False)[:N].pid

    return top_N