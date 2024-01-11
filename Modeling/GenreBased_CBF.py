# 장르 기반 추천 모델
class GenreBasedRecommendationModel:
    def __init__(self, data, vod_info):
        self.train, self.test = self.split_evaluate(data)
        self.vod_info = vod_info
        self.genre_vector, self.genre_similarity_matrix = self.calculate_genre_similarity(self.vod_info)
        self.score_matrix = self.create_score_matrix(data)
        self.score_matrix_evaluate = self.create_score_matrix(self.train)
        self.precision, self.recall = self.evaluate(self.test)


    def split_evaluate(self, data):
        train, test = train_test_split(data.dropna(), test_size=0.25, random_state=0)
        train = data.copy()
        train.loc[test.index, 'use_tms_ratio'] = np.nan
        return train, test

    def create_score_matrix(self, data):
        score_matrix = data.pivot(columns='program_id', index='subsr_id', values='use_tms_ratio')
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
    def recommend(self, subsr_id, score_matrix, N=10):
        program_id = score_matrix.loc[subsr_id].sort_values(ascending=False).dropna().index.tolist()[0]
        seen_list =  score_matrix.loc[subsr_id].dropna().index.tolist()
        unseen_list = score_matrix.columns.drop(seen_list).tolist()
        ranking = self.vod_info.merge(self.genre_similarity_matrix.loc[[program_id]][unseen_list].T.sort_values(by=program_id, ascending=False)[:100], how='right', on='program_id')
        top_N = ranking.sort_values(by=[program_id, 'click_cnt'], ascending=False)[:N]
        return top_N

    @staticmethod
    def precision_recall_at_k(target, prediction):
        num_hit = len(set(prediction).intersection(set(target)))
        precision = float(num_hit) / len(prediction) if len(prediction) > 0 else 0.0
        recall = float(num_hit) / len(target) if len(target) > 0 else 0.0
        return precision, recall

    def evaluate(self, test):
        precisions = []
        recalls = []
        for user in tqdm(test['subsr_id'].unique()):
            if len(self.score_matrix_evaluate.loc[user].dropna()) == 0:
                continue;
            targets = test[test['subsr_id']==user]['program_id'].values
            predictions = self.recommend(user, self.score_matrix_evaluate)['program_id'].values
            precision, recall = self.precision_recall_at_k(targets, predictions)
            precisions.append(precision)
            recalls.append(recall)
        return np.mean(precisions), np.mean(recalls)

    # 신규 user에 대한 추천 결과
    def new_rec(self, genre_list:list, N=10):
        new = pd.DataFrame(columns=self.genre_vector.columns)
        for genre in genre_list:
            new.loc['user_select', genre] = 1
        new = new.fillna(0)
        new_genre_similarity = pd.DataFrame(cosine_similarity(new, self.genre_vector), columns=self.genre_vector.index, index=new.index)
        ranking = self.vod_info.merge(new_genre_similarity.T.sort_values(by='user_select', ascending=False)[:100], how='right', on='program_id')
        top_N = ranking.sort_values(by=['user_select', 'click_cnt'], ascending=False)[:N]
        return top_N