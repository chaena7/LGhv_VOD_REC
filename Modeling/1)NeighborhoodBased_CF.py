# User Based CF
class UserBasedCollaborativeFiltering:
    def __init__(self, train_data, k=5, N=10):
        self.k = k
        self.N = N
        self.score_matrix = train_data.pivot(columns='program_id', index='subsr_id', values='use_tms_ratio')
        self.user_similarity = pd.DataFrame(cosine_similarity(self.score_matrix.fillna(0), self.score_matrix.fillna(0)),
                                            index=self.score_matrix.index, columns=self.score_matrix.index)

    def predict(self, subsr_id, program_id):
        sim_user_similarity = self.user_similarity[[subsr_id]].sort_values(by=subsr_id, ascending=False)[1:self.k+1]
        sim_user = sim_user_similarity.index.tolist()
        sim_user_rating_vod = self.score_matrix[[program_id]].loc[sim_user]
        pred = np.dot(np.array(sim_user_rating_vod.fillna(0).values.flatten()), sim_user_similarity.values.flatten()) / sim_user_similarity.values.sum()
        return pred

    def recommend(self, subsr_id):
        user_rated = self.score_matrix.loc[subsr_id].dropna().index.tolist()
        user_unrated = self.score_matrix.loc[subsr_id].drop(user_rated).index.tolist()
        predictions = [self.predict(subsr_id, program_id) for program_id in user_unrated]
        result = pd.DataFrame({'program_id': user_unrated, 'pred_rating': predictions})
        top_N = result.sort_values(by='pred_rating', ascending=False)[:self.N]
        return top_N

    @staticmethod
    def calculate_ranking(target, prediction):
        num_hit = len(set(prediction).intersection(set(target)))
        precision = float(num_hit) / len(prediction) if len(prediction) > 0 else 0.0
        recall = float(num_hit) / len(target) if len(target) > 0 else 0.0
        return precision, recall

    def evaluate(self, test_data):
        precisions = []
        recalls = []

        for user in tqdm(test_data['subsr_id'].unique()):
            targets = test_data[test_data['subsr_id'] == user]['program_id'].values
            predictions = self.recommend(user)['program_id'].values
            precision, recall = self.calculate_ranking(targets, predictions)
            precisions.append(precision)
            recalls.append(recall)

        return np.mean(precisions), np.mean(recalls)

    def calculate_rmse(self, test_data):
        for idx in test_data.index:
            subsr_id = test_data.loc[idx, 'subsr_id']
            program_id = test_data.loc[idx, 'program_id']
            test_data.loc[idx, 'pred'] = self.predict(subsr_id, program_id)

        mse = mean_squared_error(test_data['use_tms_ratio'], test_data['pred'].fillna(0))
        rmse = np.sqrt(mse)
        return test_data, rmse


# Item Based CF
class ItemBasedCollaborativeFiltering:
    def __init__(self, train_data, k=5, N=10):
        self.k = k
        self.N = N
        self.score_matrix = train_data.pivot(columns='subsr_id', index='program_id', values='use_tms_ratio')
        self.item_similarity = pd.DataFrame(cosine_similarity(self.score_matrix.fillna(0), self.score_matrix.fillna(0)),
                                            index=self.score_matrix.index, columns=self.score_matrix.index)

    # (User u, item i) 에 대한 rating pred 함수
    def predict(self, subsr_id, program_id):
        sim_item_similarity = self.item_similarity[[program_id]].sort_values(by=program_id, ascending=False)[1:self.k+1]
        sim_item = sim_item_similarity.index.tolist()
        sim_item_rating_vod = self.score_matrix[[subsr_id]].loc[sim_item]
        pred = np.dot(np.array(sim_item_rating_vod.fillna(0).values.flatten()), sim_item_similarity.values.flatten()) / sim_item_similarity.values.sum()
        return pred

    # User u에 대한 recommend 결과 생성
    def recommend(self, subsr_id):
        user_rated = self.score_matrix[subsr_id].dropna().index.tolist()
        user_unrated = self.score_matrix[subsr_id].drop(user_rated).index.tolist()
        predictions = [self.predict(subsr_id, program_id) for program_id in user_unrated]
        result = pd.DataFrame({'program_id': user_unrated, 'pred_rating': predictions})
        top_N = result.sort_values(by='pred_rating', ascending=False)[:self.N]
        return top_N['program_id'].values

    @staticmethod
    # 한 사람에 대한 성능 측정
    def calculate_ranking(target, prediction):
        num_hit = len(set(prediction).intersection(set(target)))
        precision = float(num_hit) / len(prediction) if len(prediction) > 0 else 0.0
        recall = float(num_hit) / len(target) if len(target) > 0 else 0.0
        return precision, recall4

    # 전체 성능 측정
    def evaluate(self, test_data):
        precisions = []
        recalls = []

        for user in tqdm(test_data['subsr_id'].unique()):
            targets = test_data[test_data['subsr_id'] == user]['program_id'].values
            predictions = self.recommend(user)
            precision, recall = self.calculate_ranking(targets, predictions)
            precisions.append(precision)
            recalls.append(recall)

        return np.mean(precisions), np.mean(recalls)

    def calculate_rmse(self, test_data):
        for idx in test_data.index:
            subsr_id = test_data.loc[idx, 'subsr_id']
            program_id = test_data.loc[idx, 'program_id']
            test_data.loc[idx, 'pred'] = self.predict(subsr_id, program_id)

        mse = mean_squared_error(test_data['use_tms_ratio'], test_data['pred'].fillna(0))
        rmse = np.sqrt(mse)
        return test_data, rmse