# SVD 모델
class SVDRecommendationModel:
    def __init__(self, vod_log, cont_log, vod_info, user_info, rating_type, factors=100, epochs=20, lr=0.005, reg=0.02):
        self.vod_log = vod_log
        self.cont_log = cont_log
        self.rating_type = rating_type
        self.vod_info = vod_info.astype({'program_id':int})
        self.user_info = user_info
        self.vod_log89, self.test = self.split_month(self.vod_log)
        self.train = self.create_rating_df(self.vod_log89, self.cont_log)
        self.data = self.create_rating_df(self.vod_log, self.cont_log)

        self.train_data= self.dataset(self.train, self.test) # suprise 라이브러리 데이터셋 형태 변환

        self.score_matrix = self.create_score_matrix(self.data)
        self.score_matrix_evaluate = self.create_score_matrix(self.train)
        #
        self.n_factors = factors
        self.n_epochs = epochs
        self.lr_all = lr
        self.reg_all = reg
        self.evaluate_model = self.fit(self.train_data)
        self.precision, self.recall, self.map, self.mar, self.test_diversity, self.user_metrics, self.result_list = self.evaluate(self.train_data, self.test)
        self.model = self.train_model(self.data)

    def split_month(self, vod_log):
        vod_log89 = vod_log[vod_log['month'].isin([8, 9])]
        vod_log10 = vod_log[vod_log['month'].isin([10])]
        vod_log89['program_id'] = vod_log89['program_id'].astype('int')
        vod_log10['program_id'] = vod_log10['program_id'].astype('int')
        return vod_log89, vod_log10

    def create_rating_df(self, vod_log, cont_log):
        vod_log['use_tms_ratio'] = vod_log['use_tms'] / vod_log['disp_rtm_sec']
        vod_log['log_dt'] = pd.to_datetime(vod_log['log_dt'])
        cont_log['log_dt'] = pd.to_datetime(cont_log['log_dt'])
        cont_log['recency'] = (cont_log['log_dt'].max() - cont_log['log_dt']).dt.days # 최근성

        log = pd.concat([vod_log[['subsr_id', 'program_id']], cont_log[['subsr_id', 'program_id']]]).drop_duplicates().reset_index(drop=True)
        log = log.merge(cont_log.groupby(['subsr_id', 'program_id'])[['program_name']].count().reset_index().rename(columns={'program_name':'click_cnt'}), how='left')
        log = log.merge(cont_log.groupby(['subsr_id', 'program_id'])[['recency']].min().reset_index().rename(columns={'recency':'click_recency'}), how='left')

        # (subsr_id, program_id) 쌍에 대해 하나의 평점만 남겨야 함
        # => use_tms_ratio 는 최대값으로 남기고, 시청 count도 해서 추가한다.
        log = log.merge(vod_log.groupby(['subsr_id', 'program_id'])[['use_tms_ratio']].max().reset_index().rename(columns={'use_tms_ratio':'watch_tms_max'}), how='left')
        log = log.merge(vod_log.groupby(['subsr_id', 'program_id'])[['use_tms_ratio']].count().reset_index().rename(columns={'use_tms_ratio':'watch_cnt'}), how='left')

        replace_value = log['click_cnt'].quantile(0.95) # 95% 로 최대값 고정
        log.loc[log[log['click_cnt'] > replace_value].index, 'click_cnt'] = replace_value

        log['click_recency'] = log['click_recency'].max() - log['click_recency']
        replace_value = log['watch_cnt'].quantile(0.95)  # 95% 로 최대값 고정
        log.loc[log[log['watch_cnt'] > replace_value].index, 'watch_cnt'] = replace_value

        if self.rating_type == 1:
            # 시청기록만 사용
            log_rating = log.copy()
            log_rating['rating'] = log_rating['watch_tms_max']
            log_rating.loc[log_rating[log_rating['rating']>0].index, '시청여부'] = 1

        elif self.rating_type == 2:
            log_rating = log.copy()
            log_rating.loc[log_rating[log_rating['watch_cnt'] > 0].index, '시청여부'] = 1
            scaler = MinMaxScaler()
            log_rating.iloc[:, 2:-1] = pd.DataFrame(scaler.fit_transform(log_rating.iloc[:, 2:-1]), columns=log_rating.columns[2:-1]).fillna(0)
            log_rating['rating'] = log_rating['click_cnt'].fillna(0) + log_rating['click_recency'].fillna(0) + log_rating['watch_tms_max'].fillna(0) + log_rating['watch_cnt'].fillna(0)

        elif self.rating_type == 3:
            log_rating = log.copy()
            log_rating.loc[log_rating[log_rating['watch_cnt'] > 0].index, '시청여부'] = 1
            scaler = MinMaxScaler()
            log_rating.iloc[:, 2:-1] = pd.DataFrame(scaler.fit_transform(log_rating.iloc[:, 2:-1]), columns=log_rating.columns[2:-1])
            weight = 0.3
            log_rating['rating'] = (log_rating['click_cnt'].fillna(0) + weight) * (log_rating['click_recency'].fillna(0) + weight) * (log_rating['watch_tms_max'].fillna(0) + weight) * (log_rating['watch_cnt'].fillna(0) + weight)

        elif self.rating_type == 4:
            log_rating = log.copy()
            log_rating.loc[log_rating[log_rating['watch_cnt'] > 0].index, '시청여부'] = 1

            scaler = MinMaxScaler()
            log_rating.iloc[:, 2:-1] = pd.DataFrame(scaler.fit_transform(log_rating.iloc[:, 2:-1]), columns=log_rating.columns[2:-1]).fillna(0)
            pca = PCA(n_components=1, svd_solver='auto') # 전체 feature 수 입력
            pca_result = pca.fit_transform(log_rating.iloc[:, 2:-1])
            log_rating['rating'] = pca_result
            log_rating['rating'] = scaler.fit_transform(log_rating[['rating']])

        rating_df = log_rating[['subsr_id', 'program_id', 'rating', '시청여부']]
        rating_df['program_id'] = rating_df['program_id'].astype('int')
        return rating_df

    def create_score_matrix(self, data):
        df = data.copy()
        df.loc[df[(df['시청여부']==1) & df['rating'].notnull()].index, 'score_matrix'] = 1
        # score_matrix = df.pivot(columns='program_id', index='subsr_id', values='score_matrix')
        score_matrix = pd.DataFrame(np.full((self.data['subsr_id'].max() + 1, self.data['program_id'].max() + 1), np.nan))
        for index, row in df[df['시청여부'].notnull()].iterrows():
            subsr_id = int(row['subsr_id'])
            program_id = int(row['program_id'])
            score = row['시청여부']
            # 빈 행렬에 값 채우기
            score_matrix.loc[subsr_id, program_id] = score
        return score_matrix

    # 데이터셋 만들기
    def dataset(self, train, test):
        # surprise dataset 생성
        reader = Reader(rating_scale=(0, 1))
        train_data = Dataset.load_from_df(train[['subsr_id', 'program_id', 'rating']].dropna(), reader)
        train_data = train_data.build_full_trainset()
        return train_data

    def fit(self, fitting_data):
        model = SVD(random_state=0, n_factors=self.n_factors, n_epochs=self.n_epochs, lr_all=self.lr_all, reg_all=self.reg_all)
        model.fit(fitting_data)
        return model

    def predict(self, subsr_id, program_id, model):
        return model.predict(subsr_id, program_id).est

    def recommend(self, subsr_id, score_matrix, model, N):
        user_rated = score_matrix.loc[subsr_id].dropna().index.tolist()
        user_unrated = score_matrix.loc[subsr_id].drop(user_rated).index.tolist()
        predictions = [self.predict(subsr_id, program_id, model) for program_id in user_unrated]
        result = pd.DataFrame({'program_id': user_unrated, 'pred_rating': predictions})
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

    def evaluate(self, train_data, test, N=10):
        # fitting,
        #evaluate_model = self.fit(train_data)
        result = pd.DataFrame()
        precisions = []
        recalls = []
        map_values = []
        mar_values = []
        user_metrics = []

        for idx, user in enumerate(tqdm(test['subsr_id'].unique())):
            targets = test[test['subsr_id']==user]['program_id'].unique()
            predictions = self.recommend(user, self.score_matrix_evaluate, self.evaluate_model, N)['program_id'].values
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
        return np.mean(precisions), np.mean(recalls), np.mean(map_values), np.mean(mar_values), 1-list_similarity, pd.DataFrame(user_metrics), result

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
        reader = Reader(rating_scale=(0, 1))
        train_data = Dataset.load_from_df(data[['subsr_id', 'program_id', 'rating']].dropna(), reader)
        train_data = train_data.build_full_trainset()
        model = self.fit(train_data)
        return model