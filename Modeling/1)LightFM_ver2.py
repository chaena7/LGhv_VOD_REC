# LightFM 모델 2 - 메타 정보 추가
class LightFM_Model_ver2:
    def __init__(self, vod_log, cont_log, vod_info, user_info, rating_type, param_loss='logistic', param_components=10, param_epochs=10, param_learning_schedule='adagrad', param_learning_rate=0.05):
        self.vod_log = vod_log
        self.cont_log = cont_log
        self.rating_type = rating_type
        self.vod_info = vod_info.astype({'program_id':int})
        self.user_info = user_info

        # 8,9월을 학습 데이터로, 10월을 테스트 데이터로 사용
        self.vod_log89, self.test = self.split_month(self.vod_log)
        self.train = self.create_rating_df(self.vod_log89, self.cont_log)
        self.data = self.create_rating_df(self.vod_log, self.cont_log)

        self.train_interactions, self.train_weights, self.item_features_dataset, self.user_features_dataset, self.item_features_col, self.user_features_col= self.dataset(self.train)
        self.score_matrix = self.create_score_matrix(self.data)
        self.score_matrix_evaluate = self.create_score_matrix(self.train)

        self.loss = param_loss
        self.components = param_components
        self.epoch = param_epochs
        self.learning_schedule = param_learning_schedule
        self.learning_rate = param_learning_rate

        # 성능 평가 진행
        self.evaluate_model = self.fit(self.train_interactions, self.train_weights, self.item_features_dataset, self.user_features_dataset)
        self.precision, self.recall, self.map, self.mar, self.test_diversity, self.user_metrics, self.result_list = self.evaluate(self.train_interactions, self.train_weights, self.test)

        # 전체 데이터로 최종 학습 진행
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
        score_matrix = pd.DataFrame(np.full((self.data['subsr_id'].max() + 1, self.data['program_id'].max() + 1), np.nan))
        for index, row in df[df['시청여부'].notnull()].iterrows():
            subsr_id = int(row['subsr_id'])
            program_id = int(row['program_id'])
            score = row['시청여부']
            score_matrix.loc[subsr_id, program_id] = score
        return score_matrix

    # Item Feature 와 User Feature 추가하기 (기본 LightFM 모델 1에서 메타 정보 추가)
    def dataset(self, train):
        # item feature 준비
        item_features = self.vod_info[['program_id', 'program_genre']]
        item_features['program_genre'] = [x.split(', ') for x in item_features['program_genre']]
        item_features = pd.concat([item_features.iloc[:, 0], item_features['program_genre'].str.join('|').str.get_dummies()], axis=1)
        item_features_col = item_features.drop(columns=['program_id']).columns.values
        item_feat = item_features.drop(columns=['program_id']).to_dict(orient='records')

        # user feature 준비
        user_features = self.user_info
        user_features_col = user_features.drop('subsr_id', axis=1).columns.values
        user_feat = user_features.drop(columns=['subsr_id']).to_dict(orient='records')

        # dataset 생성
        dataset = Dataset()
        dataset.fit(users = self.data['subsr_id'].sort_values().unique(),
                    items = self.data['program_id'].sort_values().unique(),
                    item_features = item_features_col,
                    user_features = user_features_col)

        #num_users, num_vods = dataset.interactions_shape()
        train['interaction'] = train['rating'].apply(lambda x: 0 if np.isnan(x) else 1)
        train_dropna = train.dropna()
        train_interactions = coo_matrix((train_dropna['interaction'], (train_dropna['subsr_id'], train_dropna['program_id'])), shape=(self.data['subsr_id'].max() + 1, self.data['program_id'].max() +1))
        train_weights = coo_matrix((train_dropna['rating'], (train_dropna['subsr_id'], train_dropna['program_id'])), shape=(self.data['subsr_id'].max() + 1, self.data['program_id'].max() +1))
        item_features_dataset = dataset.build_item_features((x, y) for x, y in zip(item_features['program_id'], item_feat))
        user_features_dataset = dataset.build_user_features((x, y) for x, y in zip(user_features['subsr_id'], user_feat))

        return train_interactions, train_weights, item_features_dataset, user_features_dataset, item_features_col, user_features_col

    def fit(self, fitting_interactions, fitting_weights, item_features_dataset, user_features_dataset):
        model = LightFM(random_state=0, loss=self.loss, no_components=self.components)
        model.fit(interactions=fitting_interactions, sample_weight=fitting_weights
                  ,item_features=item_features_dataset, user_features=user_features_dataset, verbose=1, epochs=self.epoch)
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
        evaluate_model = self.fit(train_interactions, train_weights, self.item_features_dataset, self.user_features_dataset)
        result = pd.DataFrame()
        precisions = []
        recalls = []
        map_values = []
        mar_values = []
        user_metrics = []

        for idx, user in enumerate(tqdm(test['subsr_id'].unique())):
            targets = test[test['subsr_id']==user]['program_id'].unique()
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
        # 최종 학습 데이터셋 만들기
        train_interactions, train_weights, item_features_dataset, user_features_dataset, item_features_col, user_features_col = self.dataset(data)
        model = self.fit(train_interactions, train_weights, item_features_dataset, user_features_dataset)
        return model