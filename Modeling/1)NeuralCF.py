def NeuralCF_Model(train, test, vod_info, user_info):
    score_matrix = train.pivot(columns='program_id', index='subsr_id', values='use_tms_ratio')

    # 데이터셋 준비
    def prepare_dataset(train, test):
        # 1)
        train_users = train['subsr_id'].values
        train_items = train['program_id'].values
        train_ratings = train['use_tms_ratio'].fillna(0).values

        # 2)
        # train_users = train.dropna()['subsr_id'].values
        # train_items = train.dropna()['program_id'].values
        # train_ratings = train.dropna()['use_tms_ratio'].values

        test_users = test['subsr_id'].values
        test_items = test['program_id'].values
        test_ratings = test['use_tms_ratio'].values
        return train_users, train_items, train_ratings, test_users, test_items, test_ratings

    # 모델 함수
    def NCF(users_num, items_num, latent_dim_gmf, latent_dim_mlp):
        # User Embedding
        user = Input(shape=(1,), dtype='int32', name='user_input')
        item = Input(shape=(1,), dtype='int32', name='item_input')

        # GMF 쌓기
        user_embedding_gmf = Embedding(users_num, latent_dim_gmf, input_length=user.shape[1])(user)
        item_embedding_gmf = Embedding(items_num, latent_dim_gmf, input_length=item.shape[1])(item)

        user_latent_gmf = Flatten()(user_embedding_gmf)
        item_latent_gmf = Flatten()(item_embedding_gmf)

        # GMF Layer - Embedding한 MF Latent Vector를 내적
        gmf_layer = dot([user_latent_gmf, item_latent_gmf], axes=1)
        # GMF Predict
        gmf_prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform')(gmf_layer)

        # MLP 쌓기
        user_embedding_mlp = Embedding(users_num, latent_dim_mlp, input_length=user.shape[1])(user)
        item_embedding_mlp = Embedding(items_num, latent_dim_mlp, input_length=item.shape[1])(item)

        user_latent_mlp = Flatten()(user_embedding_mlp)
        item_latent_mlp = Flatten()(item_embedding_mlp)

        # Concatenated
        concat_embedding = Concatenate()([user_latent_mlp, item_latent_mlp])

        # FC Layer - MLP
        layer_1 = Dense(units=64, activation='relu', name='layer1')(concat_embedding)
        layer_2 = Dense(units=32, activation='relu', name='layer2')(layer_1)
        layer_3 = Dense(units=16, activation='relu', name='layer3')(layer_2)
        layer_4 = Dense(units=8, activation='relu', name='layer4')(layer_3)

        # MLP Predict
        mlp_prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform')(layer_4)

        # GMF + MLP
        predict_vector = Concatenate()([gmf_prediction, mlp_prediction])

        # output layer
        output_layer = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform')(predict_vector)

        # Model
        model = Model([user, item], output_layer)
        model.compile(optimizer='adam', loss='binary_crossentropy')

        return model


    # 모델 훈련
    def train_model(train_users, train_items, train_ratings, test_users, test_items, test_ratings, user_info, vod_info, latent_dim_gmf=20, latent_dim_mlp=20):
        ncf_model = NCF(len(user_info), len(vod_info), latent_dim_gmf, latent_dim_mlp)
        ncf_model.fit([train_users, train_items], train_ratings, validation_data=([test_users, test_items], test_ratings), epochs=5, verbose=1)
        return ncf_model


    # (user u, item i) 에 대한 선호확률 pred 함수
    def predict(subsr_id, program_id, model):
        pred = model.predict([np.array([subsr_id]), np.array([program_id])])
        return pred

    # user 한 명에 대한 추천 결과 가져오기
    def recommend(subsr_id, model, score_matrix, N):
        # 안 본 vod 추출
        user_rated = score_matrix.loc[subsr_id].dropna().index.tolist()
        user_unrated = score_matrix.columns.drop(user_rated).tolist()

        # 안본 vod에 대해서 예측하기
        predictions = model.predict([np.full((len(user_unrated), ), subsr_id), np.array(user_unrated)], verbose=0).flatten()
        result = pd.DataFrame({'program_id':user_unrated, 'pred_rating':predictions})

        # pred값에 따른 정렬해서 결과 띄우기
        top_N = result.sort_values(by='pred_rating', ascending=False)[:N]
        return top_N

    # user 한명에 대한 precision, recall 측정
    def precision_recall_at_k(target, prediction):
        num_hit = len(set(prediction).intersection(set(target)))
        precision = float(num_hit) / len(prediction) if len(prediction) > 0 else 0.0
        recall = float(num_hit) / len(target) if len(target) > 0 else 0.0
        return precision, recall

    # 성능 평가
    def evaluate(test, model, score_matrix, N=10):
        precisions = []
        recalls = []
        for user in tqdm(test['subsr_id'].unique()):
            targets = test[test['subsr_id']==user]['program_id'].values
            predictions = recommend(user, model, score_matrix, N)['program_id'].values
            precision, recall = precision_recall_at_k(targets, predictions)
            precisions.append(precision)
            recalls.append(recall)
        return np.mean(precisions), np.mean(recalls)

    train_users, train_items, train_ratings, test_users, test_items, test_ratings = prepare_dataset(train, test)
    model = train_model(train_users, train_items, train_ratings, test_users, test_items, test_ratings, user_info, vod_info)
    precision, recall = evaluate(test, model, score_matrix)
    # result = recommend(10, model, score_matrix, 10)

    return precision, recall