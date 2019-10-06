class KMeansFeaturizer:
    """
    数値データをk-meansのクラスタIDにのone hot encodingに変換する。
    この変換器は入力データに対してk-meansを実行し、各データ点を最も近いクラスタのIDに変換する。
    ターゲット変数yが存在する場合、クラスタ分類の境界をより重視したクラスタリング結果を得るために、ターゲット変数を
    スケーリングして入力データに含めてk-meansに渡す
    
    """

    def __init__(self, k = 100, target_scale = 5.0, random_state = None):
        self.k = k #クラスタ数
        self.target_scale = target_scale #ターゲット変数の重み
        self.random_state = random_state 
        self.cluster_encoder = OneHotEncoder().fit(np.array(range(k)).reshape(-1, 1))
        
    def fit(self, X, y = None):
        """
        入力データに対しk-meansを実行し各クラスタの中心を見つける
        """
        if y is None:
            #ターゲット変数がない場合、通常のk-meansを実行する
            km_model = Kmeans(
                n_clusters = self.k, 
                n_init = 20,  #初期値選択において、異なる乱数のシードで初期の重心を選ぶ処理の実行回数
                random_state = self.random_state
            )
            
            km_model.fit(X)
            self.km_model_ = km_model
            self.cluster_centers_ = km_model.cluster_centers_
            return self
        
        
        #ターゲット変数がある場合、スケーリングして入力データに含める
        #スケーリングはどれくらいターゲット変数の影響に重みをつけるか
        data_with_target = np.hstack((X, y[:, np.newaxis] * self.target_scale))
        
        #ターゲットを組み入れたデータで事前学習するためのk-meansモデルを構築する
        #事前学習ではターゲット変数存在時に決められたクラスタ数のときのクラスタ中心を求めている。
        km_model_pretrain = KMeans(n_clusters = self.k,
                                   n_init = 20,
                                   random_state = self.random_state
        )
        km_model_pretrain.fit(data_with_target)
        
        #ターゲット変数の情報を除いて元の空間におけるクラスタを得るために
        #k-meansを再度実行する。事前学習で見つけたクラスタの中心を使って初期化し、
        #クラスタの割り当てと中心の再計算を１回だけ行う
        
        km_model = KMeans(
            n_clusters = self.k,
            #init = km_model_pretrain.cluster_centers_[:, :2], #初期値は事前学習結果のクラスタ中心に設定
            init = km_model_pretrain.cluster_centers_, #これでOKのはず。:2があると2次元データにしか対応できない。
            n_init = 1,
            max_iter = 1 #繰り返し回数の最大値。 (デフォルト値: 300)
        )
        km_model.fit(X)
        
        self.km_model = km_model
        self.cluster_centers_ = km_model.cluster_centers_
        return self
    
    
    def transform(self, X, y = None):
        """
        入力データ点に最も近いクラスタのIDのone-hotエンコーディング結果を返す
        """
        clusters = self.km_model.predict(X)
        
        return self.cluster_encoder.transform(clusters.reshape(-1, 1))


    def fit_transform(self, X, y = None):
        self.fit(X, y)
        return self.transform(X, y)





