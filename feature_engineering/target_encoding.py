def target_encoding(df_x, y, target_enc_col, isValid, df_x_test = None, isMultiClass = False, n_splits = 4, shuffle = False, random_state = 0):
    """
    処理概要:
        target_enc_colで指定したカテゴリ変数にtarget encoding適用。
        訓練データに対してはクロスバリデーションで分割した学習データでカテゴリごとの目的変数平均値計算し検証データにその値を適用。これをfoldの数繰り返す。
        テストデータに対しては、訓練データ全体でカテゴリごとの目的変数平均値計算し、その値を適用する。
    引数:
        df_X : target encodingを適用したい列が含まれた訓練用データフレーム
        y : 訓練データの目的変数
        target_enc_col : target encodingを適用したいカテゴリ変数
        isValid : Trueならテストデータに対してtarget encoding, Falseなら訓練データに対してtarget encoding
        df_x_test : テストデータに対してtarget encoding適用する際に、テストデータをデータフレームで指定
        isMultiClass : 多クラス分類の場合Trueを指定。
        n_splits : 訓練データに対してCV実施時のfold数
        shuffle : 訓練データに対してCV実施時にshuffleするか否か。
        random_state : 訓練データに対してCV実施時の乱数シード
    戻り値:
        df_result : 元のカテゴリ変数＋target encoding結果のデータフレーム。indexでsort済。
            
    """
    
    skf = StratifiedKFold(n_splits = n_splits, shuffle = shuffle, random_state = random_state)
    target_enc_col = target_enc_col
    
    #多クラス分類であれば目的変数をダミー変数化。クラスの数だけ2値分類があると考え、クラスの数だけtarget encodingによる特徴量作成。
    if isMultiClass:
        y_for_enc = pd.get_dummies(y)
    else:
        y_for_enc = y

    #target encoding後の特徴量格納DataFrame定義
    df_result = pd.DataFrame()

    #検証用データに対しては訓練データ全体で各カテゴリごとの目的変数平均値計算
    if isValid:
        for i, c in enumerate(y_for_enc.columns):

            df_tmp = pd.DataFrame({target_enc_col: df_x[target_enc_col], c : y_for_enc[c]})
            target_mean = df_tmp.groupby(target_enc_col)[c].mean().rename(f'{target_enc_col}_{c}')
            if i == 0:
                target_mean_all = pd.DataFrame(target_mean)
                continue
            target_mean_all = target_mean_all.join(target_mean)
        df_result = pd.merge(df_x_test.loc[:, [target_enc_col]], target_mean_all, how = 'left', left_on = target_enc_col, right_index = True)

    #訓練データに対してはクロスバリデーション適用。
    #クロスバリデーションの各foldにおいて、学習データでtarget encodingの計算を実施し、検証データに適用。これをfoldの数だけ繰り返す。
    else:
        for i, (idx_train, idx_valid) in enumerate(skf.split(df_x, s_y_train)):
            
            #訓練データからバリデーションデータ分離
            df_x_tr, df_x_val = df_x.iloc[idx_train].copy(), df_x.iloc[idx_valid].copy()
            y_tr, y_val   = y_for_enc.iloc[idx_train].copy(), y_for_enc.iloc[idx_valid].copy()
 
            #多クラス分類の場合、各クラスごとにtarget encoding実施。
            for i, c in enumerate(y_for_enc.columns):
                #各カテゴリ水準ごとの平均値計算
                df_tmp = pd.DataFrame({target_enc_col: df_x_tr[target_enc_col], c : y_tr[c]}) 
                target_mean = df_tmp.groupby(target_enc_col)[c].mean().rename(f'{target_enc_col}_{c}') 
                if i == 0:
                    target_mean_all = pd.DataFrame(target_mean)
                    continue
                target_mean_all = target_mean_all.join(target_mean)
            df_tmp_result = pd.merge(df_x_val.loc[:, [target_enc_col]], target_mean_all, how = 'left', left_on = target_enc_col, right_index = True)

            df_result = df_result.append(df_tmp_result) 

    return df_result.sort_index()
    