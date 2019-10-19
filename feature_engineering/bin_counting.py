
def bin_counting(df, target_col, count_col):
    
    """
    概要：
        カテゴリ変数をカテゴリ値×目的変数の値ごとにカウントし、結果をデータフレームで返します。
    
    引数：
        df : カウントしたいカテゴリ変数が含まれているデータフレーム
        target_col : 目的変数
        count_col : カウントしたい変数
        
    戻り値
        count_colのユニーク値×target_colのユニーク値のカウント結果を格納したデータフレーム
    """
    
    unique_target_val = np.unique(df[target_col]) #targetのユニーク数取得
    count_dict = {} #targetのカウント数を格納する辞書
    df_result = pd.DataFrame() #結果格納dataframe
    
    total_count = df[count_col].value_counts().rename(f'{count_col}_total') #カウントする変数について、カテゴリ値ごとのトータル件数取得
    
    #ターゲットのカテゴリ値ごとにカウント変数を数え、辞書に格納
    for val in unique_target_val:
        idx = df[target_col]  == val        
        count_dict[f'{count_col}_count_{val}'] = df[idx].groupby(count_col)[count_col].count()
    
    #df_resultにカウント結果をマージする処理
    for col_name, s in count_dict.items(): #col_nameは最終結果の変数名、sがカウント結果のシリーズ
        s = s.rename(col_name) #seriesをrenameしておく。リネーム後の名前がDFのカラム名に設定される。
        
        #seriesのindexが紐づく形で結合。片方にしかindexがなければ欠損になる。
        #indexはcount_col変数のカテゴリ値になっている。
        df_result = pd.concat([df_result, s],  axis = 1, sort = False) 
    return pd.concat([df_result, total_count], axis = 1, sort = False).fillna(0) #totalのtargetカウント数の列追加。欠損0埋め。