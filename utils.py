import pandas as pd

def replace_rare_categories_with_other(df, column, threshold=0.05):
    category_counts = df[column].value_counts(normalize=True)
    rare_categories = category_counts[category_counts < threshold].index
    df[column] = df[column].apply(lambda x: 'other' if x in rare_categories else x)
    return df


def prep_data(df,save=True):
    print('Basic preproccessing given dataset started')
    for column in df.columns[2:]:
        df = replace_rare_categories_with_other(df,column)
    print('Basic preproccessing given dataset finished\nLoading transformed additive data started')
    members_t = pd.read_csv('members_t.csv')
    songs_t = pd.read_csv('songs_t.csv')
    print('Loading transformed additive data finished\nMerging datasets started')
    merged_1 = pd.merge(df, members_t, on='msno', how='inner')
    merged_2 = pd.merge(merged_1,songs_t,on='song_id',how='inner')
    print('Merging datasets finished\nDropping irrelevant columns started')
    merged_2 = merged_2.drop(columns=['genre_ids','language','song_length','time_est','registered_via','city'],axis=1)
    print("Dropping irrelevant columns finished\nOne-hot encoding started")
    df_encoded = pd.get_dummies(merged_2, columns=['source_system_tab','source_screen_name','source_type'])
    print("One-hot encoding finished, done preproccessing!")
    if save:
        df.to_csv('train_t.csv')
    return df_encoded

def prep_test_data(df):
    print('Basic preproccessing given dataset started')
    train_t = pd.read_csv('train_t.csv').drop(columns=['target'],axis=1)
    for column in df.columns[2:]:
        uniques = train_t[column].unique()
        df[column] = df[column].apply(lambda x: 'other' if x not in uniques else x)
    print('Basic preproccessing given dataset finished\nLoading transformed additive data started')
    members_t = pd.read_csv('members_t.csv')
    songs_t = pd.read_csv('songs_t.csv')
    print('Loading transformed additive data finished\nMerging datasets started')
    merged_1 = pd.merge(df, members_t, on='msno', how='left')
    merged_2 = pd.merge(merged_1,songs_t,on='song_id',how='left')
    print('Merging datasets finished\nDropping irrelevant columns started')
    merged_2 = merged_2.drop(columns=['genre_ids','language','song_length','time_est','registered_via','city'],axis=1)
    print("Dropping irrelevant columns finished\nOne-hot encoding started")
    df_encoded = pd.get_dummies(merged_2, columns=['source_system_tab','source_screen_name','source_type'])
    print("One-hot encoding finished, done TEST preproccessing!")
    return df_encoded