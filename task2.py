import sklearn.model_selection
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd


def tts(  dataset: pd.DataFrame,
                       label_col: str, 
                       test_size: float,
                       should_stratify: bool,
                       random_state: int) -> tuple[pd.DataFrame,pd.DataFrame,pd.Series,pd.Series]:
    # TODO: Read the function description in https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task2.html and implement the function as described
    #train_features = pd.DataFrame()
    #test_features = pd.DataFrame()
    #train_labels = pd.Series()
    #test_labels = pd.Series()
    X = dataset.drop(columns=[label_col])
    y = dataset[label_col]
    if should_stratify:
        splitter = sklearn.model_selection.StratifiedShuffleSplit(n_splits=1,test_size=test_size,random_state=random_state)
        train_idx, test_idx = next(splitter.split(X, y))
    else:
        splitter = sklearn.model_selection.ShuffleSplit(n_splits=1,test_size=test_size,random_state=random_state)
        train_idx, test_idx = next(splitter.split(X))
    train_features = X.iloc[train_idx]
    test_features = X.iloc[test_idx]
    train_labels = y.iloc[train_idx]
    test_labels = y.iloc[test_idx]
    return train_features, test_features, train_labels, test_labels

def get_hour_of_day(df):
    return pd.to_datetime(df["time"]).dt.hour.rename("time_hour")
class PreprocessDataset:
    def __init__(self, 
                 one_hot_encode_cols:list[str],
                 min_max_scale_cols:list[str],
                 n_components:int,
                 feature_engineering_functions:dict
                ):
        # TODO: Add any state variables you may need to make your functions work
        #return
        self.one_hot_encode_cols = one_hot_encode_cols
        self.min_max_scale_cols = min_max_scale_cols
        self.n_components = n_components
        self.feature_engineering_functions = feature_engineering_functions
        self._pca = None
        self._mms_cols = []
        self._pca_cols = []
        self.one_hot_train_columns = []
                       
    def one_hot_encode_columns_train(self,train_features:pd.DataFrame) -> pd.DataFrame:
        # TODO: Read the function description in https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task2.html and implement the function as described
        one_hot_encoded_dataset = pd.get_dummies(train_features,columns=self.one_hot_encode_cols)
        self.one_hot_train_columns = one_hot_encoded_dataset.columns
        return one_hot_encoded_dataset

    def one_hot_encode_columns_test(self,test_features:pd.DataFrame) -> pd.DataFrame:
        # TODO: Read the function description in https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task2.html and implement the function as described
        one_hot_encoded_dataset = pd.get_dummies(test_features,columns=self.one_hot_encode_cols)
        one_hot_encoded_dataset = one_hot_encoded_dataset.reindex(columns=self.one_hot_train_columns,fill_value=0)
        return one_hot_encoded_dataset
  
    def min_max_scaled_columns_train(self,train_features:pd.DataFrame) -> pd.DataFrame:
        # TODO: Read the function description in https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task2.html and implement the function as described
        #min_max_scaled_dataset = pd.DataFrame()
        #return min_max_scaled_dataset
        #if not hasattr(self, "_mms"):
        #    self._mms = MinMaxScaler()
        #    self._mms.fit(train_features[numeric_cols])            
        #    self._mms_cols = numeric_cols

        #df = train_features.copy()
        #numeric_cols = df.select_dtypes(include='number').columns

        #df[numeric_cols] = self._mms.fit_transform(df[numeric_cols])
        #return df
        df = train_features.copy()
        if not hasattr(self, "_mms"):
            self._mms = MinMaxScaler()
        if not hasattr(self, "_mms_cols"):
            self._mms_cols = []
        scale_cols = [col for col in self.min_max_scale_cols if col in df.columns]
        if scale_cols:
            self._mms_cols = scale_cols
            df[scale_cols] = self._mms.fit_transform(df[scale_cols])
        return df 
           
    def min_max_scaled_columns_test(self,test_features:pd.DataFrame) -> pd.DataFrame:
        # TODO: Read the function description in https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task2.html and implement the function as described
        #min_max_scaled_dataset = pd.DataFrame()
        #return min_max_scaled_dataset
        result = test_features.copy()
        # Columns used 
        #if hasattr(self, "_mms"):
        #    mms_cols = list(self._mms.feature_names_in_)
            # Scale only those columns
        #    scaled = self._mms.transform(result[mms_cols])
        #    scaled_df = pd.DataFrame(
        #        scaled,
        #        columns=mms_cols,
        #        index=result.index
        #    )
        #    for col in mms_cols:
        #        result[col] = scaled_df[col]
            # Remaining numeric columns to float 
        #    for col in result.columns:
        #        if col not in mms_cols and pd.api.types.is_numeric_dtype(result[col]):
        #            result[col] = result[col].astype(float)
        if not hasattr(self, "_mms_cols"):
            self._mms_cols = []
        if not hasattr(self, "_mms"):
            self._mms = MinMaxScaler()
        if self._mms_cols:
            result[self._mms_cols] = self._mms.transform(result[self._mms_cols])
        return result

    def pca_train(self, train_features: pd.DataFrame) -> pd.DataFrame:
        # TODO: Read the function description in https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task2.html and implement the function as described
        #pca_dataset = pd.DataFrame()
        #return pca_dataset
        numeric_df = train_features.select_dtypes(include=[np.number])
        self._pca_cols = list(numeric_df.columns)
        self._pca = PCA(n_components=self.n_components)
        transformed_data = self._pca.fit_transform(numeric_df)

        columns = [f"component_{i+1}" for i in range(self.n_components)]
        return pd.DataFrame(transformed_data,index=train_features.index,columns=columns)
    
    def pca_test(self,test_features: pd.DataFrame) -> pd.DataFrame:
        # TODO: Read the function description in https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task2.html and implement the function as described
        #pca_dataset = pd.DataFrame()
        #return pca_dataset
        #numeric_df = test_features.select_dtypes(include=[np.number])
        numeric_df = test_features[self._pca_cols]
        transformed_data = self._pca.transform(numeric_df)

        columns = [f"component_{i+1}" for i in range(self.n_components)]
        return pd.DataFrame(transformed_data,index=test_features.index,columns=columns)        
 
    def feature_engineering_train(self,train_features:pd.DataFrame) -> pd.DataFrame:
        # TODO: Read the function description in https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task2.html and implement the function as described
        #feature_engineered_dataset = pd.DataFrame()
        #return feature_engineered_dataset
        feature_engineered_dataset = train_features.copy()
        for new_col, func in self.feature_engineering_functions.items():
            feature_engineered_dataset[new_col] = func(feature_engineered_dataset)
        return feature_engineered_dataset
    
    def feature_engineering_test(self,test_features:pd.DataFrame) -> pd.DataFrame:
        # TODO: Read the function description in https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task2.html and implement the function as described
        #feature_engineered_dataset = pd.DataFrame()
        #return feature_engineered_dataset
        feature_engineered_dataset = test_features.copy()
        for new_col, func in self.feature_engineering_functions.items():
            feature_engineered_dataset[new_col] = func(feature_engineered_dataset)
        return feature_engineered_dataset

    def preprocess_train(self,train_features:pd.DataFrame) -> pd.DataFrame:
        # TODO: Read the function description in https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task2.html and implement the function as described
        #preprocessed_dataset = pd.DataFrame()
        #return preprocessed_dataset
        #df = train_features.copy()
        # Feature engineering
        #for name, func in self.feature_engineering_functions.items():
        #    df[name] = func(df)
        # encoding
        #df = pd.get_dummies(df, columns=self.categorical_cols)
        # Scaling
        #df[self.min_max_scale_cols] = self._mms.fit_transform(df[self.min_max_scale_cols])
        #self.final_columns = df.columns
        df = self.feature_engineering_train(train_features)
        df = self.one_hot_encode_columns_train(df)
        df = self.min_max_scaled_columns_train(df)
        if self.n_components:
            df = self.pca_train(df)
        return df
        
          
    def preprocess_test(self,test_features:pd.DataFrame) -> pd.DataFrame:
        # TODO: Read the function description in https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task2.html and implement the function as described
        #preprocessed_dataset = pd.DataFrame()
        #return preprocessed_dataset
        #df = test_features.copy()
        # Feature engineering
        #for name, func in self.feature_engineering_functions.items():
        #    df[name] = func(df)
        # Encoding
        #df = pd.get_dummies(df, columns=self.categorical_cols)
        # Scaling
        #df[self.min_max_scale_cols] = self._mms.transform(df[self.min_max_scale_cols])
        #df = df.reindex(columns=self.final_columns, fill_value=0)
        df = self.feature_engineering_test(test_features)
        df = self.one_hot_encode_columns_test(df)
        df = self.min_max_scaled_columns_test(df)
        if self.n_components:
            df = self.pca_test(df)
        return df
