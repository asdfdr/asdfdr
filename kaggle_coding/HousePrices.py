import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn import metrics
from scipy.stats import norm, skew
from scipy import stats

# %matplotlib inline

# 读取数据
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print(train.shape)
print(test.shape)
# 显示最起那么的五条数据
print(train.head(5))

# 提取Id这一列
train_ID = train['Id']
test_ID = test['Id']


train.drop("Id", axis=1, inplace=True)
test.drop("Id", axis=1, inplace=True)
print("\nThe train data size after dropping Id feature is : {} ".format(train.shape))
print("The test data size after dropping Id feature is : {} ".format(test.shape))

# 离群点处理
fig, ax = plt.subplots()
ax.scatter(x=train['GrLivArea'], y=train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()

# 删除离群点
train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index)

# 再次查看离群点情况
fig, ax = plt.subplots()
ax.scatter(train['GrLivArea'], train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()

sns.distplot(train['SalePrice'], fit=norm);
# 使用probplot函数检测房价偏离正态分布
(mu, sigma) = norm.fit(train['SalePrice'])
print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

# 绘制
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
           loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

# 获取QQ-plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()

# 对数据做log处理
train["SalePrice"] = np.log1p(train["SalePrice"])
train_SalePrice = train["SalePrice"]

# 查看新的数据分布
sns.distplot(train['SalePrice'], fit=norm);

# 获取新的数据的分布参数
(mu, sigma) = norm.fit(train['SalePrice'])
print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

# 绘制分布
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

# 查看Q-Qplot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()

#训练集数据量
ntrain = train.shape[0]
#测试集数据量
ntest = test.shape[0]
y_train = train.SalePrice.values
# train的数据和test的数据合并，一起处理
all_data = pd.concat((train, test)).reset_index(drop=True)
# all_data.to_csv("dddd.csv")
# 去掉SalePrice，SalePrice不需要处理
all_data.drop(['SalePrice'], axis=1, inplace=True)

print(all_data.head(5))
print(all_data.shape)

# 计算每一列的缺失率，从高到低排列
percent = (all_data.isnull().sum() / len(all_data)).sort_values(ascending=False)
# 数据缺失情况
all_data_na = percent[percent > 0]
print(all_data_na)
# 缺失数据可视化
f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(x=all_data_na.index, y=all_data_na)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)
plt.show()

# 使用热力图，分析各个特征和房价的关系
corrmat = train.corr()
f, ax = plt.subplots(figsize=(12, 9))
ax.set_xticklabels(corrmat, rotation='horizontal')
sns.heatmap(corrmat, vmax=0.9, square=True)
label_y = ax.get_yticklabels()
plt.setp(label_y, rotation=360)
label_x = ax.get_xticklabels()
plt.setp(label_x, rotation=90)
plt.show()

# 缺失数据列
# missing_index=['PoolQC', 'MiscFeature', 'Alley', 'Fence',  'FireplaceQu',
#        'LotFrontage', 'GarageQual', 'GarageYrBlt', 'GarageFinish',
#        'GarageCond', 'GarageType', 'BsmtCond', 'BsmtExposure', 'BsmtQual',
#        'BsmtFinType2', 'BsmtFinType1', 'MasVnrType', 'MasVnrArea', 'MSZoning',
#        'BsmtHalfBath', 'Utilities', 'Functional', 'BsmtFullBath', 'Electrical',
#        'Exterior2nd', 'KitchenQual', 'GarageCars', 'Exterior1st', 'GarageArea',
#        'TotalBsmtSF', 'BsmtUnfSF', 'BsmtFinSF2', 'BsmtFinSF1', 'SaleType']

# PoolQC: Pool quality 游泳池质量。NA，表示没有游泳池。大量数据的值是NA
all_data["PoolQC"] = all_data["PoolQC"].fillna("None")

# MiscFeature：MiscFeature: Miscellaneous feature not covered in other categories 其它条件中未包含部分的特性
all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")

# Alley: Type of alley access 小道的路面类型。
all_data["Alley"] = all_data["Alley"].fillna("None")
print("train的alley")
print(train["Alley"])
print("all_data的allty", all_data["Alley"])

# Fence: Fence quality 围栏质量
all_data["Fence"] = all_data["Fence"].fillna("None")

# FireplaceQu: Fireplace quality 壁炉质量
all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")

# LotFrontage: Linear feet of street connected to property 房子同街道之间的距离。
# neighborhood: Physical locations within Ames city limits。Ames市区范围内的物理位置
# 通过neighborhood的所有中值来填充缺失值
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

# GarageType: Garage location 车库位置
# GarageFinish: Interior finish of the garage 车库中间建成时间（比如翻修）
# GarageQual: Garage quality 车库质量
# GarageCond: Garage condition 车库条件
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna("None")
# GarageYrBlt: Year garage was built 车库建造时间
# GarageArea: Size of garage in square feet 车库面积
# GarageCars: Size of garage in car capacity 车库大小以停车数量表示
# 这几个数据用零填充
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)

# BsmtFinSF1: Type 1 finished square feet Type 1完工面积
# BsmtFinSF2: Type 2 finished square feet Type 2完工面积
# BsmtUnfSF: Unfinished square feet of basement area 地下室区域未完工面积
# TotalBsmtSF: Total square feet of basement area 地下室总体面积
# BsmtFullBath: Basement full bathrooms 地下室全浴室
# BsmtHalfBath: Basement half bathrooms 地下室半浴室
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)

# BsmtQual: Height of the basement 地下室高度
# BsmtCond：BsmtCond: General condition of the basement 地下室总体情况
# BsmtExposure: Walkout or garden level basement walls 地下室出口或者花园层的墙面
# BsmtFinType1: Quality of basement finished area 地下室区域质量
# BsmtFinType2: Quality of second finished area (if present) 二次完工面积质量（如果有）
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna("None")

# MasVnrArea: Masonry veneer area in square feet 装饰石材面积
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
# MasVnrType: Masonry veneer type 装饰石材类型
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")

# MSZoning: The general zoning classification 区域分类
# 大部分数据都是‘RL’，这一列的数据用出现频次最高的值填充。
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])

# Utilities: Type of utilities available 配套公用设施类型
# test数据中的Utilities的所有的值都是“AllPub”,只有train的数据里面有一个"NoSeWa"，和两个空值。可以删除这一列
all_data = all_data.drop(['Utilities'], axis=1)

# Functional: Home functionality rating 功能性评级。
# 空值代表typical，所以空值用Typ代替
all_data["Functional"] = all_data["Functional"].fillna("Typ")

# Electrical: Electrical system 电力系统
# 大多数的值都是“SBrkr”，所以用众数来填充
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])

# KitchenQual: Kitchen quality 厨房质量
# 大多数值都是“TA”，所以用众数来填充
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])

# Exterior1st: Exterior covering on house 外立面材料
# Exterior2nd: Exterior covering on house (if more than one material) 外立面材料2
# 空值很少，用众数填充
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

# SaleType: Type of sale 出售类型
# 空值很少，用众数填充
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])

# MSSubClass: Identifies the type of dwelling involved in the sale. 确定销售中涉及的住宅类型。	（建筑等级）
all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")

# 检查是否还有缺失数据
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio': all_data_na})
missing_data.head()
print(missing_data.head())

# 将实际上是类别，但给出的值是数字的列进行转换
# MSSubClass: Identifies the type of dwelling involved in the sale. 确定销售中涉及的住宅类型。	（建筑等级）
all_data['MSSubClass'] = all_data['MSSubClass'].astype(str)

# OverallCond: Overall condition rating 整体条件等级
all_data['OverallCond'] = all_data['OverallCond'].astype(str)

# YrSold: Year Sold 卖出年份
# MoSold: Month Sold (MM)
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)

# 对文本类别的特征进行编号
from sklearn.preprocessing import LabelEncoder

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
        'ExterQual', 'ExterCond', 'HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
        'YrSold', 'MoSold')
# 处理这些列
for c in cols:
    lbl = LabelEncoder()
    lbl.fit(list(all_data[c].values))
    all_data[c] = lbl.transform(list(all_data[c].values))

# 添加房子的总面积
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
# 查看每个特征的偏度
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew': skewed_feats})
print(skewness.head(10))

# 将skewness过大的特征进行处理
skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p

skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    # all_data[feat] += 1
    all_data[feat] = boxcox1p(all_data[feat], lam)

# 类别变量进行哑变量转化
all_data = pd.get_dummies(all_data)
print(all_data.shape)

#最终处理后的数据
train = all_data[:ntrain]
test = all_data[ntrain:]

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb



# 定义验证函数
n_folds = 5
def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

# 模型一
# clf1 = LassoCV(alphas = [1, 0.1, 0.001, 0.0005,0.0003,0.0002, 5e-4])
# clf1.fit(X_train, y)
# lasso_preds = np.expm1(clf1.predict(X_test)) # exp(x) - 1  <---->log1p(x)==log(1+x)
# score1 = rmse_cv(clf1)
# print("\nLasso score: {:.4f} ({:.4f})\n".format(score1.mean(), score1.std()))

# 各项模型及得分情况
lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))
score = rmsle_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

#
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
score = rmsle_cv(ENet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
score = rmsle_cv(KRR)
print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state=5)
score = rmsle_cv(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                             learning_rate=0.05, max_depth=3,
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state=7, nthread=-1)
score = rmsle_cv(model_xgb)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin=55, bagging_fraction=0.8,
                              bagging_freq=5, feature_fraction=0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf=6, min_sum_hessian_in_leaf=11)
score = rmsle_cv(model_lgb)
print("LGBM score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# 模型融合
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    # 定义原始的克隆
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]

        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self

    # 对克隆模型进行预测，并对它们进行平均预测
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)


# 平均四个模型
averaged_models = AveragingModels(models=(ENet, GBoost, KRR, lasso))

score = rmsle_cv(averaged_models)
print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# 更深的模型融合
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    # 再次对原始模型的克隆数据进行拟合。
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        # 训练克隆的基本模型，然后创建折叠预测。
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        # 现在利用折叠预测作为新特征来训练克隆元模型。
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    # 对测试数据进行所有基本模型的预测，并使用平均预测作为元特征以进行元模型的最终预测。
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_])
        return self.meta_model_.predict(meta_features)

#测试融合模型的效果
stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, KRR),
                                                 meta_model = lasso)
score = rmsle_cv(stacked_averaged_models)
print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))

#
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

#最终的训练和预测
#StackedRegressor
stacked_averaged_models.fit(train.values, y_train)
stacked_train_pred = stacked_averaged_models.predict(train.values)
stacked_pred = np.expm1(stacked_averaged_models.predict(test.values))
print("stacked_averaged_models score:",(rmsle(y_train, stacked_train_pred)))

#XGBoost
model_xgb.fit(train, y_train)
xgb_train_pred = model_xgb.predict(train)
xgb_pred = np.expm1(model_xgb.predict(test))
print("XGBoost:",rmsle(y_train, xgb_train_pred))

#LightGBM
model_lgb.fit(train, y_train)
lgb_train_pred = model_lgb.predict(train)
lgb_pred = np.expm1(model_lgb.predict(test.values))
print("LightGBM:",rmsle(y_train, lgb_train_pred))

#按照不同的比例融合之后的效果
print("融合",rmsle(y_train,stacked_train_pred*0.70 +
               xgb_train_pred*0.15 + lgb_train_pred*0.15 ))

#得到最终的数据
ensemble = stacked_pred*0.70 + xgb_pred*0.15 + lgb_pred*0.15

#得到要提交的文件
sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = ensemble
sub.to_csv('submission.csv',index=False)