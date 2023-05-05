Download Link: https://assignmentchef.com/product/solved-data-201-assignment-2
<br>
<h1>Problem Statement</h1>

A retail company wants to understand the customer purchase behaviour (specifically, purchase amount) against various products of different categories. They have shared purchase summary of various customers for selected high volume products from last month. The data set also contains customer demographics (age, gender, marital status, city_type, stay_in_current_city), product details (product_id and product category) and total purchase_amount from last month.

You need to build a model to predict the purchase amount of customer against various products which will help the company to create personalized offer for customers against different products.

<h1>Data</h1>

<strong>Variable                                                                             Description</strong>

<table width="473">

 <tbody>

  <tr>

   <td width="188">User_ID</td>

   <td width="285">User ID</td>

  </tr>

  <tr>

   <td width="188">Product_ID</td>

   <td width="285">Product ID</td>

  </tr>

  <tr>

   <td width="188">Gender</td>

   <td width="285">Sex of User</td>

  </tr>

  <tr>

   <td width="188">Age</td>

   <td width="285">Age in bins</td>

  </tr>

  <tr>

   <td width="188">Occupation</td>

   <td width="285">Occupation (Masked)</td>

  </tr>

  <tr>

   <td width="188">City_Category</td>

   <td width="285">Category of the City (A, B, C)</td>

  </tr>

  <tr>

   <td width="188">Stay_In_Current_City_Years</td>

   <td width="285">Number of years stay in current city</td>

  </tr>

  <tr>

   <td width="188">Marital_Status</td>

   <td width="285">Marital Status</td>

  </tr>

  <tr>

   <td width="188">Product_Category_1</td>

   <td width="285">Product Category (Masked)</td>

  </tr>

  <tr>

   <td width="188">Product_Category_2</td>

   <td width="285">Product may belongs to other category also (Masked)</td>

  </tr>

  <tr>

   <td width="188">Product_Category_3</td>

   <td width="285">Product may belongs to other category also (Masked)</td>

  </tr>

  <tr>

   <td width="188">Purchase</td>

   <td width="285">Purchase Amount (Target Variable)</td>

  </tr>

 </tbody>

</table>

<strong>Evaluation</strong>

The root mean squared error (RMSE) will be used for model evaluation.

<h1>Questions and Code</h1>

In [1]:

<strong>import</strong> <strong>numpy</strong> <strong>as</strong> <strong>np </strong><strong>import</strong> <strong>pandas</strong> <strong>as</strong> <strong>pd </strong><strong>from</strong> <strong>sklearn</strong> <strong>import</strong> metrics

<strong>from</strong> <strong>sklearn.compose</strong> <strong>import</strong> ColumnTransformer <strong>from</strong> <strong>sklearn.impute</strong> <strong>import</strong> SimpleImputer <strong>from</strong> <strong>sklearn.linear_model</strong> <strong>import</strong> LinearRegression <strong>from</strong> <strong>sklearn.model_selection</strong> <strong>import</strong> train_test_split <strong>from</strong> <strong>sklearn.neighbors</strong> <strong>import</strong> KNeighborsRegressor <strong>from</strong> <strong>sklearn.pipeline</strong> <strong>import</strong> Pipeline

<strong>from</strong> <strong>sklearn.preprocessing</strong> <strong>import</strong> OneHotEncoder, MinMaxScaler, StandardScaler np.random.seed = 42

Load the given dataset.

In [2]:

Out[2]:

Age                            object

City_Category                  object

Gender                         object

Marital_Status                  int64

Occupation                      int64

Product_Category_1              int64

Product_Category_2              int64

Product_Category_3              int64

Product_ID                      int64

Purchase                      float64

Stay_In_Current_City_Years     object User_ID                         int64 dtype: object

<ol>

 <li><strong> Is there any missing value? [1 point]</strong></li>

</ol>

In [3]:

Out[3]:

Age                           0

City_Category                 0

Gender                        0

Marital_Status                0

Occupation                    0

Product_Category_1            0

Product_Category_2            0

Product_Category_3            0

Product_ID                    0

Purchase                      0

Stay_In_Current_City_Years    0 User_ID                       0 dtype: int64

<ol start="2">

 <li><strong>Drop attribute </strong><strong>User_ID </strong><strong>. [1 point] </strong>In [4]:</li>

 <li><strong>Then convert the following categorical attributes below to numerical values with the rule as below.</strong></li>

</ol>

<strong>[4 points]</strong>

Gender : F :0, M :1

Age : 0-17 :0, 18-25 :1, 26-35 :2, 36-45 :3, 46-50 :4, 51-55 :5, 55+ :6

Stay_In_Current_City_Years : 0 :0, 1 :1, 2 :2, 3 :3, 4+ :4

You may want to apply a lambda function to each row of a column in the dataframe. Some examples here <a href="https://thispointer.com/pandas-apply-apply-a-function-to-each-row-column-in-dataframe/">may be helpful: </a><a href="https://thispointer.com/pandas-apply-apply-a-function-to-each-row-column-in-dataframe/">https://thispointer.com/pandas-appl</a><a href="https://thispointer.com/pandas-apply-apply-a-function-to-each-row-column-in-dataframe/">y</a><a href="https://thispointer.com/pandas-apply-apply-a-function-to-each-row-column-in-dataframe/">-appl</a><a href="https://thispointer.com/pandas-apply-apply-a-function-to-each-row-column-in-dataframe/">y</a><a href="https://thispointer.com/pandas-apply-apply-a-function-to-each-row-column-in-dataframe/">-a-function-to-each-row-column-in-dataframe/</a>

<a href="https://thispointer.com/pandas-apply-apply-a-function-to-each-row-column-in-dataframe/">(https://thispointer.com/pandas-appl</a><a href="https://thispointer.com/pandas-apply-apply-a-function-to-each-row-column-in-dataframe/">y</a><a href="https://thispointer.com/pandas-apply-apply-a-function-to-each-row-column-in-dataframe/">-appl</a><a href="https://thispointer.com/pandas-apply-apply-a-function-to-each-row-column-in-dataframe/">y</a><a href="https://thispointer.com/pandas-apply-apply-a-function-to-each-row-column-in-dataframe/">-a-function-to-each-row-column-in-dataframe/)</a>

In [5]:

data[‘Gender’] = data[‘Gender’].map({‘F’:0,’M’:1})

data[‘Age’] = data[‘Age’].map({‘0-17′:0,’18-25′:1, ’26-35′:2, ’36-45′:3, ’46-50′:4, ’51

-55′:5, ’55+’:6}) data[‘Stay_In_Current_City_Years’] = data[‘Stay_In_Current_City_Years’].map({‘0′:0,’1’:

1, ‘2’:2, ‘3’:3, ‘4+’:4}) data.head() Out[5]:

<strong>Age    City_Category    Gender    Marital_Status    Occupation     Product_Category_1    Product_Cat</strong>

<ul>

 <li>0 A 0              0              10           1</li>

 <li>4 B 1              1              7              1</li>

 <li>2 A 1              1              20           1</li>

 <li>5 A 0              0              9              5</li>

</ul>

<h2>4            5                            A                0                             0                        9                                         2</h2>

<ol start="4">

 <li><strong>Randomly split the current data frame into 2 subsets for training (80%) and test (20%). Use<em>random_state = 42</em>. [2 points]</strong></li>

</ol>

In [6]:

data_train, data_test = train_test_split(data, test_size=0.2, random_state=42)

<ol start="5">

 <li><strong>Get the list of numerical predictors (all the attributes in the current data frame except the target, </strong><strong>Purchase </strong><strong>) and the list of categorical predictor. [1 point] </strong>In [7]:</li>

 <li><strong>Create a transformation pipeline including two pipelines handling the following [3 points]</strong></li>

</ol>

Numerical <em>predictors</em>: apply Standard Scaling

Categorical <em>predictor</em>: apply One-hot-encoding

You will need to use ColumnTransformer . The example in Week 3 lectures may be helpful.

In [8]:

nom_onehot = [(‘onehot’, OneHotEncoder(sparse=<strong>False</strong>, handle_unknown=’ignore’))] nom_pl = Pipeline(nom_onehot)

num_impute = SimpleImputer(strategy=’mean’) num_normalised = MinMaxScaler()

num_pl = Pipeline([(‘imp’, num_impute), (‘norm’, num_normalised)])

num_cols = list(X_train.select_dtypes([np.number]).columns) nom_cols = list(set(X_train.columns) – set(num_cols))

transformers = [(‘num’, num_pl, num_cols),

(‘nom’, nom_pl, nom_cols)] col_transform = ColumnTransformer(transformers)

<ol start="7">

 <li><strong>Train and use that transformation pipeline to transform the training data (e.g. for a machinelearning model). [2 points] </strong>In [9]:</li>

</ol>

Out[9]:

array([[0.33333333, 0.        , 1.        , …, 1.        , 0.        ,         0.        ],

[0.33333333, 1.        , 0.        , …, 1.        , 0.        ,         0.        ],

[0.33333333, 1.        , 0.        , …, 0.        , 1.        ,

<ol>

 <li>],</li>

</ol>

…,

[0.16666667, 1.        , 0.        , …, 0.        , 1.        ,         0.        ],

[0.5       , 1.        , 0.        , …, 1.        , 0.        ,         0.        ],

[0.66666667, 1.        , 1.        , …, 0.        , 1.        ,

<ol>

 <li>]])</li>

 <li><strong>Use that transformation pipeline to transform the test data (e.g. for testing a machine learningmodel). [2 points]</strong></li>

</ol>

In [10]:

Out[10]:

array([[0.5       , 1.        , 1.        , …, 1.        , 0.        ,         0.        ],

[0.16666667, 1.        , 1.        , …, 0.        , 1.        ,         0.        ],

[0.5       , 0.        , 1.        , …, 0.        , 0.        ,         1.        ],

…,

[0.33333333, 0.        , 0.        , …, 0.        , 0.        ,         1.        ],

[0.        , 0.        , 0.        , …, 1.        , 0.        ,         0.        ],

[0.5       , 1.        , 0.        , …, 0.        , 1.        ,

<ol>

 <li>]])</li>

 <li><strong>Build a Linear Regression model using the training data after transformation and test it on the testdata. Report the RMSE values on the training and test data. [3 points]</strong></li>

</ol>

<a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html">Document: </a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html">https://scikit-learn.or</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html">g</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html">/stable/modules/</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html">g</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html">enerated/sklearn.linear_model.LinearRe</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html">g</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html">ression.html (https://scikit-learn.or</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html">g</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html">/stable/modules/</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html">g</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html">enerated/sklearn.linear_model.LinearRe</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html">g</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html">ression.html)</a>

In [11]:

lr = LinearRegression()

lr_pipeline = Pipeline([(‘col_trans’, col_transform), (‘lr’, lr)]) lr_pipeline.fit(X_train, y_train)

lr_train_pred = lr_pipeline.predict(X_train) lr_test_pred = lr_pipeline.predict(X_test)

print(” Linear Regression Training Set RMSE: <strong>%.4g</strong>” % np.sqrt(metrics.mean_squared_error

(lr_train_pred, y_train)))

print(“Linear Regression Test Set RMSE: <strong>%.4g</strong>” % np.sqrt(metrics.mean_squared_error(lr_t est_pred, y_test)))

Linear Regression Training Set RMSE: 4600

Linear Regression Test Set RMSE: 4616

<ol start="10">

 <li><strong> Repeat Question 9 using a </strong><strong>KNeighborsRegressor </strong><strong>. Comment on the processing time and performance of the model in this question. [1 point]</strong></li>

</ol>

<a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html">Document: </a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html">https://scikit-learn.or</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html">g</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html">/stable/modules/</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html">g</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html">enerated/sklearn.nei</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html">g</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html">hbors.KNei</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html">g</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html">hborsRe</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html">g</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html">ressor.html</a>

<a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html">(https://scikit-learn.or</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html">g</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html">/stable/modules/</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html">g</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html">enerated/sklearn.nei</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html">g</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html">hbors.KNei</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html">g</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html">hborsRe</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html">g</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html">ressor.html)</a>

In [12]:

knn = KNeighborsRegressor()

knn_pipeline = Pipeline([(‘col_trans’, col_transform), (‘knn’, knn)]) knn_pipeline.fit(X_train, y_train)

knn_train_pred  = knn_pipeline.predict(X_train) knn_test_pred = knn_pipeline.predict(X_test)

print(“K Neighbours Regressor Training Set RMSE: <strong>%.4g</strong>” % np.sqrt(metrics.mean_squared_e rror(knn_train_pred, y_train)))

print(“K Neighbours Regressor Test Set RMSE: <strong>%.4g</strong>” % np.sqrt(metrics.mean_squared_error (knn_test_pred, y_test)))

K Neighbours Regressor Training Set RMSE: 3407

K Neighbours Regressor Test Set RMSE: 4230

The K-Nearest Neighbours Regression is significantly slower than Linear Regression because in KNN each training instance has to be compared with every other training instance one-by-one, this makes it computationally expensive especially when your dataset is wide (a lot of features, which in our instance, it does). It’s complexity is n × n because each instance has n comparisions.

KNN Regression also has poorer performance than the Linear Regression model, as observed by how RMSE for the training and test sets for Linear Regression are near identicial, it means there’s little variance in the residuals. KNN Regression on the other hand has a pretty big discrepancy between the RMSE values, most likely due to a very terrible signal-to-noise ratio thanks to our large number of features. KNN works best when it comes to small datasets without too much noise