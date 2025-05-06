## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
df = pd.read_csv("/content/Encoding Data.csv")
df
```
![image](https://github.com/user-attachments/assets/9bc7da00-0da9-4e38-934b-81482715c330)

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm = ['Hot','Warm','Cold']
e1 = OrdinalEncoder(categories=[pm])
e1.fit_transform(df[['ord_2']])
```
![image](https://github.com/user-attachments/assets/3e08076d-cf49-48e9-b816-dd4f7138d9ec)

```
df['bo2'] = e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/user-attachments/assets/d96f3ea7-9dd9-4eae-8b23-7d5d180eb2ec)

```
le = LabelEncoder()
dfc = df.copy()
dfc['ord_2'] = le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/3d5e3e50-68a2-4c84-852b-ab20c3e6c2bf)

```
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
df2 = df.copy()
enc = pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
df2 = pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/user-attachments/assets/407d05ef-6ab6-464a-83e4-0c798fc39545)

```
pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/user-attachments/assets/68a6e161-713e-4a42-be04-4cd4d82feb5e)

```
pip install --upgrade category_encoders
```
![img-7](https://github.com/user-attachments/assets/74b0ffb8-000f-41c4-844e-d8b28ace2f8c)

```
from category_encoders import BinaryEncoder
df = pd.read_csv("/content/data.csv")
df
```

![image](https://github.com/user-attachments/assets/1298db35-e15f-4367-9246-6723332f5c9b)

```
be = BinaryEncoder()
nd = be.fit_transform(df['Ord_2'])
dfb = pd.concat([df,nd],axis=1)
dfb1 = df.copy()
dfb
```

![image](https://github.com/user-attachments/assets/1e076e4b-c661-4c1b-8d17-96b1f7dd62cf)

```
from category_encoders import TargetEncoder
te = TargetEncoder()
cc = df.copy()
new = te.fit_transform(X=cc["City"],y=cc["Target"])
cc = pd.concat([cc,new],axis=1)
cc
```

![image](https://github.com/user-attachments/assets/cacfcb0d-f42e-4414-9c0c-84b467dae0cc)

Feature Transformation
```
import pandas as pd
import numpy as np
from scipy import stats
df = pd.read_csv("/content/Data_to_Transform.csv")
df
```

![image](https://github.com/user-attachments/assets/a2e343c3-103b-4c97-bd77-fc889e5abed7)

```
df.skew()
```

![img-12](https://github.com/user-attachments/assets/8f181956-0cd8-4517-b079-1ddc6c4ffea5)

```
np.log(df["Highly Positive Skew"])
```

![img-13](https://github.com/user-attachments/assets/9eb9e2b6-65fe-43e7-8efe-fe3c7ef87500)

```
np.reciprocal(df["Moderate Positive Skew"])
```

![img-14](https://github.com/user-attachments/assets/0328d97b-3e44-47f7-8dc8-91c4af3fd726)

```
np.sqrt(df["Highly Positive Skew"])
```

![img-15](https://github.com/user-attachments/assets/53988917-bdd5-4547-860b-76e4de5e96d5)

```
np.square(df["Highly Positive Skew"])
```

![img-16](https://github.com/user-attachments/assets/2d061ab2-cbe2-4856-87f6-94fb7a9fbd5c)

```
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df
```

![img-17](https://github.com/user-attachments/assets/b1e3d361-55d2-4e02-a7d1-415cfeb5b90e)

```
df["Moderate Negative Skew_yeojohnson"],parameters = stats.yeojohnson(df["Moderate Negative Skew"])
df
```

![image](https://github.com/user-attachments/assets/a8e35562-1fa0-4f0f-a749-40214b7375fd)

```
df.skew()
```

![image](https://github.com/user-attachments/assets/018c3a1d-6def-44cf-8748-e261f1043ce6)

```
df["Highly Negative Skew_yeojohnson"],parameters = stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```

![image](https://github.com/user-attachments/assets/8c53111a-5503-415f-8d59-7fb9440f18ff)

```
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"] = qt.fit_transform(df[["Moderate Negative Skew"]])
df
```

![image](https://github.com/user-attachments/assets/50414eb2-cbbe-4fa3-a436-65e4dc4d2836)

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/f7b3e226-08bf-4ca7-adfa-b510b75456d6)

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/034128f7-d98d-4caf-b6b6-22ea5bdadc50)

```
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew"] = qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/ed90fd0d-d96e-4f58-aca8-42bbe6ada9f2)

```
df["Highly Negative Skew_1"] = qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/29b28f07-7468-4bc0-9eef-845f845830a4)

```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/c7951833-921d-43bd-bff5-688a7710d23a)

# RESULT:
   Thus, the given data was successfully read, feature encoding and transformation were performed, and the resulting data was saved to a file.
       
