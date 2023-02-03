##################################################
# Pandas Alıştırmalar
# Pandas Exercises
##################################################

import numpy as np
import seaborn as sns
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

#########################################
# Görev 1: Seaborn kütüphanesi içerisinden Titanic veri setini tanımlayınız.
# Task 1:  Identify the Titanic dataset from the Seaborn library.
#########################################
df = sns.load_dataset("titanic")
df.head()

#########################################
# Görev 2: Yukarıda tanımlanan Titanic veri setindeki kadın ve erkek yolcuların sayısını bulunuz.
# Task 2:  Find the number of male and female passengers in the Titanic dataset described above.
#########################################
df["sex"].value_counts()

#########################################
# Görev 3: Her bir sutuna ait unique değerlerin sayısını bulunuz.
# Task 3:  Find the number of unique values for each column.
#########################################
df.nunique()

#########################################
# Görev 4: pclass değişkeninin unique değerleri bulunuz.
# Task 4:  Find the unique values of the variable pclass.
#########################################
df["pclass"].nunique()

#########################################
# Görev 5: pclass ve parch değişkenlerinin unique değerlerinin sayısını bulunuz.
# Task 5:  Find the number of unique values of pclass and parch variables.
#########################################
df[["pclass", "parch"]].nunique()

#########################################
# Görev 6: embarked değişkeninin tipini kontrol ediniz. Tipini category olarak değiştiriniz. Tekrar tipini kontrol ediniz.
# Task 6:  Check the type of the embarked variable. Change its type to category. Check the repetition type.
#########################################
df["embarked"].dtype
df["embarked"] = df["embarked"].astype("category")
df["embarked"].dtype

#########################################
# Görev 7: embarked değeri C olanların tüm bilgelerini gösteriniz.
# Task 7:  Show all the sages of those with embarked value C.
#########################################
df.loc[df["embarked"] == "C"]

#########################################
# Görev 8: embarked değeri S olmayanların tüm bilgelerini gösteriniz.
# Task 8:  Show all the sages of those whose embarked value is not S.
#########################################
df.loc[df["embarked"] != "S"]

#########################################
# Görev 9: Yaşı 30 dan küçük ve kadın olan yolcuların tüm bilgilerini gösteriniz.
# Task 9:  Show all the information for female passengers younger than 30 years old.
#########################################
df.query("age < 30 & sex == 'female'")  # another approach
#  df.loc[(df["age"] < 30) & (df["sex"] == "female")] #  another approach

#########################################
# Görev 10: Fare'i 500'den büyük veya yaşı 70 den büyük yolcuların bilgilerini gösteriniz.
# Task 10:  Show the information of passengers whose Fare is over 500 or 70 years old.
#########################################
df.loc[(df["fare"] > 500) | (df["age"] > 70)]

#########################################
# Görev 11: Her bir değişkendeki boş değerlerin toplamını bulunuz.
# Task 11:  Find the sum of the null values in each variable.
#########################################
df.isna().sum()

#########################################
# Görev 12: who değişkenini dataframe'den düşürün.
# Task 12:  drop the who variable from the dataframe.
#########################################
df.drop(columns=["who"]).head()
#  df = df.drop(columns=["who"]) # Don't use inplace, it doesn't save memory on top of that it mutates.
#  df.head()

#########################################
# Görev 13: deck değikenindeki boş değerleri deck değişkenin en çok tekrar eden değeri (mode) ile doldurunuz.
# Task 13:  Fill the empty values in the deck variable with the most repeated value (mode) of the deck variable.
#########################################
df["deck"].head()  # output => 0 NaN, 1 C, 2 NaN 3 C 4 NaN

mode_of_deck = df["deck"].mode().iloc[0]  # output => C    # iloc is faster than []
df["deck"] = df["deck"].fillna(mode_of_deck)  # Don't use inplace, it doesn't save memory on top of that it mutates.
df["deck"].head()  # output => 0 C, 1 C, 2 C, 3 C, 4 C

#########################################
# Görev 14: age değişkenindeki boş değerleri age değişkenin medyanı ile doldurun.
# Task 14:  Fill the empty values in the age variable with the median of the age variable.
#########################################
median_of_age = df["age"].median()
df["age"] = df["age"].fillna(median_of_age)
df["age"].head(10)  # No NaN in First 5 rows therefore .head(10)

#########################################
# Görev 15: survived değişkeninin Pclass ve Cinsiyet değişkenleri kırılımınında sum, count, mean değerlerini bulunuz.
# Task 15:  Find the sum, count, mean values of the Pclass and Gender variables of the survived variable.
#########################################
df_grouped = df.groupby(["pclass", "sex"]).agg({"survived": ["sum", "count", "mean"]}).T
# We got Transpose to see it clearly

#########################################
# Görev 16: 30 yaşın altında olanlar 1, 30'a eşit ve üstünde olanlara 0 vericek bir fonksiyon yazınız.
# Yazdığınız fonksiyonu kullanarak titanik veri setinde age_flag adında bir değişken oluşturunuz oluşturunuz. (apply ve lambda yapılarını kullanınız)
# ------------------------------------------------------------------------------------------------------------------------------------------------------
# Task 16:  Write a function that returns 1 for those under 30, 0 for those above or equal to 30.
# Create a variable named age_flag in the titanic data set using the function you wrote. (use apply and lambda constructs)
#########################################
df["age_flag"] = df["age"].apply(lambda age: 1 if age < 30 else 0)
df["age_flag"]

#########################################
# Görev 17: Seaborn kütüphanesi içerisinden Tips veri setini tanımlayınız.
# Task 17:  Define the Tips dataset from the Seaborn library.
#########################################
df_tips = sns.load_dataset("tips")
df_tips.head()

#########################################
# Görev 18: Time değişkeninin kategorilerine (Dinner, Lunch) göre total_bill  değerlerinin toplamını, min, max ve ortalamasını bulunuz.
# Task 18:  Find the sum, min, max and average of the total_bill values according to the categories (Dinner, Lunch) of the Time variable.
#########################################
agg_funcs = {"total_bill":["sum","min","max","mean"]}
df_tips.groupby("time").agg(agg_funcs).T

#########################################
# Görev 19: Günlere ve time göre total_bill değerlerinin toplamını, min, max ve ortalamasını bulunuz.
# Task 19:  Find the sum, min, max and average of total_bill values by days and time.
#########################################
df_tips.groupby(["day","time"]).agg(agg_funcs).T

#########################################
# Görev 20: Lunch zamanına ve kadın müşterilere ait total_bill ve tip  değerlerinin day'e göre toplamını, min, max ve ortalamasını bulunuz.
# Task 20:  Find the sum, min, max and average of the total_bill and type values of the female customers, according to the day, for the lunch time.
#########################################
agg_funcs = {
    "total_bill":["sum","min","max","mean"],
    "tip":["sum","min","max","mean"]
}
df_tips.query("sex=='Female' & time=='Lunch'").groupby("day").agg(agg_funcs).T  # Easier to read
# df_tips.loc[(df_tips["sex"]=="Female") & (df_tips["time"] =="Lunch")].groupby("day").agg(agg_funcs).T

#########################################
# Görev 21: size'i 3'ten küçük, total_bill'i 10'dan büyük olan siparişlerin ortalaması nedir?
# Task 21:  What is the average of orders with size less than 3 and total_bill greater than 10?
#########################################
df_tips.query("size<3 & total_bill>10")["total_bill"].mean()

#########################################
# Görev 22: total_bill_tip_sum adında yeni bir değişken oluşturun. Her bir müşterinin ödediği totalbill ve tip in toplamını versin.
# Task 22:  Create a new variable called total_bill_tip_sum. Let him give the sum of the total bill and tip paid by each customer.
#########################################
total_bill_tip_sum = df_tips["total_bill"] + df_tips["tip"]

#########################################
# Görev 23: total_bill_tip_sum değişkenine göre büyükten küçüğe sıralayınız ve ilk 30 kişiyi yeni bir dataframe'e atayınız.
# Task 23:  Sort the total_bill_tip_sum variable from largest to smallest and assign the first 30 people to a new dataframe.
#########################################
df_tips["total_bill_tip_sum"] = total_bill_tip_sum
sorted_top_thirty = df_tips.sort_values(by="total_bill_tip_sum", ascending=False).head(30).reset_index()
sorted_top_thirty