import pandas as pd
import matplotlib.pyplot as plt

file_route = r"weeks\week10+hw\homework\student_health_2.csv"
rawdata = pd.read_csv(file_route, header=0, encoding='euc-kr')  # 참고 : https://teddylee777.github.io/pandas/%EA%B3%B5%EA%B3%B5%EB%8D%B0%EC%9D%B4%ED%84%B0-%ED%95%9C%EA%B8%80%EA%B9%A8%EC%A7%90%ED%98%84%EC%83%81-%ED%95%B4%EA%B2%B0%EB%B0%A9%EB%B2%95

print(rawdata.head())
print(rawdata.info())

# with open(r"weeks\week10+hw\homework\result.txt", "w") as f:  # 참고 : https://wikidocs.net/26
#     f.write(str(rawdata.describe().transpose()))

"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 3569 entries, 0 to 3568
Data columns (total 25 columns):
 #   Column      Non-Null Count  Dtype
---  ------      --------------  -----
 0   ID          3569 non-null   object
 1   최종가중치       3569 non-null   float64
 2   학교ID        3569 non-null   object
 3   도시규모        3569 non-null   object
 4   도시규모별분석용    3569 non-null   object
 5   학년도         3569 non-null   int64
 6   광역시도        3569 non-null   object
 7   시도별         3569 non-null   object
 8   학교급별        3569 non-null   int64
 9   학교명         3569 non-null   object
 10  공학여부        3569 non-null   object
 11  학년          3569 non-null   int64
 12  반           3569 non-null   int64
 13  성별          3569 non-null   object
 14  건강검진일       3569 non-null   object
 15  키           3569 non-null   float64
 16  몸무게         3569 non-null   float64
 17  혈당식전mgdl    82 non-null     float64
 18  총콜레스테롤mgdl  82 non-null     float64
 19  ASTUL       82 non-null     float64
 20  ALTUL       82 non-null     float64
 21  혈색소gdl      0 non-null      float64
 22  간염검사        0 non-null      float64
 23  수축기         1125 non-null   float64
 24  이완기         1125 non-null   float64
dtypes: float64(11), int64(4), object(10)
memory usage: 697.2+ KB
None
"""