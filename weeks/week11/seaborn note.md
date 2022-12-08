- sns.relplot : 점으로 요소들 간의 관계를 나타낼 수 있음  
  - ex 01 ~ 04
  - hue : 특정 속성을 기준으로 색 구분
  - style : 특정 속성을 기준으로 점/선(마커) 모양 구분
  - size : 특정 속성별로 마커 크기 구분
  - kind : 그래프 종류 지정. 선, 산점도 가능
  - facet_kws : Dictionary of other keyword arguments to pass to `FacetGrid`. `FacetGrid`로 전달되는 키워드 딕셔너리
  - `FacetGrid` : An object managing one or more subplots that correspond to conditional data subsets with convenient methods for batch-setting of axes attributes. 축 속성의 배치 설정을 위한 편리한 방법으로 조건부 데이터 하위 집합에 해당하는 하나 이상의 하위 그림을 관리하는 개체입니다.
  - `facet_kws=dict(sharex=False)` 이해한 내용 : x축 또는 y축을 서브플롯끼리 따로 쓸 수 있게 함
  - 참고 : <https://dsbook.tistory.com/52>
  - 참고 : <https://seaborn.pydata.org/generated/seaborn.relplot.html>

- `distplot` : 공식 문서에 의하면 deprecate되었다고 함. `histplot`과 `displot`으로 대체할 수 있으며, ex05.png 이미지와 같이 만들려면 `sns.histplot(data, kde=True)` 이렇게 쓰면 됨.
  - ex 05
  - 자세한 대체 방법은 <https://gist.github.com/mwaskom/de44147ed2974457ad6372750bbe5751> 참고

- `jointplot` : 산점도와 함께 테두리에 바 그래프 붙여줌.
  - ex 06

- `regplot` : 추세선과 산점도, 영역 그려줌
  - `ci` : 그려주는 영역의 범위. 회귀 추정치에 대한 신뢰구간의 크기.