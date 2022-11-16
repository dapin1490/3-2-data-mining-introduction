"""
0. 필요한 라이브러리 import
1. 데이터 로드
2. 모델 로드
3. 모델 훈련
4. 데이터 샘플 테스트
5. 데이터 테스트
6. 모델 정확률 계산

* input과 output을 분리해야 한다
* 데이터 불러오고 전처리하는 게 복잡하지 이후 모델을 다루는 과정은 그렇지 않다고 한다(글쎄)
"""

import os

if not os.path.exists("scikit_learn/"):
	os.mkdir("scikit_learn/")
