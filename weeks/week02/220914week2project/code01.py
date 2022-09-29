# 넘파이 맛보기
from numpy import random

x = random.randn()

print(x)


# 클래스와 인스턴스
class MyPerson:
	i = 5

	def __init__(self, name, age):
		self.name = name
		self.age = age

	def asksomething(self):
		print("what is your name?")


cls = MyPerson("James", 20)
print(cls)
print(cls.name)
print(cls.age)
cls.asksomething()


# 모듈
from dummy1 import datamining as dm
# from : 폴더명
# import : 함수 가져오기
# as : 별칭 지정

print(dm.mul3(300))


# as가 왜 필요한가?
# 여러 개의 모듈을 쓸 때 이름이 겹칠 수 있음
from numpy import random  # 빠르지만 외부 라이브러리를 사용하기 때문에 추가 용량 필요
import random as pyrd  # 비교적 느리지만 내장 라이브러리 사용

print(random.randn())
print(pyrd.random())


# sys.path 예제
import sys

for p in sys.path:
	print(p)


# function
# 꼭 맨 위에 정의할 필요는 없으나 import와 함께 파일의 위쪽에 모아두는 것을 권장
# 파이썬 기본 내장 함수 : built-in function
def func1():
	print("this is a user defined function")


# nested function
print(sum([1, 2, 3, 4, 5]))
print(max(15, 6))


# 메소드와 함수
# 메소드 : 클래스 안에 포함됨
# 함수 : 클래스 밖에 정의됨
class MyMath:
	def add(self, a, b):
		return a + b


p1 = MyMath()
print(p1.add(3, 5))
