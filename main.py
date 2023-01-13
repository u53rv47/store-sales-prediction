from sales.exception import SalesException
import sys


def main():
    a = 1/0


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        raise SalesException(e, sys)
