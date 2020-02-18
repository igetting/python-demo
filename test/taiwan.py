import time

import requests
from lxml import etree

headers = {
    'User-Agent': 'self-defind-user-agent'
}


def get_id():
    f1 = open('/Users/c/Desktop/ids.txt', 'w', encoding='utf8')
    url = 'https://tw.51240.com/'
    while True:
        res = requests.get(url, headers=headers)
        # print(res)
        html = etree.HTML(res.text)
        # print(html)
        ids = html.xpath('//div[@id="shuaxinshenfenzheng"]/table//table//tr[position()>1]/td[2]//text()')
        print(ids)
        for content in ids:
            print(content)
            f1.write(content + '\n')
        f1.flush()
        time.sleep(1)
    f1.close()


def get_name():
    f1 = open('/Users/c/Desktop/name.txt', 'w', encoding='utf8')
    url = 'https://tw.51240.com/'
    while True:
        res = requests.get(url, headers=headers)
        # print(res)
        html = etree.HTML(res.text)
        # print(html)
        ids = html.xpath('//div[@id="shuaxinshenfenzheng"]/table//table//tr[position()>1]/td[1]//text()')
        print(ids)
        for content in ids:
            print(content)
            f1.write(content + '\n')
        f1.flush()
        time.sleep(1)
    f1.close()


def get_address():
    url = 'https://www.p2peye.com/yhwd/p16/page'
    f1 = open('/Users/c/Desktop/address.txt', 'w', encoding='utf8')
    index = 1
    while True:
        res = requests.get(url + str(index), headers=headers)
        html = etree.HTML(res.text)
        data = html.xpath(
            '//div[@class="ui-branchs-warp"]//li//dd[@class="ui-branchs-address"]/span[@class="ui-branchs-info-sp"]//text()')
        print(data)
        for content in data:
            f1.write(content + "\n")
        f1.flush()
        index = index + 1
    f1.close()


def get_company():
    url = 'https://waizi.mingluji.com/taxonomy/term/825?page='
    f1 = open('/Users/c/Desktop/company.txt', 'w', encoding='utf8')
    index = 0
    while True:
        res = requests.get(url + str(index), headers=headers)
        time.sleep(1)
        html = etree.HTML(res.text)
        data = html.xpath('//div[@class="view-content"]/div/div/h2/a/text()')
        print(data)
        if len(data) == 0:
            break
        for content in data:
            print(content)
            f1.write(content + "\n")
        f1.flush()
        index = index + 1
    f1.close()


def get_company_one():
    url = 'https://www.71ab.com/province_32'
    f1 = open('/Users/c/Desktop/company_one.txt', 'w', encoding='utf8')
    index = 1
    while True:
        if index == 1:
            uri = url + '.html'
        else:
            uri = url + '_p' + str(index) + '.html'
        print(uri)
        res = requests.get(uri, headers=headers)
        html = etree.HTML(res.text)
        data = html.xpath('//strong[@class="px14"]/text()')
        print(data)
        if len(data) == 0:
            break
        for content in data:
            print(content)
            f1.write(content + "\n")
        f1.flush()
        index = index + 1
    f1.close()


if __name__ == "__main__":
    getattr(__import__(__name__), input('func name:').strip())()
